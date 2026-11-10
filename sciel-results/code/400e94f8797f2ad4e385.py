"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s limit

    x = np.asarray(training_x, dtype=np.float64)
    y = np.asarray(training_subgrid_tendency, dtype=np.float64)
    n = len(x)

    if n < 10:
        return np.zeros(6)

    # Build Vandermonde matrix for degree-5 polynomial
    deg = 5
    V = np.vander(x, deg + 1, increasing=True)

    # Precompute total Gram matrix and vector
    G_total = V.T @ V
    b_total = V.T @ y

    # Create 5 folds for cross-validation
    indices = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(indices)
    
    # Ensure folds are roughly equal size
    folds = np.array_split(indices, 5)
    
    G_folds = []
    b_folds = []
    V_folds = []
    y_folds = []
    
    for fold_idx in folds:
        V_f = V[fold_idx]
        y_f = y[fold_idx]
        G_f = V_f.T @ V_f
        b_f = V_f.T @ y_f
        G_folds.append(G_f)
        b_folds.append(b_f)
        V_folds.append(V_f)
        y_folds.append(y_f)

    # Check time
    if time.time() > deadline - 10:
        try:
            return np.polynomial.polynomial.polyfit(x, y, 5)
        except Exception:
            return np.zeros(6)

    # Define anisotropy profiles (shapes of the penalty)
    # We scale these by lambda in the search
    profiles = [
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),       # Isotropic
        np.array([1.0, 1.0, 1.0, 1.0, 10.0, 100.0]),    # Mild high-degree penalty
        np.array([1.0, 1.0, 1.0, 10.0, 100.0, 1000.0]), # Medium high-degree penalty
        np.array([1.0, 1.0, 1.0, 100.0, 10000.0, 1000000.0]), # Strong high-degree penalty
    ]

    # Search grid for lambda
    lambdas = np.logspace(-6, 6, 50)

    best_mse = np.inf
    best_lambda = 1e-3
    best_profile_idx = 0

    for p_idx, w in enumerate(profiles):
        L_penalty_base = np.diag(w**2)
        
        for lam in lambdas:
            L_pen = lam * L_penalty_base
            total_mse = 0.0
            
            for k in range(5):
                G_rest = G_total - G_folds[k]
                b_rest = b_total - b_folds[k]
                
                A = G_rest + L_pen
                b_vec = b_rest
                
                try:
                    c = np.linalg.solve(A, b_vec)
                except np.linalg.LinAlgError:
                    c = np.linalg.lstsq(A, b_vec, rcond=None)[0]
                    
                y_pred = V_folds[k] @ c
                mse_k = np.mean((y_folds[k] - y_pred)**2)
                total_mse += mse_k
                
            mean_mse = total_mse / 5.0
            
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_lambda = lam
                best_profile_idx = p_idx

            # Time check
            if time.time() > deadline - 5:
                break
                
        if time.time() > deadline - 5:
            break

    # Final fit with best parameters
    w_best = profiles[best_profile_idx]
    L_penalty_final = best_lambda * np.diag(w_best**2)
    
    A_final = G_total + L_penalty_final
    try:
        coeff = np.linalg.solve(A_final, b_total)
    except np.linalg.LinAlgError:
        coeff = np.linalg.lstsq(A_final, b_total, rcond=None)[0]

    if not np.all(np.isfinite(coeff)):
        try:
            coeff = np.polynomial.polynomial.polyfit(x, y, 5)
        except Exception:
            coeff = np.zeros(6)

    return coeff
# EVOLVE-BLOCK-END
