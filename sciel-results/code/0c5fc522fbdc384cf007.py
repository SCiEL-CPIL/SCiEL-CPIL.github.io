"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def fit(X_train, y_train):
    t0 = time.time()
    DEADLINE = 330.0 - 15.0  # safety margin before the 330s kill

    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float).ravel()
    n, d = X.shape

    # --- Standardize inputs to unit scale for stable kernel math ---
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    # Precompute squared distances for RBF
    diff = Xs[:, None, :] - Xs[None, :, :]
    sq = np.sum(diff ** 2, axis=2)
    del diff

    # Precompute polynomial kernel matrix (degree 2)
    # K_poly = (1 + Xs @ Xs.T)^2
    Xs_dot = Xs @ Xs.T
    K_poly = (1.0 + Xs_dot) ** 2

    # Precompute periodic kernel matrix
    # K_periodic = exp(-0.5 * (2*pi/sigma)^2 * sin^2(pi*|x-y|/period))
    # We'll use a simplified version with fixed period for speed, optimizing sigma
    # Actually, let's just use a fixed periodic structure and optimize the length scale
    # For simplicity and speed, we'll use a composite of RBF and Poly, and maybe add a linear term
    # Let's stick to RBF + Poly for robustness, as periodic might be risky without knowing the function
    
    # Define composite kernel function
    def compute_kernel(alpha_rbf, alpha_poly, l_rbf):
        K_rbf = np.exp(-0.5 * (l_rbf ** 2) * sq)
        # Scale polynomial kernel to have similar magnitude as RBF
        K = alpha_rbf * K_rbf + alpha_poly * K_poly
        return K

    # Grid search over hyperparameters
    # l_rbf: length scale for RBF
    # alpha_rbf: weight for RBF
    # alpha_poly: weight for Poly
    # We'll fix alpha_poly = 1.0 and optimize alpha_rbf and l_rbf, or use a simpler approach
    # Let's try a small grid search over l_rbf and a fixed regularization, using the composite kernel
    # K = K_rbf + K_poly
    
    best_alpha = None
    best_l_rbf = None
    best_mse = np.inf
    lam = 1e-3  # Regularization

    # Grid for l_rbf
    l_rbf_vals = [0.2, 0.5, 1.0, 2.0, 5.0]
    
    for l_rbf in l_rbf_vals:
        if time.time() > DEADLINE:
            break
        K_rbf = np.exp(-0.5 * (l_rbf ** 2) * sq)
        K = K_rbf + K_poly  # Composite kernel
        A = K + lam * np.eye(n)
        try:
            alpha = np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            continue
        pred = K @ alpha
        mse = float(np.mean((pred - y) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_l_rbf = l_rbf

    # Fallback: if nothing was solved, use a simple mean predictor
    if best_alpha is None:
        mean_y = float(np.mean(y))
        return lambda X: np.full(len(np.asarray(X)), mean_y, dtype=float)

    l_rbf = best_l_rbf
    alpha = best_alpha

    def predict(Xq):
        Xqs = (np.asarray(Xq, dtype=float) - mu) / sd
        # Compute RBF part
        diff_q = Xqs[:, None, :] - Xs[None, :, :]
        sq_q = np.sum(diff_q ** 2, axis=2)
        K_rbf_q = np.exp(-0.5 * (l_rbf ** 2) * sq_q)
        # Compute Poly part
        K_poly_q = (1.0 + Xqs @ Xs.T) ** 2
        K_q = K_rbf_q + K_poly_q
        return K_q @ alpha

    return predict
# EVOLVE-BLOCK-END
