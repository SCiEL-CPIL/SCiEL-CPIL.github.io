"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    """
    Fits a 1D regression model using a combination of:
    1. Chebyshev polynomial expansion with L2 regularization (global basis).
    2. Rational function approximation (P(x)/Q(x)) for potential pole-like structures.
    3. A fallback to high-degree polynomial if both fail.
    
    The algorithm selects the best model based on training error (since data is noiseless,
    low training error indicates good generalization for smooth functions, but we use
    regularization to prevent overfitting oscillations).
    """
    start_time = time.time()
    deadline = start_time + 290  # Leave 40s margin for 4 equations + overhead
    
    x = X_train[:, 0]
    y = y_train
    
    n = len(x)
    
    # Normalize x to [-1, 1] for stable Chebyshev evaluation
    x_min = np.min(x)
    x_max = np.max(x)
    if x_max == x_min:
        # Constant input
        return lambda X: np.full(len(X), np.mean(y))
    
    def normalize(x_val):
        return (2 * x_val - (x_max + x_min)) / (x_max - x_min)
    
    x_norm = normalize(x)
    
    # --- Model 1: Chebyshev Polynomial ---
    # We try degrees from 4 up to a max (e.g., 30).
    # We use L2 regularization on the coefficients to balance fit and smoothness.
    # Since data is noiseless, we can fit very well, but high degrees cause Runge phenomenon.
    # Regularization helps.
    
    best_cheb_pred = None
    best_cheb_score = np.inf
    best_cheb_degree = 0
    
    max_degree = 30
    
    for deg in range(4, max_degree + 1):
        if time.time() > deadline:
            break
            
        # Build Chebyshev basis matrix
        # T_k(x)
        T = np.zeros((n, deg + 1))
        T[:, 0] = 1.0
        if deg >= 1:
            T[:, 1] = x_norm
        for k in range(2, deg + 1):
            T[:, k] = 2 * x_norm * T[:, k-1] - T[:, k-2]
        
        # Ridge regression
        # Minimize ||T w - y||^2 + lambda * ||w||^2
        # (T^T T + lambda I) w = T^T y
        
        # Choose lambda based on degree. Higher degrees need more regularization.
        # A common heuristic is lambda ~ (deg * 1e-3) or similar.
        # Let's try a few lambdas or a single heuristic.
        # For noiseless data, we want exact fit but stable coefficients.
        # Let's use a small lambda to regularize high-frequency components.
        lambdas = [1e-6, 1e-4, 1e-2, 1e-1, 1.0]
        
        for lam in lambdas:
            if time.time() > deadline:
                break
            I = np.eye(deg + 1)
            # Don't regularize the intercept too much? Or just all.
            # Regularizing all is fine.
            A = T.T @ T + lam * I
            b = T.T @ y
            
            try:
                w = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                w = np.linalg.lstsq(A, b, rcond=None)[0]
                
            y_pred = T @ w
            mse = np.mean((y_pred - y) ** 2)
            
            # We want low MSE. Since data is noiseless, MSE should be near 0 for sufficient degree.
            # However, we also want to avoid overfitting which might hurt test points if they are slightly different?
            # No, test points are separate but from same distribution.
            # For noiseless data, a good fit on train should generalize if the function is smooth.
            # The risk is oscillation. Regularization mitigates this.
            
            if mse < best_cheb_score:
                best_cheb_score = mse
                best_cheb_pred = w
                best_cheb_degree = deg
                best_cheb_lam = lam
                
    # --- Model 2: Rational Function Approximation ---
    # P(x) / Q(x) where P, Q are polynomials of degree p, q.
    # y * Q(x) = P(x)
    # y * (q0 + q1 x + ... + qk x^k) = p0 + p1 x + ... + pm x^m
    # This is a linear system in p and q.
    # Let's try small degrees for p and q.
    
    best_rat_pred = None
    best_rat_score = np.inf
    best_rat_params = None
    
    # Try p=1..3, q=1..3
    for p_deg in range(1, 4):
        for q_deg in range(1, 4):
            if time.time() > deadline:
                break
                
            # Basis for P: 1, x, ..., x^p_deg
            # Basis for Q: 1, x, ..., x^q_deg
            # Equation: y * Q(x) - P(x) = 0
            # [y * Q_basis] - [P_basis] = 0
            # This is a homogeneous system if we include all coeffs.
            # To make it non-trivial, we can fix q0 = 1 (if Q(0) != 0) or use SVD.
            # Better: Linear Least Squares for P given Q? No, joint fit.
            # Let's construct the matrix A such that A @ [p; q] = 0.
            # Actually, we can solve for P and Q by minimizing ||y*Q - P||^2.
            # This is a linear problem if we fix the scale.
            # Let's use the approach: minimize || y * Q(x) - P(x) ||^2 subject to ||Q||^2 = 1.
            # This is an eigenvalue problem.
            # Let M be the matrix where M[i] = [y_i * Q_basis_i, -P_basis_i]
            # We want [p; q] such that M @ [p; q] = 0.
            # This is the null space of M.
            
            # Dimensions:
            # P has p_deg+1 coeffs.
            # Q has q_deg+1 coeffs.
            # Total unknowns: p_deg + q_deg + 2.
            
            # Construct M (n x (p_deg + q_deg + 2))
            # Columns 0..p_deg: -x^0, -x^1, ..., -x^p_deg
            # Columns p_deg+1..end: y*x^0, y*x^1, ..., y*x^q_deg
            
            M = np.zeros((n, p_deg + q_deg + 2))
            
            # P part
            for i in range(p_deg + 1):
                M[:, i] = -np.power(x_norm, i)
                
            # Q part
            for j in range(q_deg + 1):
                M[:, p_deg + 1 + j] = y * np.power(x_norm, j)
                
            # SVD to find null space
            try:
                U, s, Vt = np.linalg.svd(M, full_matrices=False)
                # The solution is the last row of Vt (corresponding to smallest singular value)
                # But we need to handle the sign and scale.
                coeffs = Vt[-1]
                
                p_coeffs = coeffs[:p_deg+1]
                q_coeffs = coeffs[p_deg+1:]
                
                # Normalize Q so that q0 = 1 (or max abs = 1) to avoid zero denominator issues
                # If q_coeffs are all zero, this model is invalid.
                if np.max(np.abs(q_coeffs)) < 1e-10:
                    continue
                    
                # Scale so that the max coeff of Q is 1? Or just use as is.
                # Prediction: P(x) / Q(x)
                # We need to ensure Q(x) is not zero on the domain.
                
                # Check for zeros in Q(x) on training data
                Q_vals = np.zeros(n)
                for j in range(q_deg + 1):
                    Q_vals += q_coeffs[j] * np.power(x_norm, j)
                    
                P_vals = np.zeros(n)
                for i in range(p_deg + 1):
                    P_vals += p_coeffs[i] * np.power(x_norm, i)
                    
                # If Q has zeros or near zeros, this model is unstable.
                if np.min(np.abs(Q_vals)) < 1e-6:
                    continue
                    
                y_pred = P_vals / Q_vals
                mse = np.mean((y_pred - y) ** 2)
                
                if mse < best_rat_score:
                    best_rat_score = mse
                    best_rat_params = (p_coeffs, q_coeffs)
                    
            except np.linalg.LinAlgError:
                continue

    # --- Selection ---
    # Compare best Chebyshev and best Rational model.
    # Since data is noiseless, both should have very low MSE.
    # However, rational models can be unstable if Q has roots near the domain.
    # Chebyshev is more robust.
    # We'll pick the one with lower MSE, but with a penalty if the rational model has small Q values.
    
    # Let's just pick Chebyshev if it's good enough (MSE < 1e-6), otherwise try Rational.
    # Or pick the minimum MSE.
    
    if best_cheb_pred is not None and best_rat_params is not None:
        if best_cheb_score < best_rat_score:
            use_cheb = True
        else:
            # Check stability of rational model
            # We already checked min|Q| > 1e-6 in the loop, so it should be safe.
            use_cheb = False
    elif best_cheb_pred is not None:
        use_cheb = True
    else:
        use_cheb = False # Fallback to something simple

    if use_cheb:
        # Reconstruct prediction function
        def predict(X):
            x_test = X[:, 0]
            x_norm_test = normalize(x_test)
            T_test = np.zeros((len(x_test), best_cheb_degree + 1))
            T_test[:, 0] = 1.0
            if best_cheb_degree >= 1:
                T_test[:, 1] = x_norm_test
            for k in range(2, best_cheb_degree + 1):
                T_test[:, k] = 2 * x_norm_test * T_test[:, k-1] - T_test[:, k-2]
            return T_test @ best_cheb_pred
    else:
        p_coeffs, q_coeffs = best_rat_params
        p_deg = len(p_coeffs) - 1
        q_deg = len(q_coeffs) - 1
        
        def predict(X):
            x_test = X[:, 0]
            x_norm_test = normalize(x_test)
            
            P_vals = np.zeros(len(x_test))
            for i in range(p_deg + 1):
                P_vals += p_coeffs[i] * np.power(x_norm_test, i)
                
            Q_vals = np.zeros(len(x_test))
            for j in range(q_deg + 1):
                Q_vals += q_coeffs[j] * np.power(x_norm_test, j)
                
            # Avoid division by zero
            Q_vals = np.where(np.abs(Q_vals) < 1e-10, 1e-10, Q_vals)
            
            return P_vals / Q_vals

    return predict
# EVOLVE-BLOCK-END
