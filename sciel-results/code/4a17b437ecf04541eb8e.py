"""Strategy: Legendre polynomial basis fit with adaptive regularization and coefficient pruning."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct(training_x, training_subgrid_tendency):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    x = np.asarray(training_x, dtype=np.float64).ravel()
    y = np.asarray(training_subgrid_tendency, dtype=np.float64).ravel()
    
    n = len(x)
    if n < 6:
        return np.zeros(6)
    
    # Filter non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 6:
        return np.zeros(6)
    
    # Time budget check early
    if time.time() > deadline - 5:
        return np.zeros(6)

    # Define the mapping from original x to normalized t in [-1, 1]
    # This ensures Legendre polynomials are well-behaved and orthogonal over the support
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0
    x_center = (x_max + x_min) / 2.0
    
    # t = (x - center) / (range/2)
    t = (x - x_center) / (x_range / 2.0)
    
    # Construct Legendre design matrix P(x) = [L_0(t), L_1(t), ..., L_5(t)]
    # L_0 = 1
    # L_1 = t
    # L_2 = 0.5*(3t^2 - 1)
    # L_3 = 0.5*(5t^3 - 3t)
    # L_4 = (35t^4 - 30t^2 + 3)/8
    # L_5 = (63t^5 - 70t^3 + 15t)/8
    
    # Using recurrence for stability and generality, though explicit is fine for deg 5
    # Let's use explicit formulas for speed and clarity
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    t5 = t4 * t
    
    P = np.column_stack((
        np.ones_like(t),
        t,
        0.5 * (3.0 * t2 - 1.0),
        0.5 * (5.0 * t3 - 3.0 * t),
        (35.0 * t4 - 30.0 * t2 + 3.0) / 8.0,
        (63.0 * t5 - 70.0 * t3 + 15.0 * t) / 8.0
    ))
    
    # Cross-validation to select regularization lambda and potentially prune high-degree terms
    # We will test a range of lambdas and also consider truncating the polynomial degree
    # if higher degrees don't help validation error (effective model selection)
    
    np.random.seed(42)
    indices = np.random.permutation(n)
    val_frac = min(0.2, max(0.1, 2000.0 / n))
    val_size = int(n * val_frac)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    t_train = t[train_idx]
    y_train = y[train_idx]
    t_val = t[val_idx]
    y_val = y[val_idx]
    
    # Rebuild P for train/val
    P_train = _legendre_matrix(t_train)
    P_val = _legendre_matrix(t_val)
    
    # Candidate regularization strengths
    lambda_candidates = np.logspace(-6, 4, 20)
    
    best_lambda = 1e-3
    best_score = np.inf
    best_deg = 5
    
    # Try different maximum degrees (1 to 5) and lambdas
    # Note: If we truncate degree, we are effectively setting higher coefficients to 0
    # This is a structural simplification if the signal is low-order
    
    for max_deg in range(1, 6):
        if time.time() > deadline - 10:
            break
            
        P_tr_sub = P_train[:, :max_deg+1]
        P_va_sub = P_val[:, :max_deg+1]
        
        for lam in lambda_candidates:
            if time.time() > deadline - 10:
                break
            try:
                # Ridge regression in Legendre space
                A = P_tr_sub.T @ P_tr_sub + lam * np.eye(max_deg+1)
                b = P_tr_sub.T @ y_train
                coeff_leg = np.linalg.solve(A, b)
                
                if not np.all(np.isfinite(coeff_leg)):
                    continue
                
                # Evaluate on validation set
                y_pred = P_va_sub @ coeff_leg
                mse = np.mean((y_val - y_pred) ** 2)
                
                if mse < best_score:
                    best_score = mse
                    best_lambda = lam
                    best_deg = max_deg
            except Exception:
                continue
    
    # Refit on ALL data with the best configuration
    try:
        P_all = P[:, :best_deg+1]
        A = P_all.T @ P_all + best_lambda * np.eye(best_deg+1)
        b = P_all.T @ y
        coeff_leg = np.linalg.solve(A, b)
        
        if not np.all(np.isfinite(coeff_leg)):
            coeff_leg = np.zeros(best_deg+1)
    except Exception:
        coeff_leg = np.zeros(best_deg+1)
    
    # Pad coefficients to size 6 in Legendre space
    coeff_leg_full = np.zeros(6)
    coeff_leg_full[:best_deg+1] = coeff_leg
    
    # Transform Legendre coefficients back to monomial coefficients in original x space
    # We need c_monomial such that sum(c_monomial[i] * x^i) = sum(coeff_leg[j] * L_j(t))
    # where t = (x - center) / half_range
    
    # Let's use the basis transformation matrix.
    # We know L_j(t) is a polynomial in t of degree j.
    # t = (x - c) / h
    # We can evaluate the Legendre polynomial at specific points to build the transformation?
    # Or use the explicit formulas derived earlier.
    
    # Let's build the transformation matrix M such that L(t) = M @ x_monomial
    # Actually, it's easier to express each L_j(t) in terms of x.
    # L_j((x-c)/h) = sum_{k=0}^j M_{j,k} x^k
    
    # We can compute M by evaluating L_j at basis points? No, let's use the explicit polynomials.
    # L_0(t) = 1
    # L_1(t) = t = (x-c)/h
    # L_2(t) = 0.5(3t^2-1) = 0.5(3((x-c)/h)^2 - 1)
    # ...
    
    # This manual expansion is error-prone. Let's use a numerical integration/projection approach?
    # No, let's just construct the matrix by evaluating the Legendre polynomials in terms of x.
    
    # Alternative: Use the fact that we can convert Legendre coeffs to Monomial coeffs
    # by solving a linear system on a set of points.
    # Let's pick 6 distinct points in the domain, evaluate L_j(t_i) and x_i^k, and solve.
    
    # Points in x-space
    x_points = np.linspace(x_min + 1e-6, x_max - 1e-6, 6)
    t_points = (x_points - x_center) / (x_range / 2.0)
    
    # Matrix L_pts: 6x6, L_pts[i, j] = L_j(t_points[i])
    L_pts = _legendre_matrix(t_points)
    
    # Matrix V_pts: 6x6, V_pts[i, k] = x_points[i]^k
    V_pts = np.vander(x_points, N=6, increasing=True)
    
    # We want coeff_monomial such that:
    # V_pts @ coeff_monomial = L_pts @ coeff_leg_full
    # coeff_monomial = inv(V_pts) @ L_pts @ coeff_leg_full
    
    try:
        # V_pts is Vandermonde, should be invertible for distinct points
        # Use solve for stability
        # V_pts @ c_mono = L_pts @ c_leg
        # c_mono = V_pts^{-1} @ (L_pts @ c_leg)
        
        rhs = L_pts @ coeff_leg_full
        coeff_monomial = np.linalg.solve(V_pts, rhs)
        
        if not np.all(np.isfinite(coeff_monomial)):
            return np.zeros(6)
            
        return coeff_monomial
        
    except Exception:
        # Fallback: if solve fails, return zeros or a simple fit
        # This shouldn't happen with distinct points
        return np.zeros(6)

def _legendre_matrix(t):
    """
    Construct the Legendre polynomial design matrix for input t.
    Returns array of shape (len(t), 6) containing L_0(t) to L_5(t).
    """
    t = np.asarray(t)
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    t5 = t4 * t
    
    L0 = np.ones_like(t)
    L1 = t
    L2 = 0.5 * (3.0 * t2 - 1.0)
    L3 = 0.5 * (5.0 * t3 - 3.0 * t)
    L4 = (35.0 * t4 - 30.0 * t2 + 3.0) / 8.0
    L5 = (63.0 * t5 - 70.0 * t3 + 15.0 * t) / 8.0
    
    return np.column_stack((L0, L1, L2, L3, L4, L5))

# EVOLVE-BLOCK-END
