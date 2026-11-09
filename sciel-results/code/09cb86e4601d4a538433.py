import numpy as np
import time
import scipy.optimize as opt

# Pre-compute the index maps for projections
# x is column (0-9), y is row (0-9)
# P shape is (10, 10)
x_indices = np.arange(10)
y_indices = np.arange(10)
# Meshgrid with indexing='xy' means X[i, j] = x_indices[j], Y[i, j] = y_indices[i]
X_grid, Y_grid = np.meshgrid(x_indices, y_indices, indexing='xy')

# Values for each projection
D_vals = (X_grid - Y_grid)       # x - y
H0_vals = X_grid                 # x
H1_vals = (X_grid + Y_grid)      # x + y
H2_vals = (X_grid + 2 * Y_grid)  # x + 2y

# Unique values and inverse indices for bincount
D_uniq, D_inv = np.unique(D_vals, return_inverse=True)
H0_uniq, H0_inv = np.unique(H0_vals, return_inverse=True)
H1_uniq, H1_inv = np.unique(H1_vals, return_inverse=True)
H2_uniq, H2_inv = np.unique(H2_vals, return_inverse=True)

# Flatten inverse indices for use with bincount
D_inv_flat = D_inv.ravel()
H0_inv_flat = H0_inv.ravel()
H1_inv_flat = H1_inv.ravel()
H2_inv_flat = H2_inv.ravel()

# Number of unique bins
D_bins = len(D_uniq)
H0_bins = len(H0_uniq)
H1_bins = len(H1_uniq)
H2_bins = len(H2_uniq)

def get_projection_entropy_and_grad(P, inv_flat, n_bins):
    """
    Computes entropy of the projection and its gradient with respect to P.
    P: flattened probability vector (size 100)
    inv_flat: inverse indices for binning (size 100)
    n_bins: number of unique values
    Returns: entropy, gradient (size 100)
    """
    # Sum probabilities into bins
    q = np.bincount(inv_flat, weights=P, minlength=n_bins)
    
    # Mask non-zero
    mask = q > 0
    q_nonzero = q[mask]
    
    if len(q_nonzero) == 0:
        return 0.0, np.zeros_like(P)
        
    # Entropy
    H = -np.sum(q_nonzero * np.log2(q_nonzero))
    
    # Gradient of H with respect to q
    # dH/dq_j = -log2(q_j) - 1/ln(2)
    grad_q = np.zeros(n_bins)
    grad_q[mask] = -np.log2(q_nonzero) - 1.0 / np.log(2.0)
    
    # Gradient with respect to P
    # dH/dP_i = grad_q[bin(i)]
    grad_P = grad_q[inv_flat]
    
    return H, grad_P

def objective_and_grad(z):
    """
    z is the flattened vector of probabilities P (size 100).
    We assume z sums to 1 during optimization or we work in unconstrained space?
    SLSQP can handle constraints. Let's pass P directly.
    """
    P = z
    
    # Calculate D (H_minus_1)
    D_val, D_grad = get_projection_entropy_and_grad(P, D_inv_flat, D_bins)
    
    # Calculate H0, H1, H2
    H0_val, H0_grad = get_projection_entropy_and_grad(P, H0_inv_flat, H0_bins)
    H1_val, H1_grad = get_projection_entropy_and_grad(P, H1_inv_flat, H1_bins)
    H2_val, H2_grad = get_projection_entropy_and_grad(P, H2_inv_flat, H2_bins)
    
    H012 = np.array([H0_val, H1_val, H2_val])
    grads_012 = [H0_grad, H1_grad, H2_grad]
    
    M = np.max(H012)
    
    if M < 1e-9:
        # Invalid, return large penalty
        return -1e9, np.zeros_like(P)
        
    ratio = D_val / M
    
    # Gradient of ratio = (M * dD - D * dM) / M^2
    # dM is the gradient of the argmax term
    idx_max = np.argmax(H012)
    M_grad = grads_012[idx_max]
    
    grad_ratio = (M * D_grad - D_val * M_grad) / (M * M)
    
    # We want to maximize, so we minimize -ratio
    obj = -ratio
    grad = -grad_ratio
    
    return obj, grad

def solve_kakeya():
    n = 10
    start_time = time.time()
    timeout = 600  # seconds
    safety_margin = 30
    
    best_P = np.ones((n, n)) / (n * n)
    best_score = 0.0
    
    # Initializations
    inits = []
    # 1. Uniform
    inits.append(np.ones((n, n)) / (n * n))
    
    # 2. Random
    for _ in range(5):
        p = np.random.rand(n, n)
        p /= p.sum()
        inits.append(p)
        
    # 3. Heuristic: Focus on diagonal x=y (high D, low H0? No, D=0)
    # Focus on anti-diagonal x+y=const?
    # Let's try a distribution that spreads x-y but concentrates x.
    # x is low entropy if x is constant.
    # If x is constant, D = H(-y). H0 = 0. H1 = H(x+y)=H(y). H2 = H(x+2y)=H(y).
    # Ratio = H(y)/H(y) = 1.
    # To beat 1, we need D > H1 and D > H2.
    # This is non-trivial.
    
    for P_init in inits:
        P_flat = P_init.ravel()
        
        # Check time
        if time.time() - start_time > timeout - safety_margin:
            break
            
        # Constraints: sum(P) = 1, P >= 0
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0,
                        'jac': lambda x: np.ones_like(x)}]
        bounds = [(0.0, 1.0) for _ in range(n * n)]
        
        try:
            result = opt.minimize(
                objective_and_grad,
                P_flat,
                jac=True,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            P_opt = result.x.reshape(n, n)
            
            # Clip and normalize to be safe
            P_opt = np.clip(P_opt, 0, None)
            s = P_opt.sum()
            if s > 1e-9:
                P_opt /= s
            else:
                continue
                
            # Evaluate raw score
            P_flat_eval = P_opt.ravel()
            D_val, _ = get_projection_entropy_and_grad(P_flat_eval, D_inv_flat, D_bins)
            H0_val, _ = get_projection_entropy_and_grad(P_flat_eval, H0_inv_flat, H0_bins)
            H1_val, _ = get_projection_entropy_and_grad(P_flat_eval, H1_inv_flat, H1_bins)
            H2_val, _ = get_projection_entropy_and_grad(P_flat_eval, H2_inv_flat, H2_bins)
            
            M = max(H0_val, H1_val, H2_val)
            if M > 1e-9:
                score = D_val / M
                if score > best_score:
                    best_score = score
                    best_P = P_opt
                    
        except Exception as e:
            pass # Ignore optimization errors for this init
            
    # Final safety check and return
    # Ensure best_P is valid
    if best_P.sum() <= 1e-9:
        best_P = np.ones((n, n)) / (n * n)
        
    best_P = np.clip(best_P, 0, None)
    s = best_P.sum()
    if s > 0:
        best_P /= s
        
    return best_P

def arithmetic_kakeya_distribution() -> np.ndarray:
    return solve_kakeya()
