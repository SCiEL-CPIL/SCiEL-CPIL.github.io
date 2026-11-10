import numpy as np
import time
import scipy.optimize as opt
import random

N = 10
X_GRID, Y_GRID = np.meshgrid(np.arange(N), np.arange(N), indexing='xy')
X_FLAT = X_GRID.flatten()
Y_FLAT = Y_GRID.flatten()

# Pre-compute projection values
U_VALS = X_FLAT - Y_FLAT       # x - y
V_VALS = X_FLAT                # x
W_VALS = X_FLAT + Y_FLAT       # x + y
Z_VALS = X_FLAT + 2 * Y_FLAT   # x + 2y

# Unique values and inverse indices for bincount
U_UNIQ, U_INV = np.unique(U_VALS, return_inverse=True)
V_UNIQ, V_INV = np.unique(V_VALS, return_inverse=True)
W_UNIQ, W_INV = np.unique(W_VALS, return_inverse=True)
Z_UNIQ, Z_INV = np.unique(Z_VALS, return_inverse=True)

def _entropy_from_probs(probs):
    """Calculate Shannon entropy in bits, ignoring zeros."""
    mask = probs > 1e-12
    if not np.any(mask):
        return 0.0
    p = probs[mask]
    p_sum = np.sum(p)
    if p_sum <= 1e-12:
        return 0.0
    p = p / p_sum
    return -np.sum(p * np.log2(p))

def _compute_ratio(P_flat):
    """
    Compute the ratio D / M for a given flat probability vector.
    P_flat: 1D array of length 100, non-negative, summing to 1.
    """
    P_flat = np.clip(P_flat, 0, None)
    total = np.sum(P_flat)
    if total <= 1e-9:
        return 0.0
    P_flat = P_flat / total
    
    p_u = np.bincount(U_INV, weights=P_flat, minlength=len(U_UNIQ))
    p_v = np.bincount(V_INV, weights=P_flat, minlength=len(V_UNIQ))
    p_w = np.bincount(W_INV, weights=P_flat, minlength=len(W_UNIQ))
    p_z = np.bincount(Z_INV, weights=P_flat, minlength=len(Z_UNIQ))
    
    h_u = _entropy_from_probs(p_u)
    h_v = _entropy_from_probs(p_v)
    h_w = _entropy_from_probs(p_w)
    h_z = _entropy_from_probs(p_z)
    
    D = h_u
    M = max(h_v, h_w, h_z)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def _field_to_prob(params):
    """
    Convert field parameters to a probability distribution.
    We use a sum of Gaussian blobs parameterized by:
    params = [cx1, cy1, s1, cx2, cy2, s2, ...]
    s is sigma (standard deviation).
    We also allow a uniform background component if we add a param, 
    but let's stick to pure Gaussians first, or maybe a mix.
    
    Actually, to ensure we can represent "sparse" or "diagonal" structures well,
    we might need to allow sharp peaks. Small sigma handles that.
    
    Let's define the field as:
    F(x, y) = sum_i exp(-((x-cx_i)^2 + (y-cy_i)^2) / (2*s_i^2))
    P(x, y) = F(x, y) / sum(F)
    
    To avoid division by zero or extremely small values, we add a tiny epsilon.
    """
    n_blobs = len(params) // 3
    field = np.zeros(100)
    for i in range(n_blobs):
        cx = params[i*3]
        cy = params[i*3+1]
        s = params[i*3+2]
        
        # Ensure sigma is not too small to cause numerical issues, 
        # but small enough to allow sharp features.
        s = max(s, 0.1)
        
        dist_sq = (X_FLAT - cx)**2 + (Y_FLAT - cy)**2
        field += np.exp(-dist_sq / (2 * s**2))
        
    field += 1e-9 # Numerical stability
    total = np.sum(field)
    if total < 1e-9:
        return np.ones(100) / 100.0
    return field / total

def _objective(params):
    """
    Objective function for minimization.
    We want to maximize ratio, so we minimize -ratio.
    """
    P = _field_to_prob(params)
    ratio = _compute_ratio(P)
    return -ratio

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin for 600s limit
    
    # We will try a few different numbers of blobs
    # and use differential evolution for each.
    
    best_P = np.ones(100) / 100.0
    best_ratio = _compute_ratio(best_P)
    
    # Initial guess: Uniform
    # Also try some structured initial guesses for DE to start near?
    # DE is global, so it doesn't strictly need a good start, but bounds matter.
    
    # Bounds for parameters:
    # cx, cy in [0, 9]
    # s in [0.2, 5.0] (0.2 is quite sharp, 5.0 is very broad)
    
    # Let's run DE for n_blobs = 1, 2, 3, 4
    # Time budget per DE run: ~100s each? 
    # 4 runs * 100s = 400s. Leaves 190s for refinement.
    
    rng = np.random.default_rng(42)
    
    for n_blobs in [1, 2, 3, 4]:
        if time.time() > deadline - 50: # Leave 50s for final steps
            break
            
        bounds = []
        for _ in range(n_blobs):
            bounds.append((0.0, 9.0)) # cx
            bounds.append((0.0, 9.0)) # cy
            bounds.append((0.2, 5.0)) # sigma
            
        # Differential Evolution settings
        # popsize: 15 * n_params is standard, but maybe too big for time.
        # Let's use popsize=10 or 15.
        # maxiter: depends on time.
        # tol: 1e-8
        
        # We want to use as much time as possible for the best configurations.
        # Let's allocate time dynamically.
        
        # Estimate time per function evaluation.
        # _field_to_prob + _compute_ratio is fast (< 1ms).
        # DE overhead is in Python loops.
        
        # Let's try a fixed number of iterations first, see if it fits.
        # If n_blobs=4, n_params=12. popsize=15 -> 180 evals/iter.
        # 100 iters -> 18000 evals. Should be fast (< 10s).
        # So we can afford more iterations or larger popsize.
        
        # Let's use maxiter=200, popsize=20.
        
        try:
            res = opt.differential_evolution(
                _objective, 
                bounds, 
                maxiter=300, 
                popsize=20, 
                tol=1e-8,
                seed=42,
                init='latinhypercube',
                polish=True # Uses L-BFGS-B at the end
            )
            
            if res.success or res.fun < best_ratio * -1: # Note: res.fun is negative ratio
                P_candidate = _field_to_prob(res.x)
                ratio_candidate = _compute_ratio(P_candidate)
                
                if ratio_candidate > best_ratio:
                    best_ratio = ratio_candidate
                    best_P = P_candidate
                    
        except Exception as e:
            pass # Ignore errors in specific DE runs
            
    # Refine the best found P using a more direct local optimizer on the 100-dim simplex?
    # Or just stick with the field parameters?
    # The field representation might be a bottleneck if the optimal P is not smooth.
    # However, the entropy ratio is a "smooth" objective in terms of P.
    # Let's do a quick coordinate descent on the 100-dim P to see if we can improve
    # the specific P found by DE.
    
    if time.time() < deadline - 10:
        P_refined = best_P.copy()
        # Simple coordinate descent: try moving mass from i to j
        # This is slow. Let's try Nelder-Mead on the 100-dim P?
        # Nelder-Mead is bad for high dim.
        
        # Let's try a few random restarts of DE with different seeds if time permits.
        # Or just run DE again with more iterations on the best n_blobs?
        
        # Actually, let's just run one more DE with n_blobs=5 if time allows, 
        # or increase iterations for n_blobs=3/4.
        
        # For now, let's just ensure the output is valid.
        
    P_out = best_P.reshape((10, 10))
    P_out = np.clip(P_out, 0, None)
    total = np.sum(P_out)
    if total <= 1e-9:
        P_out = np.ones((10, 10)) / 100.0
    else:
        P_out = P_out / total
        
    return P_out

# EVOLVE-BLOCK-END
