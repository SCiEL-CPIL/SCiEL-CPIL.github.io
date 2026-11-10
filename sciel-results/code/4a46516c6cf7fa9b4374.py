import numpy as np
import time
import random
from scipy.optimize import differential_evolution

N = 10
X_GRID, Y_GRID = np.meshgrid(np.arange(N), np.arange(N), indexing='xy')
X_FLAT = X_GRID.flatten()
Y_FLAT = Y_GRID.flatten()

# Pre-compute projection values for the full 100x100 grid
# These are used to verify or if we fallback, but the main strategy uses product form
U_VALS = X_FLAT - Y_GRID.flatten()       # x - y
V_VALS = X_FLAT                          # x
W_VALS = X_FLAT + Y_GRID.flatten()       # x + y
Z_VALS = X_FLAT + 2 * Y_GRID.flatten()   # x + 2y

# For product form P[x,y] = R[x] * C[y] / Z (normalized)
# We need to compute entropies of:
# 1. x - y
# 2. x
# 3. x + y
# 4. x + 2y

# Since P is product form, the marginal of x is R (normalized) and y is C (normalized).
# H_x = H(R)
# H_y = H(C)
# The joint distribution of (x,y) is R[x]*C[y].
# We need H_{x-y}, H_{x+y}, H_{x+2y}.

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

def _compute_ratio_product(R, C):
    """
    Compute the ratio D / M for a product-form distribution P[x,y] proportional to R[x]*C[y].
    R: array of length 10 (unnormalized weights for x)
    C: array of length 10 (unnormalized weights for y)
    """
    R = np.clip(R, 0, None)
    C = np.clip(C, 0, None)
    sum_R = np.sum(R)
    sum_C = np.sum(C)
    if sum_R <= 1e-12 or sum_C <= 1e-12:
        return 0.0
    
    R_norm = R / sum_R
    C_norm = C / sum_C
    
    # H_x is entropy of R_norm
    h_x = _entropy_from_probs(R_norm)
    
    # We need distributions for:
    # 1. x - y
    # 2. x + y
    # 3. x + 2y
    
    # Construct the joint probability matrix P[x,y] = R_norm[x] * C_norm[y]
    P_joint = np.outer(R_norm, C_norm)
    
    # 1. Distribution of x - y
    # Values of x-y range from -9 to 9
    # We can sum P_joint[x,y] where x-y = k
    # Efficient way: Convolution or direct loop
    # Since N=10, direct loop is fine, but let's use a structured approach
    
    # Helper to get distribution of x + c*y
    def get_dist(c):
        # x + c*y
        # Min value: 0 + c*0 = 0 (if c>=0) or 0 + c*9 (if c<0)
        # Max value: 9 + c*9 (if c>=0) or 9 + c*0 (if c<0)
        # Let's just iterate over all x,y and bin them
        bins = {}
        for x in range(10):
            for y in range(10):
                val = x + c * y
                if val in bins:
                    bins[val] += P_joint[x, y]
                else:
                    bins[val] = P_joint[x, y]
        if not bins:
            return np.array([0.0])
        max_val = max(bins.keys())
        min_val = min(bins.keys())
        width = max_val - min_val + 1
        arr = np.zeros(width)
        for val, prob in bins.items():
            arr[val - min_val] = prob
        return arr

    dist_u = get_dist(-1)  # x - y
    dist_w = get_dist(1)   # x + y
    dist_z = get_dist(2)   # x + 2y
    
    h_u = _entropy_from_probs(dist_u)
    h_w = _entropy_from_probs(dist_w)
    h_z = _entropy_from_probs(dist_z)
    
    D = h_u
    M = max(h_x, h_w, h_z)
    
    if M <= 1e-9:
        return 0.0
        
    return D / M

def _objective(params):
    """
    Objective function for differential evolution.
    params: 20 floats (10 for R, 10 for C)
    Returns negative ratio (since DE minimizes).
    """
    R = params[:10]
    C = params[10:20]
    ratio = _compute_ratio_product(R, C)
    return -ratio

def arithmetic_kakeya_distribution() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin for 600s limit
    
    # Initial best is uniform
    best_ratio = _compute_ratio_product(np.ones(10), np.ones(10))
    best_params = np.ones(20)
    
    # Strategy 1: Differential Evolution on the 20-dimensional product space
    # Bounds: [0, 1] for each weight. Normalization happens inside _compute_ratio_product
    bounds = [(0, 1) for _ in range(20)]
    
    try:
        # Run DE for a reasonable number of generations
        # maxiter=100, popsize=20 -> ~2000 evaluations
        # _compute_ratio_product is fast (10x10 matrix ops)
        res = differential_evolution(
            _objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=20, 
            tol=1e-8,
            init='sobol'
        )
        
        if res.fun < -best_ratio:
            best_ratio = -res.fun
            best_params = res.x
            
    except Exception as e:
        pass # Fallback if DE fails
    
    # Strategy 2: Local refinement using Nelder-Mead on the product space
    # This helps polish the result from DE
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            
            # Use best_params as start
            res_nm = minimize(
                _objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res_nm.fun < -best_ratio:
                best_ratio = -res_nm.fun
                best_params = res_nm.x
                
        except Exception as e:
            pass
    
    # Strategy 3: Random restarts with small DE runs if time permits
    # This helps escape local optima in the 20-dim space
    if time.time() < deadline - 30:
        rng = np.random.default_rng(42)
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random initial population
            init_pop = rng.uniform(0, 1, size=(20, 20))
            
            try:
                res_re = differential_evolution(
                    _objective, 
                    bounds, 
                    seed=42+restart, 
                    maxiter=50, 
                    popsize=20, 
                    tol=1e-6,
                    init=init_pop
                )
                
                if res_re.fun < -best_ratio:
                    best_ratio = -res_re.fun
                    best_params = res_re.x
                    
            except Exception as e:
                pass
    
    # Convert best_params (20 floats) to 10x10 P
    R = np.clip(best_params[:10], 0, None)
    C = np.clip(best_params[10:20], 0, None)
    
    # Normalize R and C
    sum_R = np.sum(R)
    sum_C = np.sum(C)
    
    if sum_R <= 1e-9 or sum_C <= 1e-9:
        # Fallback to uniform if something went wrong
        P_out = np.ones((10, 10)) / 100.0
    else:
        R_norm = R / sum_R
        C_norm = C / sum_C
        P_out = np.outer(R_norm, C_norm)
        
    # Ensure shape and validity
    P_out = np.clip(P_out, 0, None)
    total = np.sum(P_out)
    if total <= 1e-9:
        P_out = np.ones((10, 10)) / 100.0
    else:
        P_out = P_out / total
        
    return P_out

# EVOLVE-BLOCK-END
