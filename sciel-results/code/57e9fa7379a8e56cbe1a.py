import numpy as np
import time
import scipy.optimize as opt
import itertools

N = 10
X_GRID, Y_GRID = np.meshgrid(np.arange(N), np.arange(N), indexing='xy')
X_FLAT = X_GRID.flatten()
Y_FLAT = Y_GRID.flatten()

# Pre-compute projection values
U_VALS = X_FLAT - Y_FLAT       # x - y
V_VALS = X_FLAT                # x
W_VALS = X_FLAT + Y_GRID.flatten()       # x + y
Z_VALS = X_FLAT + 2 * Y_GRID.flatten()   # x + 2y

# Unique values and inverse indices for bincount
U_UNIQ, U_INV = np.unique(U_VALS, return_inverse=True)
V_UNIQ, V_INV = np.unique(V_VALS, return_inverse=True)
W_UNIQ, W_INV = np.unique(W_VALS, return_inverse=True)
Z_UNIQ, Z_INV = np.unique(Z_VALS, return_inverse=True)

def _entropy_from_probs(probs):
    """Calculate Shannon entropy in bits, ignoring zeros."""
    # Filter out zeros to avoid log(0)
    probs = probs[probs > 1e-15]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))

def _compute_ratio(P_flat):
    """
    Compute the ratio D / M for a given flat probability vector.
    P_flat: 1D array of length 100.
    """
    P_flat = np.clip(P_flat, 1e-15, None)
    total = np.sum(P_flat)
    if total <= 1e-9:
        return 0.0
    P_flat = P_flat / total
    
    # Project probabilities
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

def _softmax(z):
    """
    Stable softmax transformation.
    z: 1D array of length 100.
    """
    z = np.clip(z, -500, 500) # Prevent overflow
    z_max = np.max(z)
    exp_z = np.exp(z - z_max)
    sum_exp_z = np.sum(exp_z)
    if sum_exp_z == 0:
        return np.ones(100) / 100.0
    return exp_z / sum_exp_z

def _objective(z):
    """
    Objective function for minimization (negative ratio).
    z: 1D array of length 100 (log-probabilities).
    """
    P_flat = _softmax(z)
    ratio = _compute_ratio(P_flat)
    return -ratio

def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Optimize the 100 cell probabilities directly using Nelder-Mead 
    with multiple restarts and adaptive steps.
    """
    start_time = time.time()
    deadline = start_time + 550  # 550s safety margin for 600s limit
    
    best_ratio = -np.inf
    best_z = np.zeros(100) # Start with uniform (z=0 -> softmax is uniform)
    
    # Initial evaluation
    best_ratio = _compute_ratio(_softmax(best_z))
    
    rng = np.random.default_rng(42)
    
    # List of initial strategies
    # 1. Uniform (z=0)
    # 2. Diagonal band (high prob on x=y)
    # 3. Anti-diagonal band (high prob on x+y=9)
    # 4. Sparse corners
    # 5. Random
    
    initial_guesses = []
    
    # Uniform
    initial_guesses.append(np.zeros(100))
    
    # Diagonal: P[x,y] high if x==y
    z_diag = np.full(100, -10.0)
    for i in range(10):
        z_diag[i*10+i] = 10.0 # High value on diagonal
    initial_guesses.append(z_diag)
    
    # Anti-diagonal: P[x,y] high if x+y=9
    z_anti = np.full(100, -10.0)
    for x in range(10):
        y = 9 - x
        z_anti[y*10+x] = 10.0
    initial_guesses.append(z_anti)
    
    # Sparse corners
    z_corners = np.full(100, -10.0)
    z_corners[0] = 5.0       # (0,0)
    z_corners[9] = 5.0       # (0,9)
    z_corners[90] = 5.0      # (9,0)
    z_corners[99] = 5.0      # (9,9)
    initial_guesses.append(z_corners)
    
    # Random initializations
    for _ in range(20):
        # Scale random noise to create varied structures
        scale = rng.uniform(1.0, 10.0)
        z_rand = rng.normal(0, scale, 100)
        initial_guesses.append(z_rand)
        
    # Specific structured guesses based on "band" intuition
    # High entropy on x-y means spread across many x-y values.
    # Low entropy on x, x+y, x+2y means concentrated in those projections.
    # Try to concentrate on a line where x-y varies but x is constant? No, x constant -> H_x=0.
    # If x is constant, H_x=0. H_{x+y} depends on y. H_{x+2y} depends on y.
    # If we pick x=5, and spread y uniformly:
    # P[5, y] = 1/10 for all y.
    # x-y = 5-y -> 10 unique values, uniform -> H_{x-y} = log2(10) ~ 3.32
    # x = 5 -> 1 value -> H_x = 0
    # x+y = 5+y -> 10 unique values, uniform -> H_{x+y} = log2(10) ~ 3.32
    # x+2y = 5+2y -> 10 unique values, uniform -> H_{x+2y} = log2(10) ~ 3.32
    # Ratio = 3.32 / 3.32 = 1.0.
    # We want D > M.
    # To get D > M, we need H_{x-y} to be large while H_x, H_{x+y}, H_{x+2y} are small.
    # H_x small -> x concentrated.
    # H_{x+y} small -> x+y concentrated.
    # H_{x+2y} small -> x+2y concentrated.
    # If x and x+y are concentrated, y is also concentrated (y = (x+y)-x).
    # If x, x+y are concentrated, then x-y is also concentrated?
    # Let's check linear dependence.
    # v1 = (1, 0) -> x
    # v2 = (1, 1) -> x+y
    # v3 = (1, 2) -> x+2y
    # u = (1, -1) -> x-y
    # u = 2*v2 - 3*v1? No. 2(x+y) - 3x = 2y - x.
    # u = v1 - v2 + 2x? No.
    # u = (1, -1).
    # v1 = (1, 0), v2 = (1, 1).
    # u = v1 - v2 + (0, -2)? No.
    # u = -v2 + 2*v1 - (0, 2)?
    # Actually, u, v1, v2 are linearly dependent in R2.
    # u = 2*v1 - v2? 2(1,0) - (1,1) = (1, -1). Yes!
    # So x-y = 2x - (x+y).
    # If x and x+y are concentrated, x-y is determined by them.
    # However, entropy is not linear.
    # H(A-B) can be larger than H(A) if B adds noise?
    # If A is concentrated and B is uniform, A-B is spread out.
    # So we want x concentrated, x+y spread out?
    # But M = max(H_x, H_{x+y}, H_{x+2y}).
    # If H_{x+y} is large, M is large.
    # We want H_{x-y} large, and H_x, H_{x+y}, H_{x+2y} small.
    # This seems difficult if they are linearly related.
    # Let's try to make x concentrated (H_x ~ 0).
    # Then x-y is basically -y (shifted). H_{x-y} ~ H_y.
    # x+y is basically y (shifted). H_{x+y} ~ H_y.
    # So H_{x-y} ~ H_{x+y}. Ratio ~ 1.
    # To get ratio > 1, we need H_{x-y} > H_{x+y} (assuming H_x is small).
    # This requires the distribution of y to have higher entropy in the -y projection than in the y projection?
    # Wait, the grid is discrete.
    # Let's try a 2D structure.
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
            
        try:
            # Nelder-Mead is derivative-free and handles non-smoothness better than L-BFGS for this
            res = opt.minimize(
                _objective, 
                guess, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if res.fun < -best_ratio: # -fun is ratio
                best_ratio = -res.fun
                best_z = res.x
                
        except Exception:
            continue
            
    # If time remains, do a second pass with smaller steps around the best solution
    if time.time() < deadline - 50:
        # Local refinement with tighter tolerances
        try:
            res = opt.minimize(
                _objective, 
                best_z, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res.fun < -best_ratio:
                best_ratio = -res.fun
                best_z = res.x
        except Exception:
            pass

    # Convert best_z to P
    P_final = _softmax(best_z).reshape((10, 10))
    
    # Final validation and normalization
    P_final = np.clip(P_final, 0, None)
    total = np.sum(P_final)
    if total <= 1e-9:
        P_final = np.ones((10, 10)) / 100.0
    else:
        P_final = P_final / total
        
    return P_final

# EVOLVE-BLOCK-END
