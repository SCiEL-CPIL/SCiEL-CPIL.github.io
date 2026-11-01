"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 110.0  # Leave 10s margin for the 120s limit

def _halton_sequence(n, base):
    """Generate n Halton sequence points in [0, 1]."""
    result = []
    for i in range(1, n + 1):
        f = 1.0
        r = 0.0
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        result.append(r)
    return result

def _initial_guesses(num_starts):
    """Generate diverse initial guesses using Halton sequences."""
    guesses = []
    # Base 2 for x, Base 3 for y to decorrelate
    x_coords = _halton_sequence(num_starts, 2)
    y_coords = _halton_sequence(num_starts, 3)
    
    # We need N circles per guess.
    # We can use the first N Halton points for a specific start, 
    # or generate a grid-like perturbation.
    # Better: Use a grid as the base structure, then perturb with Halton noise.
    # Or simply use Halton points for centers and small radii.
    
    # Let's use a hybrid: Grid structure is often a good basin of attraction.
    # We will create 5 different grid-based starts with different perturbations.
    
    side = int(np.ceil(np.sqrt(N))) # 6
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < N:
                cells.append(((i + 0.5) / side, (j + 0.5) / side))
    
    base_x = np.array([c[0] for c in cells])
    base_y = np.array([c[1] for c in cells])
    
    for k in range(num_starts):
        # Perturb the grid with Halton-like noise
        # We can use the k-th Halton value to scale the noise
        scale = 0.01 + 0.02 * (k / max(1, num_starts-1))
        
        # Generate random offsets using Halton for reproducibility and coverage
        # We need N x and N y offsets
        offsets_x = np.array(_halton_sequence(N, 2 + k)) - 0.5
        offsets_y = np.array(_halton_sequence(N, 3 + k)) - 0.5
        
        x_init = base_x + scale * offsets_x
        y_init = base_y + scale * offsets_y
        
        # Ensure within bounds
        x_init = np.clip(x_init, 0.01, 0.99)
        y_init = np.clip(y_init, 0.01, 0.99)
        
        # Initial radii: small enough to not overlap in the grid
        # Grid spacing is 1/side. Max radius approx 1/(2*side)
        r_init = np.full(N, 1.0 / (2.0 * side) - 0.005)
        
        # Flatten
        guess = np.concatenate([x_init, y_init, r_init])
        guesses.append(guess)
        
    return guesses

def _objective(x):
    """Objective function: minimize negative sum of radii."""
    r = x[2*N:]
    return -np.sum(r)

def _constraints(x):
    """List of constraint functions for SLSQP."""
    x_c = x[:N]
    y_c = x[N:2*N]
    r = x[2*N:]
    
    cons = []
    
    # Boundary constraints
    # x_i - r_i >= 0
    cons.append({'type': 'ineq', 'fun': lambda x: x[:N] - x[2*N:]})
    # 1 - x_i - r_i >= 0
    cons.append({'type': 'ineq', 'fun': lambda x: 1 - x[:N] - x[2*N:]})
    # y_i - r_i >= 0
    cons.append({'type': 'ineq', 'fun': lambda x: x[N:2*N] - x[2*N:]})
    # 1 - y_i - r_i >= 0
    cons.append({'type': 'ineq', 'fun': lambda x: 1 - x[N:2*N] - x[2*N:]})
    # r_i >= 0.001
    cons.append({'type': 'ineq', 'fun': lambda x: x[2*N:] - 0.001})
    
    # Non-overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    # This is O(N^2) constraints. 32 circles -> 496 constraints.
    # SLSQP can handle this, but it might be slow.
    # We can combine them into a single vector constraint function.
    
    def non_overlap(x):
        x_c = x[:N]
        y_c = x[N:2*N]
        r = x[2*N:]
        
        # Vectorized calculation of distances
        # Create NxN matrices
        dx = x_c[:, np.newaxis] - x_c[np.newaxis, :]
        dy = y_c[:, np.newaxis] - y_c[np.newaxis, :]
        dist_sq = dx**2 + dy**2
        
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        r_sum_sq = r_sum**2
        
        # We only care about i < j
        # The diagonal is 0, which is < 0 if r_i > 0, so we must exclude diagonal.
        # Mask for upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        # The constraint value is dist_sq - r_sum_sq
        # We need this to be >= 0 for all i<j
        vals = dist_sq[mask] - r_sum_sq[mask]
        return vals
    
    cons.append({'type': 'ineq', 'fun': non_overlap})
    
    return cons

def _is_valid(x):
    """Check if a solution is valid."""
    x_c = x[:N]
    y_c = x[N:2*N]
    r = x[2*N:]
    
    if np.any(r < 0):
        return False
    if np.any(x_c - r < -1e-6) or np.any(1 - x_c - r < -1e-6):
        return False
    if np.any(y_c - r < -1e-6) or np.any(1 - y_c - r < -1e-6):
        return False
        
    dx = x_c[:, np.newaxis] - x_c[np.newaxis, :]
    dy = y_c[:, np.newaxis] - y_c[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    vals = dist_sq[mask] - r_sum_sq[mask]
    
    if np.any(vals < -1e-5): # Allow small tolerance
        return False
        
    return True

def construct():
    best_x = None
    best_val = -np.inf
    
    # Generate initial guesses
    # We try 10 starts to balance quality and time
    num_starts = 10
    guesses = _initial_guesses(num_starts)
    
    bounds = [(0, 1)] * (2*N) + [(0.001, 0.5)] * N
    
    constraints = _constraints(None) # Just to get the structure, funs use x
    
    for i, x0 in enumerate(guesses):
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        try:
            res = minimize(_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 500, 'ftol': 1e-12})
            
            if res.success or res.status == 0:
                val = -res.fun
                if val > best_val:
                    # Verify validity
                    if _is_valid(res.x):
                        best_val = val
                        best_x = res.x
                    else:
                        # Try to repair or skip. SLSQP might report success but be slightly invalid.
                        # We can try a quick polish or just accept if violation is tiny.
                        if _is_valid(res.x) or np.all(_constraints(res.x)[5]['fun'](res.x) > -1e-4):
                             best_val = val
                             best_x = res.x
        except Exception as e:
            pass
            
    if best_x is None:
        # Fallback to seed if optimization failed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    # Final polish on the best result with tighter tolerance if time remains
    if time.time() - START_TIME < TIME_LIMIT - 5:
        try:
            res_final = minimize(_objective, best_x, method='SLSQP', bounds=bounds, constraints=constraints,
                                 options={'maxiter': 1000, 'ftol': 1e-14})
            if res_final.success:
                if _is_valid(res_final.x):
                    best_x = res_final.x
        except Exception:
            pass
            
    # Format output: (x, y, r)
    x_c = best_x[:N]
    y_c = best_x[N:2*N]
    r = best_x[2*N:]
    
    result = np.column_stack((x_c, y_c, r))
    return result

# EVOLVE-BLOCK-END
