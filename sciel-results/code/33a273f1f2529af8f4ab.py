"""Circle packing in unit square via parallel tempering Monte Carlo (2D lattice gas)."""
import numpy as np
from scipy.spatial.distance import cdist
import random
import time

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
import random
import time

N = 26

def _check_feasible(x, y, r, tol=1e-9):
    """Vectorized feasibility check."""
    n = len(r)
    # Boundary check
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    # Pairwise distance check using cdist
    if n > 1:
        pts = np.column_stack([x, y])
        dists = cdist(pts, pts)
        req = r[:, None] + r[None, :]
        # Check all pairs i < j
        for i in range(n):
            for j in range(i + 1, n):
                if dists[i, j] < req[i, j] - tol:
                    return False
    return True

def _penalty(x, y, r, beta):
    """Compute penalty for constraint violations (vectorized)."""
    n = len(r)
    pen = 0.0
    # Boundary penalties
    pen += beta * np.sum(np.maximum(0, r - x)**2)
    pen += beta * np.sum(np.maximum(0, r - (1 - x))**2)
    pen += beta * np.sum(np.maximum(0, r - y)**2)
    pen += beta * np.sum(np.maximum(0, r - (1 - y))**2)
    # Pairwise overlap penalties
    if n > 1:
        pts = np.column_stack([x, y])
        dists = cdist(pts, pts)
        req = r[:, None] + r[None, :]
        for i in range(n):
            for j in range(i + 1, n):
                overlap = req[i, j] - dists[i, j]
                if overlap > 0:
                    pen += beta * overlap * overlap
    return pen

def _score(x, y, r, beta):
    """Objective = sum(r) - penalty."""
    return np.sum(r) - _penalty(x, y, r, beta)

def _grid_init(n_cols=None, n_rows=None):
    """Initialize with a grid-like feasible configuration."""
    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(N)))
    if n_rows is None:
        n_rows = int(np.ceil(N / n_cols))
    
    x = np.zeros(N)
    y = np.zeros(N)
    r = np.zeros(N)
    
    idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if idx >= N:
                break
            x[idx] = (j + 0.5) / n_cols
            y[idx] = (i + 0.5) / n_rows
            r[idx] = min(0.5 / n_cols, 0.5 / n_rows) * 0.95
            idx += 1
    
    # Ensure feasibility
    if not _check_feasible(x, y, r):
        r = r * 0.9
        if not _check_feasible(x, y, r):
            r = r * 0.85
    return x, y, r

def _random_feasible_init(seed=None):
    """Generate a random feasible initial configuration."""
    rng = np.random.RandomState(seed)
    for attempt in range(200):
        x = rng.uniform(0.05, 0.95, N)
        y = rng.uniform(0.05, 0.95, N)
        r = rng.uniform(0.01, 0.06, N)
        if _check_feasible(x, y, r):
            return x, y, r
    # Fallback: grid
    return _grid_init()

def _pt_mc(x_init, y_init, r_init, num_replicas=12, n_iter=8000,
           T_low=0.01, T_high=1.0, beta=200.0,
           sigma_pos=0.02, sigma_r=0.003, seed=42):
    """Parallel tempering Monte Carlo."""
    rng = np.random.RandomState(seed)
    
    # Initialize replicas with perturbations of initial state
    states = []
    scores = []
    for i in range(num_replicas):
        x = x_init.copy() + rng.normal(0, 0.01 * (i + 1), N)
        y = y_init.copy() + rng.normal(0, 0.01 * (i + 1), N)
        r = r_init.copy() + rng.normal(0, 0.002 * (i + 1), N)
        r = np.clip(r, 0.005, 0.4)
        x = np.clip(x, 0.01, 0.99)
        y = np.clip(y, 0.01, 0.99)
        states.append((x, y, r))
        scores.append(_score(x, y, r, beta))
    
    # Temperatures
    temps = np.geomspace(T_low, T_high, num_replicas)
    
    best_state = None
    best_feasible_obj = -np.inf
    
    # Check initial states for feasibility
    for i in range(num_replicas):
        x, y, r = states[i]
        if _check_feasible(x, y, r):
            obj = np.sum(r)
            if obj > best_feasible_obj:
                best_feasible_obj = obj
                best_state = (x.copy(), y.copy(), r.copy())
    
    start_time = time.time()
    time_limit = 90.0  # Leave time for post-processing
    
    for it in range(n_iter):
        # Check time
        if time.time() - start_time > time_limit:
            break
        
        for rep in range(num_replicas):
            x, y, r = states[rep]
            idx = rng.randint(0, N)
            
            # Propose move
            x_new = x.copy()
            y_new = y.copy()
            r_new = r.copy()
            
            # Position move
            x_new[idx] += rng.normal(0, sigma_pos)
            y_new[idx] += rng.normal(0, sigma_pos)
            
            # Radius move
            r_new[idx] += rng.normal(0, sigma_r)
            r_new[idx] = np.clip(r_new[idx], 0.005, 0.4)
            
            # Clamp to bounds
            x_new[idx] = np.clip(x_new[idx], 0.01, 0.99)
            y_new[idx] = np.clip(y_new[idx], 0.01, 0.99)
            
            old_score = scores[rep]
            new_score = _score(x_new, y_new, r_new, beta)
            
            # Metropolis acceptance
            delta = new_score - old_score
            if delta >= 0 or rng.random() < np.exp(delta / temps[rep]):
                states[rep] = (x_new, y_new, r_new)
                scores[rep] = new_score
                
                # Track best feasible
                if _check_feasible(x_new, y_new, r_new):
                    obj = np.sum(r_new)
                    if obj > best_feasible_obj:
                        best_feasible_obj = obj
                        best_state = (x_new.copy(), y_new.copy(), r_new.copy())
        
        # Swap adjacent replicas
        for rep in range(num_replicas - 1):
            a, b = rep, rep + 1
            x_a, y_a, r_a = states[a]
            x_b, y_b, r_b = states[b]
            s_a, s_b = scores[a], scores[b]
            T_a, T_b = temps[a], temps[b]
            
            delta = (s_b - s_a) * (1.0 / T_a - 1.0 / T_b)
            if delta >= 0 or rng.random() < np.exp(delta):
                states[a] = (x_b, y_b, r_b)
                states[b] = (x_a, y_a, r_a)
                scores[a] = s_b
                scores[b] = s_a
    
    return best_state, best_feasible_obj

def _anneal_radii(x, y, r, max_rounds=500, tol=1e-9):
    """Simulated annealing to maximize sum of radii while maintaining feasibility."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    
    current_obj = np.sum(r)
    T = 0.01
    T_min = 0.0001
    
    for round_idx in range(max_rounds):
        improved = False
        # Try increasing each radius
        for i in range(N):
            for dr in [0.002, 0.001, 0.0005, 0.0002, 0.0001]:
                r_test = r.copy()
                r_test[i] += dr
                if _check_feasible(x, y, r_test, tol):
                    r = r_test
                    current_obj = np.sum(r)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            # Try small position adjustments to free up space
            for i in range(N):
                for dx in [-0.005, 0.005]:
                    for dy in [-0.005, 0.005]:
                        x_test = x.copy()
                        y_test = y.copy()
                        x_test[i] += dx
                        y_test[i] += dy
                        x_test[i] = np.clip(x_test[i], 0.01, 0.99)
                        y_test[i] = np.clip(y_test[i], 0.01, 0.99)
                        if _check_feasible(x_test, y_test, r, tol):
                            x = x_test
                            y = y_test
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
        
        T *= 0.95
        if T < T_min:
            break
    
    return x, y, r, current_obj

def construct():
    """Construct 26 disjoint circles in unit square maximizing sum of radii."""
    best_overall = None
    best_overall_obj = -np.inf
    
    # Strategy 1: Grid init + PT-MC
    x0, y0, r0 = _grid_init()
    state1, obj1 = _pt_mc(x0, y0, r0, num_replicas=12, n_iter=8000,
                          T_low=0.01, T_high=1.0, beta=200.0,
                          sigma_pos=0.02, sigma_r=0.003, seed=42)
    if state1 is not None:
        x, y, r = state1
        x, y, r, obj = _anneal_radii(x, y, r, max_rounds=300)
        if _check_feasible(x, y, r):
            if obj > best_overall_obj:
                best_overall_obj = obj
                best_overall = (x, y, r)
    
    # Strategy 2: Random feasible init + PT-MC with different seed
    x0b, y0b, r0b = _random_feasible_init(seed=123)
    state2, obj2 = _pt_mc(x0b, y0b, r0b, num_replicas=10, n_iter=6000,
                          T_low=0.02, T_high=0.8, beta=150.0,
                          sigma_pos=0.015, sigma_r=0.002, seed=999)
    if state2 is not None:
        x, y, r = state2
        x, y, r, obj = _anneal_radii(x, y, r, max_rounds=300)
        if _check_feasible(x, y, r):
            if obj > best_overall_obj:
                best_overall_obj = obj
                best_overall = (x, y, r)
    
    # Strategy 3: Another random init with different parameters
    x0c, y0c, r0c = _random_feasible_init(seed=777)
    state3, obj3 = _pt_mc(x0c, y0c, r0c, num_replicas=8, n_iter=5000,
                          T_low=0.05, T_high=0.5, beta=100.0,
                          sigma_pos=0.025, sigma_r=0.004, seed=555)
    if state3 is not None:
        x, y, r = state3
        x, y, r, obj = _anneal_radii(x, y, r, max_rounds=200)
        if _check_feasible(x, y, r):
            if obj > best_overall_obj:
                best_overall_obj = obj
                best_overall = (x, y, r)
    
    if best_overall is None:
        # Last resort: simple grid
        x, y, r = _grid_init()
        x, y, r, _ = _anneal_radii(x, y, r, max_rounds=100)
        best_overall = (x, y, r)
    
    x, y, r = best_overall
    
    # Final safety check
    if not _check_feasible(x, y, r):
        for _ in range(50):
            r = r * 0.999
            if _check_feasible(x, y, r):
                break
    
    return np.column_stack([x, y, r])
# EVOLVE-BLOCK-END
