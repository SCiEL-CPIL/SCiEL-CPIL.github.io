import numpy as np
import time
from scipy.optimize import linprog

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog

def _compute_overlaps(h):
    n = len(h)
    overlaps = np.zeros(n - 1)
    for k in range(1, n):
        h_shifted = np.roll(h, k)
        overlaps[k-1] = np.sum(h * (1 - h_shifted))
    return overlaps

def _compute_grads(h):
    n = len(h)
    grads = np.zeros((n - 1, n))
    for k in range(1, n):
        h_minus = np.roll(h, -k)
        h_plus = np.roll(h, k)
        grads[k-1, :] = 1 - h_minus - h_plus
    return grads

def _solve_lp(h_init, overlaps_init, grads_init):
    n = N
    c = np.zeros(n + 1)
    c[n] = 1.0
    
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    b_eq = [N / 2.0]
    
    A_ub = np.zeros((n - 1, n + 1))
    b_ub = np.zeros(n - 1)
    
    for k in range(1, n):
        A_ub[k-1, :n] = grads[k-1, :]
        A_ub[k-1, n] = -1.0
        const_k = overlaps_init[k-1] - np.dot(grads[k-1, :], h_init)
        b_ub[k-1] = -const_k
        
    bounds = [(0, 1)] * n + [(0, N)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
        if res.success:
            h_new = res.x[:n]
            t_new = res.x[n]
            return h_new, t_new
        else:
            return None, None
    except Exception:
        return None, None

def _local_search(h_init, deadline, max_iters=2000):
    best_h = h_init.copy()
    best_score = np.max(_compute_overlaps(best_h))
    current_h = h_init.copy()
    current_score = best_score
    T = 0.05
    cooling = 0.99
    
    while time.time() < deadline and max_iters > 0:
        max_iters -= 1
        i, j = np.random.choice(N, 2, replace=False)
        new_h = current_h.copy()
        # Swap or small move
        # Let's do a small move towards 0 or 1 or swap
        if np.random.rand() < 0.5:
             # Swap
             new_h[i], new_h[j] = current_h[j], current_h[i]
        else:
             # Perturb
             new_h[i] += np.random.uniform(-0.1, 0.1)
             new_h[j] += np.random.uniform(-0.1, 0.1)
        
        # Project to sum = N/2 and [0,1]
        new_h = np.clip(new_h, 0, 1)
        diff = N/2 - np.sum(new_h)
        if abs(diff) > 1e-8:
            mask = (new_h < 1) & (new_h > 0)
            if np.sum(mask) > 0:
                new_h[mask] += diff / np.sum(mask)
            else:
                # Fallback
                new_h = np.full(N, 0.5)
            new_h = np.clip(new_h, 0, 1)
            
        new_score = np.max(_compute_overlaps(new_h))
        delta = new_score - current_score
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_h = new_h
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = new_h.copy()
        
        T = max(T * cooling, 0.001)
        
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Initialize with perturbed uniform to break symmetry
    np.random.seed(42)
    h = np.full(N, 0.5)
    h += np.random.uniform(-0.1, 0.1, N)
    h = np.clip(h, 0, 1)
    # Adjust sum
    diff = N/2 - np.sum(h)
    if diff != 0:
        mask = (h < 1) & (h > 0)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0, 1)
        
    best_h = h.copy()
    best_score = np.max(_compute_overlaps(h))
    
    overlaps = _compute_overlaps(h)
    
    # SLP Loop
    for i in range(50):
        if time.time() > deadline - 10:
            break
            
        grads = _compute_grads(h)
        h_new, t_lp = _solve_lp(h, overlaps, grads)
        
        if h_new is None:
            break
            
        overlaps_new = _compute_overlaps(h_new)
        score_new = np.max(overlaps_new)
        
        if score_new < best_score:
            best_score = score_new
            best_h = h_new.copy()
            
        # If no improvement, try local search
        if score_new >= best_score - 1e-6:
             # Run a short local search from h_new
             h_ls, score_ls = _local_search(h_new, deadline - 5, max_iters=500)
             if score_ls < best_score:
                 best_score = score_ls
                 best_h = h_ls.copy()
             # Update h to the local search result to continue SLP from there
             h = h_ls
             overlaps = _compute_overlaps(h)
        else:
             h = h_new
             overlaps = overlaps_new
             
    return best_h
# EVOLVE-BLOCK-END
