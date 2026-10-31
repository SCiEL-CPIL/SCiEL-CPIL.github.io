import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.signal import fftconvolve

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = fftconvolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def is_sidon_set(indices):
    """Check if a set of indices forms a Sidon set (all pairwise sums distinct)."""
    sums = set()
    n = len(indices)
    for i in range(n):
        for j in range(i, n):
            s = indices[i] + indices[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def find_sidon_sets(max_size=12, n_trials=20000, deadline=None):
    """Find large Sidon sets in [0, N-1]."""
    best_sets = []
    best_size = 0
    
    # Start with known good constructions
    # Greedy Sidon set construction
    def greedy_sidon(target_size):
        indices = []
        for candidate in range(N):
            trial = indices + [candidate]
            if is_sidon_set(trial):
                indices = trial
                if len(indices) >= target_size:
                    break
        return indices
    
    # Try greedy with different starting points
    for start in range(N):
        indices = [start]
        for candidate in range(N):
            if candidate == start:
                continue
            trial = indices + [candidate]
            if is_sidon_set(trial):
                indices = trial
        if len(indices) > best_size:
            best_size = len(indices)
            best_sets = [indices.copy()]
        elif len(indices) == best_size:
            best_sets.append(indices.copy())
    
    # Random search for larger Sidon sets
    for _ in range(n_trials):
        if deadline and time.time() > deadline:
            break
        size = min(best_size + 1, 15)
        if size <= 2:
            continue
        trial = np.random.choice(N, size=size, replace=False).tolist()
        if is_sidon_set(trial):
            if size > best_size:
                best_size = size
                best_sets = [trial.copy()]
            else:
                best_sets.append(trial.copy())
    
    return best_sets, best_size

def optimize_weights_for_support(support, deadline):
    """Optimize weights on a given support to maximize the ratio."""
    k = len(support)
    if k == 0:
        return None, 0.0
    
    # Start with uniform weights
    h = np.zeros(N)
    h[support] = 1.0 / k
    best_h = h.copy()
    best_score = compute_ratio(h)
    
    def objective_neg(h_flat):
        h = np.maximum(h_flat, 0.0)
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        h = h / s
        h2 = np.sum(h**2)
        conv = fftconvolve(h, h, mode='full')
        cinf = np.max(conv)
        if cinf < 1e-15:
            return 1e10
        return -h2 / cinf
    
    # Local optimization from uniform start
    if time.time() < deadline:
        try:
            res = minimize(objective_neg, h, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            h_opt = np.maximum(res.x, 0.0)
            s = np.sum(h_opt)
            if s > 1e-15:
                h_opt = h_opt / s
                sc = compute_ratio(h_opt)
                if sc > best_score:
                    best_score = sc
                    best_h = h_opt.copy()
        except Exception:
            pass
    
    # Try different weight distributions
    for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
        if time.time() > deadline:
            break
        h_trial = np.zeros(N)
        # Power-law weights on the support
        positions = np.arange(k)
        weights = (positions + 1) ** alpha
        weights = weights / np.sum(weights)
        h_trial[support] = weights
        sc = compute_ratio(h_trial)
        if sc > best_score:
            best_score = sc
            best_h = h_trial.copy()
    
    return best_h, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    best_h = np.zeros(N, dtype=np.float64)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    def update_best(h, sc):
        nonlocal best_h, best_score
        if sc > best_score:
            best_score = sc
            best_h = h.copy()
    
    # Phase 1: Find large Sidon sets
    sidon_sets, max_sidon_size = find_sidon_sets(max_size=12, n_trials=50000, deadline=deadline - 60.0)
    
    # Phase 2: Optimize weights on each Sidon set
    for support in sidon_sets[:20]:  # Limit to top 20 to save time
        if time.time() > deadline - 20.0:
            break
        h_opt, sc = optimize_weights_for_support(support, deadline)
        if h_opt is not None and sc > best_score:
            update_best(h_opt, sc)
    
    # Phase 3: Try sparse patterns with optimization (non-Sidon but potentially good)
    if time.time() < deadline - 10.0:
        # Try all pairs
        for i in range(N):
            for j in range(i + 1, N):
                h = np.zeros(N)
                h[i] = 0.5
                h[j] = 0.5
                sc = compute_ratio(h)
                update_best(h, sc)
            if time.time() > deadline - 10.0:
                break
    
    # Phase 4: Final local refinement
    if time.time() < deadline - 5.0:
        try:
            def objective_neg(h_flat):
                h = np.maximum(h_flat, 0.0)
                s = np.sum(h)
                if s < 1e-15:
                    return 1e10
                h = h / s
                h2 = np.sum(h**2)
                conv = fftconvolve(h, h, mode='full')
                cinf = np.max(conv)
                if cinf < 1e-15:
                    return 1e10
                return -h2 / cinf
            
            res = minimize(objective_neg, best_h, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True})
            h_opt = np.maximum(res.x, 0.0)
            s = np.sum(h_opt)
            if s > 1e-15:
                h_opt = h_opt / s
                sc = compute_ratio(h_opt)
                update_best(h_opt, sc)
        except Exception:
            pass
    
    return list(best_h)
