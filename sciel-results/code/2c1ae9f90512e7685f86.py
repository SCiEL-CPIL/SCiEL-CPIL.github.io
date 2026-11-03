import numpy as np
import time
import sys
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    x_idx = np.arange(N)
    
    # Helper: optimize amplitudes for a fixed support
    def optimize_amplitudes(support_indices):
        k = len(support_indices)
        if k < 2:
            return None, 0.0
        
        def neg_obj(x):
            a = np.exp(x)
            a = a / np.sum(a)
            h = np.zeros(N)
            h[support_indices] = a
            s = _score_raw(h)
            if s <= 0:
                return 0.0
            return -s
        
        x0 = np.zeros(k)
        
        try:
            res = minimize(neg_obj, x0, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-12})
            if res.fun < 0:
                a = np.exp(res.x)
                a = a / np.sum(a)
                h = np.zeros(N)
                h[support_indices] = a
                s = _score_raw(h)
                return h, s
        except Exception:
            pass
        
        # Fallback: uniform amplitudes
        h = np.zeros(N)
        h[support_indices] = 1.0 / k
        s = _score_raw(h)
        return h, s

    # Phase 1: Parametric Initializations
    # Uniform
    h_init = np.ones(N) / N
    s = _score_raw(h_init)
    if s > best_score:
        best_score = s
        best_h = h_init.copy()
        
    # Gaussians
    for c in np.linspace(2, N-2, 10):
        for w in np.linspace(1, N/3, 10):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            h_gauss /= np.sum(h_gauss)
            s = _score_raw(h_gauss)
            if s > best_score:
                best_score = s
                best_h = h_gauss.copy()
                
    # Triangular
    for c in np.linspace(2, N-2, 10):
        for w in np.linspace(1, N, 10):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                h_tri /= np.sum(h_tri)
                s = _score_raw(h_tri)
                if s > best_score:
                    best_score = s
                    best_h = h_tri.copy()

    # Phase 2: Combinatorial Support Search
    # Focus on sparse supports which minimize autoconvolution overlap
    
    all_indices = np.arange(N)
    
    # Try various support sizes
    # Based on previous insights, small supports (2-8) are likely optimal
    support_sizes = [2, 3, 4, 5, 6, 7, 8, 10, 12]
    
    for size in support_sizes:
        if time.time() > deadline:
            break
        
        candidates = []
        
        # Random subsets
        # More random subsets for smaller sizes to cover the space better
        n_random = 500 if size <= 4 else 200
        for _ in range(n_random):
            subset = np.random.choice(all_indices, size, replace=False)
            subset = np.sort(subset)
            candidates.append(tuple(subset))
        
        # Structured: evenly spaced
        if size <= N:
            even = np.linspace(0, N-1, size, dtype=int)
            candidates.append(tuple(even))
        
        # Structured: clustered at edges
        for cluster_pos in [0, N//2, N-1]:
            if size <= 10:
                start = max(0, cluster_pos - size//2)
                end = min(N, start + size)
                if end - start >= size:
                    candidates.append(tuple(range(start, start+size)))
        
        # Structured: "Sidon-like" greedy construction
        # Try to construct sets where differences are unique (or mostly unique)
        # This minimizes collisions in the autoconvolution.
        # Greedy: start with 0, then pick smallest x such that x - y is not in previous differences?
        # Actually, just picking widely spaced points helps.
        # Let's add a few "greedy max-min distance" sets
        for _ in range(20):
            if time.time() > deadline: break
            # Greedy max-min
            points = [0]
            for _ in range(size - 1):
                best_candidate = -1
                best_min_dist = -1
                for c in range(N):
                    if c in points:
                        continue
                    min_dist = min(abs(c - p) for p in points)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_candidate = c
                if best_candidate != -1:
                    points.append(best_candidate)
                else:
                    break
            if len(points) == size:
                candidates.append(tuple(sorted(points)))

        # Deduplicate
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique_candidates.append(c)
        
        # Optimize for each candidate
        for support in unique_candidates:
            if time.time() > deadline:
                break
            
            h_candidate, s_candidate = optimize_amplitudes(list(support))
            if h_candidate is not None and s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
    
    # Phase 3: Local Refinement around best solution
    # Try perturbing the support of the best solution
    
    if time.time() < deadline - 20:
        active = np.where(best_h > 1e-6)[0]
        
        # Try adding one more element
        for add_idx in all_indices:
            if add_idx in active:
                continue
            if time.time() > deadline:
                break
            new_support = list(active) + [add_idx]
            new_support = sorted(new_support)
            h_candidate, s_candidate = optimize_amplitudes(new_support)
            if h_candidate is not None and s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
        
        # Try removing one element from active
        for rem_idx in active:
            if len(active) <= 2:
                continue
            if time.time() > deadline:
                break
            new_support = [i for i in active if i != rem_idx]
            h_candidate, s_candidate = optimize_amplitudes(new_support)
            if h_candidate is not None and s_candidate > best_score:
                best_score = s_candidate
                best_h = h_candidate.copy()
                
        # Try swapping one element
        active_list = list(active)
        for i in range(len(active_list)):
            if time.time() > deadline: break
            for j in all_indices:
                if j in active_list:
                    continue
                new_support = active_list.copy()
                new_support[i] = j
                new_support = sorted(new_support)
                h_candidate, s_candidate = optimize_amplitudes(new_support)
                if h_candidate is not None and s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate.copy()
                    active_list = new_support.copy()
    
    # Phase 4: Final continuous optimization from best solution
    # Use a soft-max or direct log optimization to fine-tune the amplitudes
    if time.time() < deadline - 10:
        h_start = best_h.copy()
        h_start[h_start < 1e-8] = 1e-8
        x0 = np.log(h_start)
        
        def neg_obj_full(x):
            h = np.exp(x)
            h = h / np.sum(h)
            s = _score_raw(h)
            if s <= 0:
                return 0.0
            return -s
        
        try:
            res = minimize(neg_obj_full, x0, method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-14})
            if res.fun < 0:
                h_candidate = np.exp(res.x)
                h_candidate = h_candidate / np.sum(h_candidate)
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass

    return list(best_h)
