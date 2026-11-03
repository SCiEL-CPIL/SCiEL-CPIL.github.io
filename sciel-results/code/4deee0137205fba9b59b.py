import numpy as np
import time
from scipy.optimize import minimize

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

def _get_peak_lag(h):
    """Return the lag index (in full convolve output) where autoconvolution peaks."""
    h_conv = np.convolve(h, h, mode='full')
    return int(np.argmax(h_conv))

def _peak_contributors(h, peak_lag):
    """
    For a given peak lag in full convolve, find which pairs (i,j) with i+j=peak_lag
    contribute to h_conv[peak_lag]. Returns list of (i,j) pairs where h[i]*h[j] is nonzero.
    """
    # In full convolve of length N, output has length 2N-1
    # h_conv[k] = sum_{i} h[i] * h[k-i] where 0<=i<N, 0<=k-i<N
    pairs = []
    for i in range(N):
        j = peak_lag - i
        if 0 <= j < N:
            if h[i] > 1e-12 and h[j] > 1e-12:
                pairs.append((i, j))
    return pairs

def _coordinate_descent(h, max_iters=200, time_budget=30.0):
    """
    Directed coordinate descent: identify the peak of autoconvolution,
    then perform moves that reduce the peak while maintaining L1 and nonnegativity.
    
    Strategy: 
    1. Find peak lag
    2. Among contributing pairs, reduce the larger of h[i], h[j] and transfer to a 
       non-contributing index (or reduce both proportionally)
    3. This reduces the peak contribution
    """
    h = np.array(h, dtype=np.float64)
    h = np.maximum(0, h)
    l1 = np.sum(h)
    if l1 <= 0:
        return h
    h = h / l1  # normalize to L1=1
    
    best_h = h.copy()
    best_score = _score_raw(best_h)
    start_t = time.time()
    
    for it in range(max_iters):
        if time.time() - start_t > time_budget:
            break
        
        # Find current peak
        h_conv = np.convolve(h, h, mode='full')
        peak_lag = int(np.argmax(h_conv))
        peak_val = h_conv[peak_lag]
        
        if peak_val <= 1e-15:
            break
        
        # Get contributing pairs
        pairs = _peak_contributors(h, peak_lag)
        if not pairs:
            break
        
        # Strategy: for each contributing pair (i,j), consider reducing h[i] and h[j]
        # and redistributing to indices that don't contribute to this peak
        # 
        # Key insight: if we reduce h[i] by delta and add delta to h[k] where k is
        # NOT in any pair that sums to peak_lag, the peak decreases.
        
        # Find "safe" indices: indices k such that for ALL contributing pairs (i,j),
        # k is not equal to i or j, AND (peak_lag - k) is not in [0,N) or h[peak_lag-k] is small
        # Actually simpler: find k such that k doesn't form a pair summing to peak_lag
        # i.e., peak_lag - k is outside [0, N)
        
        safe_indices = []
        for k in range(N):
            j = peak_lag - k
            if j < 0 or j >= N:
                safe_indices.append(k)
            else:
                # k contributes to peak if h[k] and h[j] are both nonzero
                # But we want to ADD to k, so we need to check if adding to k would
                # create a new peak contribution. If h[j] is already small, it's relatively safe.
                # For now, mark as unsafe if both are significant
                if h[k] < 0.01 and h[j] < 0.01:
                    safe_indices.append(k)
        
        if not safe_indices:
            # No safe indices; try reducing peak contributors symmetrically
            # Reduce all contributing indices proportionally
            contrib_idx = set()
            for i, j in pairs:
                contrib_idx.add(i)
                contrib_idx.add(j)
            if not contrib_idx:
                break
            # Find an index outside contrib_idx to receive the mass
            outside = [k for k in range(N) if k not in contrib_idx]
            if not outside:
                break
            # Pick the outside index with smallest current value
            target_k = min(outside, key=lambda k: h[k])
            
            # Reduce all contributors by a small amount
            n_contrib = len(contrib_idx)
            delta = 0.01 / n_contrib
            for idx in contrib_idx:
                if h[idx] > delta * 2:
                    h[idx] -= delta
            h[target_k] += delta * n_contrib
            
            # Check if this improved the score
            new_score = _score_raw(h)
            if new_score > best_score:
                best_score = new_score
                best_h = h.copy()
            else:
                # Revert
                for idx in contrib_idx:
                    h[idx] += delta
                h[target_k] -= delta * n_contrib
                # Try smaller delta
                delta = 0.001 / n_contrib
                for idx in contrib_idx:
                    if h[idx] > delta * 2:
                        h[idx] -= delta
                h[target_k] += delta * n_contrib
                new_score = _score_raw(h)
                if new_score > best_score:
                    best_score = new_score
                    best_h = h.copy()
                else:
                    for idx in contrib_idx:
                        h[idx] += delta
                    h[target_k] -= delta * n_contrib
                continue
        
        # Pick a random safe index as target
        target_k = np.random.choice(safe_indices)
        
        # Pick a contributing index to reduce (prefer the larger one)
        # Group by unique indices in pairs
        contrib_counts = {}
        for i, j in pairs:
            contrib_counts[i] = contrib_counts.get(i, 0) + 1
            contrib_counts[j] = contrib_counts.get(j, 0) + 1
        
        # Reduce the index with highest value among contributors
        contrib_idx_list = list(contrib_counts.keys())
        if not contrib_idx_list:
            break
        source_idx = max(contrib_idx_list, key=lambda i: h[i])
        
        # Try different transfer amounts
        improved = False
        for frac in [0.05, 0.02, 0.01, 0.005, 0.001]:
            delta = h[source_idx] * frac
            if delta < 1e-8:
                continue
            h[source_idx] -= delta
            h[target_k] += delta
            
            new_score = _score_raw(h)
            if new_score > best_score + 1e-10:
                best_score = new_score
                best_h = h.copy()
                improved = True
                break
            else:
                h[source_idx] += delta
                h[target_k] -= delta
        
        if not improved:
            # Try reducing two contributors simultaneously
            if len(contrib_idx_list) >= 2:
                # Sort by value descending
                contrib_idx_list.sort(key=lambda i: -h[i])
                s1, s2 = contrib_idx_list[0], contrib_idx_list[1]
                for frac in [0.05, 0.02, 0.01]:
                    d1 = h[s1] * frac
                    d2 = h[s2] * frac
                    if d1 < 1e-8 or d2 < 1e-8:
                        continue
                    h[s1] -= d1
                    h[s2] -= d2
                    h[target_k] += d1 + d2
                    
                    new_score = _score_raw(h)
                    if new_score > best_score + 1e-10:
                        best_score = new_score
                        best_h = h.copy()
                        improved = True
                        break
                    else:
                        h[s1] += d1
                        h[s2] += d2
                        h[target_k] -= d1 + d2
            if not improved:
                # No improvement this iteration; add small perturbation to escape local opt
                h += np.random.rand(N) * 0.001
                h = np.maximum(0, h)
                h = h / np.sum(h)
    
    return best_h

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    x_idx = np.arange(N)
    
    # --- Phase 1: Parametric initializations ---
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones(N))
    
    # 2. Gaussians
    for c in np.linspace(3, N-3, 15):
        for w in np.linspace(0.5, N/2, 15):
            h_gauss = np.exp(-0.5 * ((x_idx - c) / w)**2)
            candidates.append(h_gauss)
    
    # 3. Triangular
    for c in np.linspace(3, N-3, 15):
        for w in np.linspace(1, N, 15):
            h_tri = np.maximum(0, 1 - np.abs(x_idx - c) / w)
            if np.sum(h_tri) > 0:
                candidates.append(h_tri)
    
    # 4. Cosine bumps
    for c in np.linspace(3, N-3, 10):
        for w in np.linspace(2, N/2, 10):
            h_cos = np.maximum(0, np.cos(np.pi * (x_idx - c) / (2*w)))
            if np.sum(h_cos) > 0:
                candidates.append(h_cos)
    
    # 5. Sparse peaks (Sidon-like)
    for _ in range(150):
        if time.time() > deadline - 40: break
        h_sparse = np.zeros(N)
        num_peaks = np.random.randint(3, 12)
        indices = np.random.choice(N, num_peaks, replace=False)
        h_sparse[indices] = 1.0
        h_sparse += np.random.rand(N) * 0.05
        candidates.append(h_sparse)
    
    # 6. Random smooth
    for _ in range(80):
        if time.time() > deadline - 40: break
        h_r = np.random.rand(N)
        kernel = np.ones(np.random.randint(3, 8)) / 5.0
        h_r = np.convolve(h_r, kernel, mode='same')
        h_r = np.maximum(0, h_r)
        if np.sum(h_r) > 0:
            candidates.append(h_r)
    
    # 7. New: "spiky with gaps" - high values at well-separated positions
    for _ in range(100):
        if time.time() > deadline - 40: break
        h_spiky = np.zeros(N)
        # Choose 4-8 well-separated peaks
        n_peaks = np.random.randint(4, 9)
        positions = np.sort(np.random.choice(N, n_peaks, replace=False))
        # Ensure minimum separation
        valid = True
        for i in range(1, len(positions)):
            if positions[i] - positions[i-1] < 3:
                valid = False
                break
        if valid:
            for p in positions:
                h_spiky[p] = np.random.uniform(0.5, 1.5)
                # Add small neighbors
                if p > 0 and np.random.rand() < 0.3:
                    h_spiky[p-1] = h_spiky[p] * np.random.uniform(0.1, 0.3)
                if p < N-1 and np.random.rand() < 0.3:
                    h_spiky[p+1] = h_spiky[p] * np.random.uniform(0.1, 0.3)
            if np.sum(h_spiky) > 0:
                candidates.append(h_spiky)
    
    # Evaluate all candidates
    best_h = None
    best_score = 0.0
    for h_cand in candidates:
        h_cand = np.asarray(h_cand, dtype=np.float64)
        if np.sum(h_cand) <= 0:
            continue
        s = _score_raw(h_cand)
        if s > best_score:
            best_score = s
            best_h = h_cand.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_score = _score_raw(best_h)
    
    # --- Phase 2: Directed Coordinate Descent ---
    # This is the structural change: instead of generic SLSQP, use peak-aware
    # coordinate moves that directly target the autoconvolution peak
    
    remaining_time = deadline - time.time()
    cd_budget = min(40.0, remaining_time * 0.5)
    
    h_cd = _coordinate_descent(best_h, max_iters=500, time_budget=cd_budget)
    s_cd = _score_raw(h_cd)
    if s_cd > best_score:
        best_score = s_cd
        best_h = h_cd
    
    # --- Phase 3: SLSQP polish with analytic structure ---
    def neg_obj_h(h):
        l1 = np.sum(h)
        if l1 <= 1e-15:
            return 0.0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv <= 1e-15:
            return 0.0
        return -(l2_sq / (l1 * max_conv))
    
    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
    ]
    bounds = [(0.0, 1.0)] * N
    
    remaining_time = deadline - time.time()
    if remaining_time > 15:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 300, 'ftol': 1e-12})
            if res.fun < 0:
                h_candidate = np.maximum(0, res.x)
                if np.sum(h_candidate) > 0:
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
        except Exception:
            pass
    
    # --- Phase 4: More coordinate descent rounds from multiple starts ---
    remaining_time = deadline - time.time()
    if remaining_time > 20:
        # Generate a few diverse starts and run CD from each
        diverse_starts = [best_h / np.sum(best_h)]
        
        # Top candidates
        scored_cands = []
        for h_cand in candidates:
            h_cand = np.asarray(h_cand, dtype=np.float64)
            if np.sum(h_cand) <= 0:
                continue
            s = _score_raw(h_cand)
            scored_cands.append((s, h_cand))
        scored_cands.sort(key=lambda x: -x[0])
        for s, h_cand in scored_cands[:5]:
            diverse_starts.append(h_cand / np.sum(h_cand))
        
        # Random sparse starts
        for _ in range(5):
            h_s = np.zeros(N)
            k = np.random.randint(4, 10)
            idx = np.random.choice(N, k, replace=False)
            h_s[idx] = 1.0
            h_s += np.random.rand(N) * 0.03
            if np.sum(h_s) > 0:
                diverse_starts.append(h_s / np.sum(h_s))
        
        for h_init in diverse_starts:
            if time.time() > deadline - 10:
                break
            h_init = np.maximum(0, h_init)
            if np.sum(h_init) <= 0:
                continue
            h_init = h_init / np.sum(h_init)
            
            cd_time = min(15.0, deadline - 10 - time.time())
            if cd_time <= 2:
                break
            h_cd = _coordinate_descent(h_init, max_iters=300, time_budget=cd_time)
            s_cd = _score_raw(h_cd)
            if s_cd > best_score:
                best_score = s_cd
                best_h = h_cd
    
    # Final SLSQP polish
    remaining_time = deadline - time.time()
    if remaining_time > 8:
        h_start = best_h / np.sum(best_h)
        try:
            res = minimize(neg_obj_h, h_start, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 500, 'ftol': 1e-13})
            if res.fun < 0:
                h_candidate = np.maximum(0, res.x)
                if np.sum(h_candidate) > 0:
                    s_candidate = _score_raw(h_candidate)
                    if s_candidate > best_score:
                        best_score = s_candidate
                        best_h = h_candidate
        except Exception:
            pass
    
    # Return the optimized shape (normalized to L1=1)
    best_h = np.maximum(0, best_h)
    if np.sum(best_h) > 0:
        best_h = best_h / np.sum(best_h)
    
    return list(best_h)
