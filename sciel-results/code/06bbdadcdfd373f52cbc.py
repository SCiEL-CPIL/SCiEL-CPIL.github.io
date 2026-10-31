import numpy as np
import time

N = 50

def compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # The key insight: for the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf),
    # with h normalized so ||h||_1 = 1, the ratio is ||h||_2^2 / ||h*h||_inf.
    # A delta function gives ratio = 1/1 = 1.
    # Sparse configurations with well-separated peaks can exceed 1.
    # For a Sidon set (all pairwise differences distinct), the autoconvolution
    # has minimal collisions, maximizing the ratio.
    
    # Strategy: 
    # 1. Find good Sidon-like sets (minimize autoconvolution collisions)
    # 2. Optimize heights on those sets using coordinate ascent
    # 3. Also try random sparse configurations with local search
    
    best_h = np.zeros(N)
    best_h[0] = 1.0
    best_score = compute_ratio(best_h)
    
    def make_sparse_peaks(heights, positions):
        h = np.zeros(N)
        for ht, pos in zip(heights, positions):
            h[pos] = ht
        return h
    
    def local_search_heights(positions, h_init, max_iter=2000):
        """Coordinate ascent on heights at fixed positions."""
        k = len(positions)
        h = np.array(h_init, dtype=np.float64)
        h = np.maximum(h, 1e-10)
        h = h / np.sum(h)
        cur_score = 0.0
        
        def eval_h(h_local):
            full = np.zeros(N)
            for i, pos in enumerate(positions):
                full[pos] = h_local[i]
            return compute_ratio(full)
        
        cur_score = eval_h(h)
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            i = it % k
            best_delta = 0.0
            best_val = h[i]
            # Try a grid of values around current
            for scale in [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0, 3.0, 0.1, 0.3]:
                trial = h.copy()
                trial[i] = best_val * scale
                if trial[i] < 0:
                    continue
                s = eval_h(trial)
                if s > cur_score + 1e-10:
                    best_delta = s
                    best_val = trial[i]
            if best_delta > 0:
                h[i] = best_val
                h = np.maximum(h, 1e-10)
                h = h / np.sum(h)
                cur_score = best_delta
            else:
                # Try random perturbation
                trial = h.copy()
                trial[i] *= np.random.uniform(0.3, 3.0)
                if trial[i] > 0:
                    s = eval_h(trial)
                    if s > cur_score + 1e-10:
                        h = trial
                        h = np.maximum(h, 1e-10)
                        h = h / np.sum(h)
                        cur_score = s
        
        full = np.zeros(N)
        for i, pos in enumerate(positions):
            full[pos] = h[i]
        return full, cur_score
    
    # Try various position sets
    # Key: positions should be "spread out" to minimize convolution overlap
    position_sets = []
    
    # Uniformly spaced
    for spacing in range(3, 12):
        pos = list(range(0, N, spacing))
        if len(pos) >= 2:
            position_sets.append(pos)
    
    # Random sparse sets
    np.random.seed(42)
    for trial in range(50):
        k = np.random.randint(3, 12)
        positions = np.sort(np.random.choice(N, k, replace=False))
        position_sets.append(list(positions))
    
    # Specific well-separated sets
    position_sets.append([0, 8, 16, 24, 32, 40])
    position_sets.append([0, 7, 14, 21, 28, 35, 42])
    position_sets.append([0, 10, 20, 30, 40])
    position_sets.append([0, 12, 24, 36, 48])
    position_sets.append([0, 5, 11, 17, 23, 29, 35, 41])
    position_sets.append([0, 6, 13, 21, 30, 40])
    position_sets.append([0, 9, 18, 27, 36, 45])
    
    # Evaluate candidates
    for positions in position_sets:
        if time.time() > deadline - 10:
            break
        k = len(positions)
        # Start with equal heights
        h_init = np.ones(k) / k
        h_local, score_local = local_search_heights(positions, h_init, max_iter=1500)
        if score_local > best_score:
            best_score = score_local
            best_h = h_local.copy()
        
        # Also try geometric initialization
        for r in [0.3, 0.5, 0.7]:
            if time.time() > deadline - 5:
                break
            h_geo = np.array([r**i for i in range(k)])
            h_geo = h_geo / np.sum(h_geo)
            h_local2, score_local2 = local_search_heights(positions, h_geo, max_iter=1000)
            if score_local2 > best_score:
                best_score = score_local2
                best_h = h_local2.copy()
    
    # Random restarts with local search
    if time.time() < deadline - 10:
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            k = np.random.randint(3, 10)
            positions = np.sort(np.random.choice(N, k, replace=False))
            h_init = np.random.dirichlet(np.ones(k) * np.random.uniform(0.5, 5.0, k))
            h_local, score_local = local_search_heights(list(positions), h_init, max_iter=2000)
            if score_local > best_score:
                best_score = score_local
                best_h = h_local.copy()
    
    return list(best_h)
