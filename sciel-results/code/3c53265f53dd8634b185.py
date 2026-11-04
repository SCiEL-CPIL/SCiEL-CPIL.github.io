import numpy as np
import time

N = 100
TARGET_SUM = N / 2.0

def _max_overlap(h):
    """
    Compute the maximum linear overlap C_k = sum_i h_i (1 - h_{i+k}) for k != 0.
    """
    g = 1.0 - h
    c = np.correlate(h, g, mode='full')
    # c has length 2N-1.
    # c[N] is lag 0.
    # We want max of c[:N] and c[N+1:] (excluding lag 0)
    c_neg = c[:N]
    c_pos = c[N+1:]
    overlaps = np.concatenate([c_neg, c_pos])
    return float(np.max(overlaps))

def _adjust_sum(h, target=TARGET_SUM):
    """
    Adjust h to have exact sum, keeping values in [0,1].
    """
    h = np.clip(h, 0, 1).copy()
    for _ in range(100):
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            mask = h < 1.0
            if not np.any(mask):
                break
            step = diff / np.sum(mask)
            h[mask] = np.minimum(h[mask] + step, 1.0)
        else:
            mask = h > 0.0
            if not np.any(mask):
                break
            step = (-diff) / np.sum(mask)
            h[mask] = np.maximum(h[mask] - step, 0.0)
    return h

def _local_search(h, max_iters=1000, deadline=None):
    """
    Local search to minimize max overlap.
    """
    current_max = _max_overlap(h)
    improved = True
    iter_count = 0
    
    while improved and iter_count < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        best_swap = None
        best_new_max = current_max
        
        # Try all pairs
        for i in range(N):
            for j in range(i+1, N):
                if h[i] == h[j]:
                    continue
                h[i], h[j] = h[j], h[i]
                new_max = _max_overlap(h)
                h[i], h[j] = h[j], h[i]
                
                if new_max < best_new_max - 1e-12:
                    best_new_max = new_max
                    best_swap = (i, j)
        
        if best_swap is not None:
            i, j = best_swap
            h[i], h[j] = h[j], h[i]
            current_max = best_new_max
            improved = True
        iter_count += 1
    
    return h, current_max

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Start with sorted binary vector: 0s then 1s
    h = np.zeros(N)
    h[N//2:] = 1.0
    
    # Adjust sum (it is already N/2, but ensure precision)
    h = _adjust_sum(h)
    
    # Local search
    h, score = _local_search(h, max_iters=2000, deadline=deadline)
    
    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    best_h = h
    best_score = score
    
    for _ in range(5):
        if time.time() > deadline - 20.0:
            break
        h_rand = rng.uniform(0, 1, N)
        h_rand = _adjust_sum(h_rand)
        h_ls, s_ls = _local_search(h_rand, max_iters=500, deadline=deadline - 10.0)
        if s_ls < best_score:
            best_score = s_ls
            best_h = h_ls
    
    return best_h
