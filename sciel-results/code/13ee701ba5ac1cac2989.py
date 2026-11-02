"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def _ratio(h):
    h = np.array(h, dtype=float)
    if np.sum(h) == 0:
        return 0.0
    l2_sq = np.dot(h, h)
    l1 = np.sum(h)
    # autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup = np.max(conv)
    if sup == 0:
        return 0.0
    return l2_sq / (l1 * sup)

def _optimize_amplitudes(positions, max_iters=200, tol=1e-10):
    """Optimize amplitudes for given support positions via coordinate ascent."""
    n = len(positions)
    if n == 0:
        return [0.0] * N
    # Normalize: set first amplitude to 1 (scale invariance)
    h = np.zeros(N)
    h[positions[0]] = 1.0
    for i in range(1, n):
        h[positions[i]] = 1.0 / (i + 1)
    
    best_h = h.copy()
    best_r = _ratio(h)
    
    for it in range(max_iters):
        improved = False
        for idx in range(n):
            if idx == 0:
                continue  # keep h[positions[0]] = 1 for normalization
            old_val = h[positions[idx]]
            # Try a small grid around current value
            best_local = old_val
            best_local_r = best_r
            for candidate in np.linspace(0.01, 2.0, 20):
                h[positions[idx]] = candidate
                r = _ratio(h)
                if r > best_local_r:
                    best_local_r = r
                    best_local = candidate
            h[positions[idx]] = best_local
            if abs(best_local - old_val) > 1e-8:
                improved = True
        if not improved:
            break
    
    # Final ratio
    final_r = _ratio(h)
    if final_r > best_r:
        best_h = h.copy()
        best_r = final_r
    
    return best_h.tolist()

def _try_support(positions):
    """Try a given set of support positions."""
    h = _optimize_amplitudes(positions)
    return h, _ratio(h)

def construct():
    start_time = time.time()
    deadline = 120.0  # safety margin
    
    best_h = [0.0] * N
    best_r = 0.0
    
    # Start with 2-element support (known to give 1.25)
    h2, r2 = _try_support([0, 1])
    if r2 > best_r:
        best_h, best_r = h2, r2
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_h
    
    # Try 3-element supports at various positions
    # Sidon-like sets: pairs have distinct sums
    three_elem_sets = [
        [0, 1, 3],
        [0, 2, 3],
        [0, 1, 4],
        [0, 2, 5],
        [0, 3, 4],
        [0, 1, 5],
        [0, 2, 4],
        [0, 1, 2],
        [0, 3, 5],
        [0, 4, 5],
        [1, 2, 4],
        [0, 1, 6],
        [0, 3, 6],
        [0, 2, 6],
        [1, 3, 4],
        [0, 5, 6],
        [0, 1, 7],
        [0, 4, 7],
        [1, 4, 5],
        [0, 2, 7],
    ]
    
    for positions in three_elem_sets:
        if time.time() - start_time > deadline:
            break
        h, r = _try_support(positions)
        if r > best_r:
            best_h, best_r = h, r
    
    # Try 4-element supports
    four_elem_sets = [
        [0, 1, 3, 4],
        [0, 1, 3, 7],
        [0, 1, 4, 6],
        [0, 2, 3, 5],
        [0, 1, 3, 5],
        [0, 1, 4, 5],
        [0, 2, 3, 4],
        [0, 1, 2, 4],
        [0, 1, 3, 6],
        [0, 2, 3, 7],
        [0, 1, 5, 6],
        [0, 2, 5, 6],
        [0, 1, 4, 7],
        [0, 2, 4, 5],
        [1, 2, 4, 5],
    ]
    
    for positions in four_elem_sets:
        if time.time() - start_time > deadline:
            break
        h, r = _try_support(positions)
        if r > best_r:
            best_h, best_r = h, r
    
    # If we still have time, try more 3-element sets
    if time.time() - start_time < deadline - 5:
        for a in range(0, 10):
            for b in range(a+1, 15):
                for c in range(b+1, 20):
                    if time.time() - start_time > deadline:
                        break
                    positions = [a, b, c]
                    h, r = _try_support(positions)
                    if r > best_r:
                        best_h, best_r = h, r
    
    return best_h
# EVOLVE-BLOCK-END
