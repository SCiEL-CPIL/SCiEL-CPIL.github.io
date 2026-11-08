"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 50

def _autoconv(h):
    """Full linear autoconvolution of 1D array h (nonnegative)."""
    return np.convolve(h, h, mode='full')

def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l1 = h.sum()
    if l1 <= 0:
        return 0.0
    l2sq = float(np.dot(h, h))
    ac = _autoconv(h)
    ac_inf = float(ac.max())
    if ac_inf <= 0:
        return 0.0
    return l2sq / (l1 * ac_inf)

def _make_cluster(size, ratios):
    """Create a cluster of given size with given ratios, L1-normalized, padded to N."""
    r = np.array(ratios, dtype=float)
    r = r / r.sum()
    h = np.zeros(N)
    h[:len(r)] = r
    return h

# EVOLVE-BLOCK-START
def construct():
    """
    Maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf) for 50 nonnegative heights.
    
    Strategy:
    1. Start with promising analytic shapes (2-point clusters with various ratios).
    2. Perform coordinate ascent / local search on the ratio.
    3. Return the best shape found within the time budget.
    """
    t_start = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill
    
    def elapsed():
        return time.time() - t_start
    
    def time_left():
        return DEADLINE - elapsed()
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # --- Phase 1: Analytic / heuristic initial shapes ---
    candidates = []
    
    # 2-point clusters with various ratios
    for ratio in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        candidates.append(_make_cluster(2, [ratio, 1.0]))
        candidates.append(_make_cluster(2, [1.0, ratio]))
    
    # 3-point clusters
    for r1 in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for r2 in [0.5, 1.0, 1.5, 2.0, 3.0]:
            candidates.append(_make_cluster(3, [r1, 1.0, r2]))
            candidates.append(_make_cluster(3, [1.0, r1, r2]))
    
    # Single point (Dirac)
    h_single = np.zeros(N)
    h_single[0] = 1.0
    candidates.append(h_single)
    
    # Uniform
    candidates.append(np.ones(N) / N)
    
    # Evaluate all candidates
    for h in candidates:
        s = _ratio(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # --- Phase 2: Local search (coordinate ascent) ---
    def _local_search(h, max_iters=20000):
        """Coordinate ascent on nonnegative heights to maximize ratio."""
        h = h.copy()
        h = h / h.sum()  # normalize
        current = _ratio(h)
        n = len(h)
        
        for it in range(max_iters):
            if elapsed() > DEADLINE - 5:
                break
            improved = False
            # Try perturbing each coordinate
            for i in range(n):
                old = h[i]
                # Try increasing
                for delta in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]:
                    new_val = old + delta
                    if new_val < 0:
                        continue
                    h[i] = new_val
                    s = _ratio(h)
                    if s > current + 1e-10:
                        current = s
                        improved = True
                        break
                    h[i] = old
                # Try decreasing
                for delta in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]:
                    new_val = old - delta
                    if new_val < 0:
                        continue
                    h[i] = new_val
                    s = _ratio(h)
                    if s > current + 1e-10:
                        current = s
                        improved = True
                        break
                    h[i] = old
                if not improved:
                    # Try adding mass to a zero coordinate
                    zero_indices = np.where(h < 1e-10)[0]
                    if len(zero_indices) > 0:
                        # Pick a few random zero positions
                        for _ in range(min(5, len(zero_indices))):
                            j = zero_indices[np.random.randint(len(zero_indices))]
                            for delta in [0.001, 0.01, 0.05, 0.1]:
                                h[j] = delta
                                s = _ratio(h)
                                if s > current + 1e-10:
                                    current = s
                                    improved = True
                                    break
                                h[j] = 0.0
                            if improved:
                                break
            if not improved:
                break
        return h, current
    
    # Run local search from best candidate
    h_ls, s_ls = _local_search(best_h)
    if s_ls > best_score:
        best_score = s_ls
        best_h = h_ls
    
    # --- Phase 3: More aggressive search if time permits ---
    if time_left() > 20:
        # Try random restarts with different initial shapes
        np.random.seed(42)
        for restart in range(50):
            if elapsed() > DEADLINE - 10:
                break
            # Random sparse shape
            h_rand = np.zeros(N)
            n_nonzero = np.random.randint(1, 8)
            idx = np.random.choice(N, size=n_nonzero, replace=False)
            h_rand[idx] = np.random.dirichlet(np.ones(n_nonzero))
            h_ls2, s_ls2 = _local_search(h_rand, max_iters=5000)
            if s_ls2 > best_score:
                best_score = s_ls2
                best_h = h_ls2
    
    # --- Phase 4: Fine-tune with smaller steps ---
    if time_left() > 10:
        def _fine_tune(h, max_iters=10000):
            h = h.copy()
            h = h / h.sum()
            current = _ratio(h)
            n = len(h)
            for it in range(max_iters):
                if elapsed() > DEADLINE - 3:
                    break
                improved = False
                for i in range(n):
                    old = h[i]
                    for delta in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
                        new_val = old + delta
                        if new_val < 0:
                            continue
                        h[i] = new_val
                        s = _ratio(h)
                        if s > current + 1e-12:
                            current = s
                            improved = True
                            break
                        h[i] = old
                    for delta in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
                        new_val = old - delta
                        if new_val < 0:
                            continue
                        h[i] = new_val
                        s = _ratio(h)
                        if s > current + 1e-12:
                            current = s
                            improved = True
                            break
                        h[i] = old
                    if not improved:
                        break
                if not improved:
                    break
            return h, current
        
        h_ft, s_ft = _fine_tune(best_h)
        if s_ft > best_score:
            best_score = s_ft
            best_h = h_ft
    
    # Ensure nonnegativity and return as list
    best_h = np.maximum(best_h, 0.0)
    return list(best_h)
# EVOLVE-BLOCK-END
