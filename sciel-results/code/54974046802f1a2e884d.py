import numpy as np
import time
import random

N = 50
DELTA = 1.0 / N
_NUM_LAGS = 2 * N - 1

# Precompute overlap matrices
def _build_overlap_matrices():
    M = np.zeros((_NUM_LAGS, N, N))
    for k in range(_NUM_LAGS):
        for i in range(N):
            for j in range(N):
                lo = max(i, k - j)
                hi = min(i + 1, k - j + 1)
                M[k, i, j] = max(0.0, hi - lo)
    return M

_M_matrices = _build_overlap_matrices()

def _compute_ratio_full(h):
    """Compute ratio for full 50-element vector."""
    h = np.maximum(h, 0)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15 or l2_sq < 1e-15:
        return 0.0
    ff_vals = np.array([DELTA * (h @ _M_matrices[k] @ h) for k in range(_NUM_LAGS)])
    sup_ff = np.max(ff_vals)
    if sup_ff < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_ff)

def _generate_sidon_set(n, size, seed=42):
    """Generate a Sidon set of given size in [0, n)."""
    rng = random.Random(seed)
    seq = [0]
    used_sums = {0}
    candidates = list(range(1, n))
    rng.shuffle(candidates)
    for candidate in candidates:
        valid = True
        for i in seq:
            if i + candidate in used_sums:
                valid = False
                break
        if valid:
            seq.append(candidate)
            used_sums.add(candidate + candidate)
            for i in seq[:-1]:
                used_sums.add(i + candidate)
        if len(seq) >= size:
            break
    return seq[:size]

def _initialize_sparse(h, K, seed=42):
    """Initialize with sparse spike pattern."""
    rng = random.Random(seed)
    support = _generate_sidon_set(N, K, seed)
    if len(support) < K:
        # Fallback to random
        support = rng.sample(range(N), K)
    
    h = np.zeros(N)
    # Random weights
    weights = np.array([rng.random() + 0.1 for _ in range(K)])
    weights = weights / np.sum(weights) * 10  # Scale up
    for i, pos in enumerate(support):
        h[pos] = weights[i]
    return h

def _simulated_annealing(h_init, max_iter=2000, T_start=1.0, T_end=0.001, deadline=None):
    """Simulated annealing to maximize the ratio."""
    h = h_init.copy()
    current_score = _compute_ratio_full(h)
    best_h = h.copy()
    best_score = current_score
    
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iter - 1))
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate neighbor: modify 1-3 random positions
        num_mods = rng.randint(1, 3)
        h_new = h.copy()
        for _ in range(num_mods):
            idx = rng.randint(0, N - 1)
            if h_new[idx] > 0:
                # Either scale or set to zero
                if rng.random() < 0.3:
                    h_new[idx] = 0.0
                else:
                    h_new[idx] *= (0.5 + rng.random() * 1.0)
            else:
                # Add new spike
                h_new[idx] = rng.random() * 0.5
        
        new_score = _compute_ratio_full(h_new)
        
        if new_score >= current_score:
            h = h_new
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_h = h_new.copy()
        else:
            # Accept with probability exp((new - current) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / max(T, 1e-15))
                if rng.random() < prob:
                    h = h_new
                    current_score = new_score
        
        T *= cooling
    
    return best_h, best_score

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    rng = random.Random(12345)
    np.random.seed(12345)
    
    best_h = np.zeros(N)
    best_score = 0.0
    
    # Multiple restarts with different sparse initializations
    num_restarts = 8
    restarts_per_second = 0.05  # Rough estimate
    
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Vary the sparsity level
        K = 5 + (restart % 4) * 2  # K = 5, 7, 9, 11, 5, 7, 9, 11
        seed = 1000 + restart * 7
        
        h_init = _initialize_sparse(np.zeros(N), K, seed)
        
        # Run simulated annealing
        h_sa, score_sa = _simulated_annealing(
            h_init, 
            max_iter=1500, 
            T_start=0.5, 
            T_end=0.0005,
            deadline=DEADLINE - 5
        )
        
        if score_sa > best_score:
            best_score = score_sa
            best_h = h_sa.copy()
    
    # Final polish: coordinate descent on the best solution
    if time.time() - start_time < DEADLINE - 5 and best_score > 0:
        h_polished = best_h.copy()
        for iteration in range(200):
            if time.time() - start_time > DEADLINE - 3:
                break
            improved = False
            for i in range(N):
                h_old_i = h_polished[i]
                best_r = _compute_ratio_full(h_polished)
                
                if h_old_i < 1e-10:
                    # Try adding a small spike
                    h_polished[i] = 0.05
                    r_new = _compute_ratio_full(h_polished)
                    if r_new > best_r + 1e-10:
                        improved = True
                    else:
                        h_polished[i] = 0.0
                    continue
                
                # Golden section search
                lo = 0.0
                hi = max(1.0, 3.0 * h_old_i)
                gr = (np.sqrt(5) - 1) / 2
                
                for _ in range(10):
                    x1 = hi - gr * (hi - lo)
                    x2 = lo + gr * (hi - lo)
                    
                    h_polished[i] = x1
                    r1 = _compute_ratio_full(h_polished)
                    h_polished[i] = x2
                    r2 = _compute_ratio_full(h_polished)
                    
                    if r1 > r2:
                        hi = x2
                    else:
                        lo = x1
                
                best_h_i = (lo + hi) / 2
                h_polished[i] = best_h_i
                r_new = _compute_ratio_full(h_polished)
                
                if r_new > best_r + 1e-12:
                    improved = True
                else:
                    h_polished[i] = h_old_i
            
            if not improved:
                break
        
        final_score = _compute_ratio_full(h_polished)
        if final_score > best_score:
            best_score = final_score
            best_h = h_polished.copy()
    
    return best_h.tolist()
