import time
import math
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 50
DH = 1.0 / 20.0
OFFSET = 10

_DEADLINE = 150.0
_SAFETY = 10.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _autoconv(h):
    """Discrete autoconvolution sampled on the grid defined by OFFSET."""
    h = np.asarray(h, dtype=float)
    full = np.convolve(h, h, mode='full')
    return full[OFFSET:OFFSET + N]


def _ratio_exact(h):
    """Compute the exact ratio R = ||h||^2_2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 0.0 or l2sq <= 0.0:
        return 0.0
    ac = _autoconv(h)
    acmax = float(np.max(ac))
    if acmax <= 0.0:
        return 0.0
    return l2sq / (l1 * acmax)


def _sidon_set(n, max_val=N-1):
    """
    Generate a Sidon set (B_2 sequence) of size n within [0, max_val].
    A Sidon set has the property that all pairwise sums are distinct.
    This minimizes collisions in the autoconvolution.
    Uses greedy construction with backtracking.
    """
    if n <= 1:
        return list(range(n))
    
    # Use known small Sidon sets as seeds
    known_sets = {
        1: [0],
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 12],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 25],
        8: [0, 1, 4, 10, 12, 17, 25, 29],
        9: [0, 1, 4, 10, 12, 17, 25, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 25, 29, 36, 41],
        11: [0, 1, 4, 10, 12, 17, 25, 29, 36, 41, 44],
        12: [0, 1, 4, 10, 12, 17, 25, 29, 36, 41, 44, 47],
    }
    
    if n in known_sets:
        s = known_sets[n]
        if max(s) <= max_val:
            return s
    
    # Greedy construction
    def is_sidon(candidate, existing):
        """Check if adding candidate to existing set preserves Sidon property."""
        new_set = sorted(existing + [candidate])
        sums = set()
        for i in range(len(new_set)):
            for j in range(i, len(new_set)):
                s = new_set[i] + new_set[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    result = [0]
    for val in range(1, max_val + 1):
        if len(result) >= n:
            break
        if is_sidon(val, result):
            result.append(val)
    
    # If we didn't find enough, use a different strategy
    if len(result) < n:
        # Fallback: just take first n integers
        result = list(range(n))
    
    return result[:n]


def _sidon_to_heights(sidon_positions, N):
    """Convert a Sidon set of positions to a height vector."""
    h = np.zeros(N)
    for pos in sidon_positions:
        if 0 <= pos < N:
            h[pos] = 1.0
    return h


def _local_search_sidon(sidon_positions, N, max_iters=500):
    """
    Local search on Sidon-like sets.
    Try swapping elements to improve the ratio.
    """
    current = list(sidon_positions)
    current_h = _sidon_to_heights(current, N)
    current_ratio = _ratio_exact(current_h)
    
    best_ratio = current_ratio
    best_set = list(current)
    
    for _ in range(max_iters):
        if _time_left() < 5.0:
            break
        
        improved = False
        # Try single element moves
        for i in range(len(current)):
            old_val = current[i]
            for new_val in range(N):
                if new_val == old_val:
                    continue
                # Check if new_val is already in the set
                if new_val in current:
                    continue
                
                trial = list(current)
                trial[i] = new_val
                
                # Check Sidon property
                new_set = sorted(trial)
                sums = set()
                valid = True
                for a in range(len(new_set)):
                    for b in range(a, len(new_set)):
                        s = new_set[a] + new_set[b]
                        if s in sums:
                            valid = False
                            break
                        sums.add(s)
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                trial_h = _sidon_to_heights(trial, N)
                trial_ratio = _ratio_exact(trial_h)
                
                if trial_ratio > current_ratio + 1e-10:
                    current = trial
                    current_h = trial_h
                    current_ratio = trial_ratio
                    improved = True
                    if current_ratio > best_ratio:
                        best_ratio = current_ratio
                        best_set = list(current)
                    break
            if improved:
                break
        
        if not improved:
            # Try pair swaps
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    old_i, old_j = current[i], current[j]
                    for new_i in range(N):
                        if new_i == old_i or new_i == old_j:
                            continue
                        if new_i in current:
                            continue
                        for new_j in range(N):
                            if new_j == old_i or new_j == old_j or new_j == new_i:
                                continue
                            if new_j in current:
                                continue
                            
                            trial = list(current)
                            trial[i] = new_i
                            trial[j] = new_j
                            
                            # Check Sidon property
                            new_set = sorted(trial)
                            sums = set()
                            valid = True
                            for a in range(len(new_set)):
                                for b in range(a, len(new_set)):
                                    s = new_set[a] + new_set[b]
                                    if s in sums:
                                        valid = False
                                        break
                                    sums.add(s)
                            if not valid:
                                continue
                            
                            trial_h = _sidon_to_heights(trial, N)
                            trial_ratio = _ratio_exact(trial_h)
                            
                            if trial_ratio > current_ratio + 1e-10:
                                current = trial
                                current_h = trial_h
                                current_ratio = trial_ratio
                                improved = True
                                if current_ratio > best_ratio:
                                    best_ratio = current_ratio
                                    best_set = list(current)
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_set, best_ratio


def _optimize_weights(h_base, max_iters=1000):
    """
    Optimize the weights on a fixed support (Sidon set positions).
    Use gradient-based optimization on the weights.
    """
    support = np.where(h_base > 0)[0]
    n_support = len(support)
    
    if n_support == 0:
        return h_base
    
    def _neg_log_ratio_weighted(w):
        h = np.zeros(N)
        h[support] = np.exp(w)  # Ensure positivity
        return -math.log(max(_ratio_exact(h), 1e-12))
    
    def _neg_log_ratio_weighted_jac(w):
        h = np.zeros(N)
        h[support] = np.exp(w)
        r = _ratio_exact(h)
        if r <= 0:
            return -1e12, np.zeros(n_support)
        
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros(n_support)
        for i in range(n_support):
            w_plus = w.copy()
            w_plus[i] += eps
            h_plus = np.zeros(N)
            h_plus[support] = np.exp(w_plus)
            grad[i] = (-math.log(max(_ratio_exact(h_plus), 1e-12)) + math.log(r)) / eps
        
        return -math.log(r), -grad
    
    # Start from uniform weights
    w0 = np.zeros(n_support)
    
    try:
        res = minimize(
            _neg_log_ratio_weighted,
            w0,
            method='L-BFGS-B',
            jac=_neg_log_ratio_weighted_jac,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False},
            callback=lambda x: (np.nan if _time_left() < 10.0 else x)
        )
        w_opt = res.x
        h_opt = np.zeros(N)
        h_opt[support] = np.exp(w_opt)
        return h_opt
    except Exception:
        return h_base


def _try_sparse_configurations(N, max_k=15):
    """
    Try various sparse configurations with different numbers of non-zero elements.
    """
    best_h = None
    best_v = 0.0
    
    rng = np.random.default_rng(12345)
    
    for k in range(2, min(max_k + 1, N + 1)):
        if _time_left() < 10.0:
            break
        
        # Try Sidon-based sets
        if k <= 12:
            sidon = _sidon_set(k, max_val=N-1)
            if len(sidon) == k:
                h = _sidon_to_heights(sidon, N)
                v = _ratio_exact(h)
                if v > best_v:
                    best_v = v
                    best_h = h.copy()
                
                # Local search
                if _time_left() > 15.0:
                    best_set, best_ratio_ls = _local_search_sidon(sidon, N, max_iters=100)
                    if best_ratio_ls > best_v:
                        best_v = best_ratio_ls
                        best_h = _sidon_to_heights(best_set, N)
        
        # Try random sparse sets
        for _ in range(20):
            if _time_left() < 5.0:
                break
            idx = rng.choice(N, size=k, replace=False)
            h = np.zeros(N)
            h[idx] = 1.0
            v = _ratio_exact(h)
            if v > best_v:
                best_v = v
                best_h = h.copy()
    
    return best_h, best_v


def construct():
    global _start
    _start = time.time()
    
    best_h = None
    best_v = 0.0
    
    # Phase 1: Try sparse Sidon-based configurations
    h_sparse, v_sparse = _try_sparse_configurations(N, max_k=12)
    if h_sparse is not None and v_sparse > best_v:
        best_h = h_sparse.copy()
        best_v = v_sparse
    
    # Phase 2: Try denser configurations with optimization
    
    # 2a: Random dense initializations
    rng = np.random.default_rng(42)
    
    for _ in range(30):
        if _time_left() < 15.0:
            break
        
        # Different distribution types
        choice = rng.integers(0, 4)
        if choice == 0:
            h = rng.random(N) + 0.1
        elif choice == 1:
            # Sparse with random weights
            k = rng.integers(5, 15)
            idx = rng.choice(N, size=k, replace=False)
            h = np.zeros(N)
            h[idx] = rng.uniform(0.5, 2.0, size=k)
        elif choice == 2:
            # Bump
            center = rng.uniform(0, N)
            width = rng.uniform(2.0, 8.0)
            x = np.arange(N)
            h = np.exp(-0.5 * ((x - center) / width) ** 2)
        else:
            # Cosine window
            x = np.arange(N)
            h = 0.5 * (1 + np.cos(2 * np.pi * x / N))
        
        v = _ratio_exact(h)
        if v > best_v:
            best_v = v
            best_h = h.copy()
    
    # 2b: Optimize the best sparse solution found
    if best_h is not None and _time_left() > 20.0:
        h_opt = _optimize_weights(best_h, max_iters=500)
        v_opt = _ratio_exact(h_opt)
        if v_opt > best_v:
            best_v = v_opt
            best_h = h_opt.copy()
    
    # 2c: Try a few more Sidon-based local searches
    for k in [5, 7, 9, 11]:
        if _time_left() < 15.0:
            break
        
        sidon = _sidon_set(k, max_val=N-1)
        if len(sidon) == k:
            h_init = _sidon_to_heights(sidon, N)
            v_init = _ratio_exact(h_init)
            
            if _time_left() > 10.0:
                best_set, best_ratio_ls = _local_search_sidon(sidon, N, max_iters=200)
                if best_ratio_ls > best_v:
                    best_v = best_ratio_ls
                    best_h = _sidon_to_heights(best_set, N)
    
    # Phase 3: Final polish with Nelder-Mead on exact objective
    if best_h is not None and _time_left() > 15.0:
        try:
            t_final = time.time()
            time_budget_final = _time_left() * 0.7
            
            def _neg_ratio_polish(h):
                v = _ratio_exact(h)
                if v <= 0:
                    return 1e12
                return -math.log(v)
            
            res_final = minimize(
                _neg_ratio_polish,
                best_h,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True, 'disp': False},
                callback=lambda x: (np.nan if (time.time() - t_final) > time_budget_final else x)
            )
            h_final = np.maximum(res_final.x, 0.0)
            v_final = _ratio_exact(h_final)
            if v_final > best_v:
                best_v = v_final
                best_h = h_final.copy()
        except Exception:
            pass
    
    if best_h is None:
        best_h = np.ones(N)
    
    best_h = np.maximum(best_h, 0.0)
    return [float(x) for x in best_h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio_exact(h))
    print("time =", time.time() - t0)
