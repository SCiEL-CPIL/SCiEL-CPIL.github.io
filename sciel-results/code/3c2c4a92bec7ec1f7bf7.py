import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """Compute the raw score: L2^2 / (L1 * max_autoconv)"""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    # Autoconvolution: (h*h)(k) = sum_i h_i * h_{i+k}
    # Using np.convolve(h, h, mode='full') gives the full convolution
    h_conv = np.convolve(h, h, mode='full')
    # The autoconvolution peak is DELTA * max(h_conv) for step functions
    # But since DELTA is constant, we can just use max(h_conv)
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _find_sidon_set(k, n):
    """
    Find a Sidon set (B_2 set) of size k from {0, 1, ..., n-1}.
    A Sidon set has all pairwise sums distinct (or at least, 
    each sum appears at most once).
    This ensures that the autoconvolution has minimal peak.
    """
    best_set = None
    best_score = 0
    
    # Try random initialization with greedy refinement
    for _ in range(200):
        # Start with a random subset
        candidate = set(np.random.choice(n, k, replace=False))
        
        # Greedy improvement
        improved = True
        while improved:
            improved = False
            # Check all pairwise sums
            sums = {}
            for i in range(len(candidate)):
                for j in range(i+1, len(candidate)):
                    s = list(candidate)[i] + list(candidate)[j]
                    if s in sums:
                        # Conflict found, try to fix
                        # Remove one of the conflicting elements
                        a = list(candidate)[i]
                        b = list(candidate)[j]
                        # Try removing b and adding a new element
                        for new_elem in range(n):
                            if new_elem not in candidate:
                                candidate.remove(b)
                                candidate.add(new_elem)
                                improved = True
                                break
                        if not improved:
                            # Try removing a
                            candidate.remove(a)
                            for new_elem in range(n):
                                if new_elem not in candidate:
                                    candidate.add(new_elem)
                                    improved = True
                                    break
                        break
                    else:
                        sums[s] = (i, j)
            # Also check if any element can be improved
            if not improved:
                # Try to find a better element to replace the worst one
                # (Heuristic: try to minimize max sum frequency)
                pass
        
        # Score: check how many sum collisions
        sums_count = {}
        elems = sorted(candidate)
        for i in range(len(elems)):
            for j in range(i, len(elems)):  # Include i=j for 2*x
                s = elems[i] + elems[j]
                sums_count[s] = sums_count.get(s, 0) + 1
        
        # Max sum multiplicity
        max_mult = max(sums_count.values()) if sums_count else 0
        # We want to minimize max_mult
        if max_mult < best_score or best_set is None:
            best_score = max_mult
            best_set = candidate
        
        # Early exit if perfect Sidon set found
        if max_mult == 1:
            break
    
    return sorted(best_set) if best_set else list(range(k))

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Strategy: Use a sparse support (Sidon-like set) to minimize convolution peak
    # while maintaining high L2/L1 ratio.
    
    # 1. Find a good Sidon set
    k_candidates = [10, 12, 14, 15, 16, 18]
    best_support = None
    best_support_score = 0
    
    for k in k_candidates:
        if time.time() > deadline - 20:
            break
        support = _find_sidon_set(k, N)
        
        # Score the support with equal weights
        h = np.zeros(N)
        h[support] = 1.0
        score = _score_raw(h)
        
        if score > best_support_score:
            best_support_score = score
            best_support = support
    
    if best_support is None:
        best_support = list(range(10))
    
    # 2. Initialize with the best support
    best_h = np.zeros(N)
    best_h[best_support] = 1.0
    best_score = _score_raw(best_h)
    
    # 3. Optimize weights on the fixed support using SLSQP
    # Objective: Maximize L2^2 / (L1 * max_conv)
    # Subject to: h_i >= 0
    
    def neg_obj(h):
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv == 0:
            return 0.0
        return -(l2_sq / (l1 * max_conv))
    
    # For optimization, fix the support and only optimize weights on those indices
    support_indices = np.array(best_support)
    n_support = len(support_indices)
    
    # Bounds for each weight
    bounds = [(0.0, 10.0)] * n_support
    
    # Initial weights
    h_init = np.ones(n_support)
    
    def obj_from_support_weights(w):
        h = np.zeros(N)
        h[support_indices] = w
        return neg_obj(h)
    
    try:
        res = minimize(
            obj_from_support_weights,
            h_init,
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        if res.fun < 0:  # Valid solution
            h_candidate = np.zeros(N)
            h_candidate[support_indices] = res.x
            score = _score_raw(h_candidate)
            if score > best_score:
                best_score = score
                best_h = h_candidate.copy()
    except Exception:
        pass
    
    # 4. Try multiple different supports
    for _ in range(30):
        if time.time() > deadline - 30:
            break
        
        k = np.random.choice(k_candidates)
        support = _find_sidon_set(k, N)
        
        h_init_full = np.zeros(N)
        h_init_full[support] = 1.0
        
        # Optimize weights
        support_indices = np.array(support)
        n_support = len(support_indices)
        bounds = [(0.0, 10.0)] * n_support
        h_init_w = np.ones(n_support)
        
        try:
            res = minimize(
                obj_from_support_weights,
                h_init_w,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-10}
            )
            if res.fun < 0:
                h_candidate = np.zeros(N)
                h_candidate[support_indices] = res.x
                score = _score_raw(h_candidate)
                if score > best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # 5. Final polish: optimize all 50 variables with SLSQP
    if time.time() < deadline - 10:
        # Normalize for better conditioning
        h_start = best_h.copy()
        if np.sum(h_start) > 0:
            h_start = h_start / np.sum(h_start)
        
        constraints = [
            {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0},
            {'type': 'ineq', 'fun': lambda h: h}
        ]
        bounds = [(0.0, 1.0)] * N
        
        try:
            res = minimize(
                neg_obj,
                h_start,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.fun < 0:
                h_candidate = res.x
                score = _score_raw(h_candidate)
                if score > best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    # Ensure non-negative
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)
