import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Compute the score: L2^2 / (L1 * Sup(autoconv))
    For step heights h on intervals of width DELTA:
    L1 = DELTA * sum(h)
    L2^2 = DELTA * sum(h^2)
    autoconv_max = DELTA^2 * max(conv(h, h))
    
    Ratio = (DELTA * sum(h^2)) / (DELTA * sum(h) * DELTA^2 * max_conv)
          = sum(h^2) / (DELTA^2 * sum(h) * max_conv)
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max)) / (DELTA**2)

def _sidon_sets(n, max_k=15):
    """
    Generate Sidon (B_2) sets of size up to max_k within [0, n).
    A Sidon set has all pairwise sums (with repetition) distinct.
    Uses greedy construction: start with smallest, add next element that
    doesn't create sum collisions.
    """
    sidon_sets = []
    
    def is_sidon(s):
        """Check if set s is a Sidon set."""
        s = sorted(s)
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sm = s[i] + s[j]
                if sm in sums:
                    return False
                sums.add(sm)
        return True
    
    # Generate Sidon sets of various sizes using greedy + random restarts
    for k in range(2, max_k + 1):
        found = set()
        # Try multiple random orderings
        for attempt in range(50):
            np.random.seed(attempt * 1000 + k)
            # Greedy: start with 0, then add elements in increasing order
            # that don't violate Sidon property
            s = [0]
            for candidate in range(1, n):
                if len(s) >= k:
                    break
                test_s = s + [candidate]
                if is_sidon(test_s):
                    s.append(candidate)
            if len(s) == k:
                key = tuple(sorted(s))
                if key not in found:
                    found.add(key)
                    sidon_sets.append(list(s))
        
        # Also try random-based Sidon construction
        for attempt in range(30):
            np.random.seed(attempt * 7000 + k * 13)
            s = [0]
            available = list(range(1, n))
            np.random.shuffle(available)
            for candidate in available:
                if len(s) >= k:
                    break
                test_s = s + [candidate]
                if is_sidon(test_s):
                    s.append(candidate)
            if len(s) == k:
                key = tuple(sorted(s))
                if key not in found:
                    found.add(key)
                    sidon_sets.append(list(s))
    
    return sidon_sets

def _optimize_weights_on_support(support, n_total, deadline):
    """
    Given a support (list of indices), optimize the weights h_i at those positions
    to maximize the score. Uses geometric progression initialization + SLSQP refinement.
    """
    k = len(support)
    support = sorted(support)
    
    def make_h(weights):
        h = np.zeros(n_total)
        for i, idx in enumerate(support):
            h[idx] = weights[i]
        return h
    
    best_h = None
    best_score = -1
    
    # Try geometric progressions with various bases
    for r in np.linspace(0.3, 2.5, 40):
        weights = np.array([r**i for i in range(k)])
        h = make_h(weights)
        s = _score_raw(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Try reverse geometric
    for r in np.linspace(0.3, 2.5, 40):
        weights = np.array([r**(k-1-i) for i in range(k)])
        h = make_h(weights)
        s = _score_raw(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Try uniform
    weights = np.ones(k)
    h = make_h(weights)
    s = _score_raw(h)
    if s > best_score:
        best_score = s
        best_h = h.copy()
    
    # SLSQP refinement on weights
    if time.time() > deadline:
        return best_h, best_score
    
    def neg_obj_w(w):
        w = np.clip(w, 1e-10, None)
        h = make_h(w)
        s = _score_raw(h)
        return -s
    
    bounds_w = [(1e-10, None)] * k
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    
    # Start from best geometric init
    w_init = np.zeros(k)
    for i, idx in enumerate(support):
        w_init[i] = best_h[idx]
    s_init = np.sum(w_init)
    if s_init > 0:
        w_init = w_init / s_init
    
    try:
        res = minimize(neg_obj_w, w_init, method='SLSQP', bounds=bounds_w,
                       constraints=constraints, options={'maxiter': 300, 'ftol': 1e-10})
        if res.success:
            w_opt = np.clip(res.x, 1e-10, None)
            h_opt = make_h(w_opt)
            s_opt = _score_raw(h_opt)
            if s_opt > best_score:
                best_score = s_opt
                best_h = h_opt
    except Exception:
        pass
    
    # Try a few random weight inits
    for _ in range(5):
        if time.time() > deadline:
            break
        w_rand = np.random.rand(k)
        w_rand /= np.sum(w_rand)
        try:
            res = minimize(neg_obj_w, w_rand, method='SLSQP', bounds=bounds_w,
                           constraints=constraints, options={'maxiter': 200, 'ftol': 1e-9})
            if res.success:
                w_opt = np.clip(res.x, 1e-10, None)
                h_opt = make_h(w_opt)
                s_opt = _score_raw(h_opt)
                if s_opt > best_score:
                    best_score = s_opt
                    best_h = h_opt
        except Exception:
            pass
    
    return best_h, best_score

def _try_support_swap(h_best, support, deadline):
    """
    Try swapping one element of the support with a nearby non-support element
    to explore near-Sidon structures.
    """
    best_h = h_best.copy()
    best_score = _score_raw(h_best)
    
    support_set = set(support)
    non_support = [i for i in range(N) if i not in support_set]
    
    # Try swapping each support element with a few nearby non-support elements
    for s_idx in support:
        if time.time() > deadline:
            break
        # Try nearby positions
        for offset in [-3, -2, -1, 1, 2, 3]:
            new_pos = s_idx + offset
            if 0 <= new_pos < N and new_pos not in support_set:
                new_support = list(support)
                new_support.remove(s_idx)
                new_support.append(new_pos)
                new_support = sorted(new_support)
                
                # Check if still Sidon (or close)
                def is_sidon(s):
                    s = sorted(s)
                    sums = set()
                    for i in range(len(s)):
                        for j in range(i, len(s)):
                            sm = s[i] + s[j]
                            if sm in sums:
                                return False
                            sums.add(sm)
                    return True
                
                if not is_sidon(new_support):
                    continue
                
                # Optimize weights on new support
                h_new, s_new = _optimize_weights_on_support(new_support, N, deadline)
                if h_new is not None and s_new > best_score:
                    best_score = s_new
                    best_h = h_new
                    support = new_support
    
    return best_h, best_score, support

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    # Generate Sidon sets
    sidon_sets = _sidon_sets(N, max_k=15)
    
    best_h = np.ones(N) / N
    best_score = _score_raw(best_h)
    
    # Optimize weights on each Sidon set
    for sidon in sidon_sets:
        if time.time() > deadline - 5:
            break
        h_opt, s_opt = _optimize_weights_on_support(sidon, N, deadline)
        if h_opt is not None and s_opt > best_score:
            best_score = s_opt
            best_h = h_opt
    
    # Try support swaps on the best result
    # Extract support from best_h
    support = [i for i in range(N) if best_h[i] > 1e-8]
    if len(support) >= 2:
        h_new, s_new, support_new = _try_support_swap(best_h, support, deadline)
        if s_new > best_score:
            best_score = s_new
            best_h = h_new
    
    # Final refinement: SLSQP on the full vector
    if time.time() < deadline - 10:
        def neg_obj_full(h):
            h = np.clip(h, 0, None)
            s = _score_raw(h)
            return -s
        
        bounds_full = [(0.0, None)] * N
        constraints_full = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0}]
        
        h_init = best_h.copy()
        s_init = np.sum(h_init)
        if s_init > 0:
            h_init = h_init / s_init
        
        try:
            res = minimize(neg_obj_full, h_init, method='SLSQP', bounds=bounds_full,
                           constraints=constraints_full, options={'maxiter': 500, 'ftol': 1e-10})
            if res.success:
                h_candidate = np.clip(res.x, 0, None)
                s_cand = np.sum(h_candidate)
                if s_cand > 0:
                    h_candidate = h_candidate / s_cand
                s_candidate = _score_raw(h_candidate)
                if s_candidate > best_score:
                    best_score = s_candidate
                    best_h = h_candidate
        except Exception:
            pass
    
    # Ensure non-negativity
    best_h = np.clip(best_h, 0, None)
    
    return list(best_h)
