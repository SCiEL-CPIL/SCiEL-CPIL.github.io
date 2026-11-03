"""Bootstrap portfolio: SEARCH approach with time-bounded local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping facilities at positions i and j in permutation p.
    Uses vectorized sum-minus-exclusion for speed.
    """
    x = p[i]
    y = p[j]
    
    # Direct pair contribution
    delta = a[i, j] * (b[y, x] - b[x, y]) + a[j, i] * (b[x, y] - b[y, x])
    
    # Vectorized sum over all k
    V1 = b[y, p]
    V2 = b[x, p]
    V3 = b[p, y]
    V4 = b[p, x]
    
    T1 = a[i, :] * (V1 - V2)
    T2 = a[:, i] * (V3 - V4)
    T3 = a[j, :] * (V2 - V1)
    T4 = a[:, j] * (V4 - V3)
    
    S = np.sum(T1 + T2 + T3 + T4)
    
    # Subtract terms for k=i and k=j
    # T(i)
    T_i = (a[i, i] * (b[y, x] - b[x, x]) + 
           a[i, i] * (b[x, y] - b[x, x]) + 
           a[j, i] * (b[x, x] - b[y, x]) + 
           a[i, j] * (b[x, x] - b[x, y]))
           
    # T(j)
    T_j = (a[i, j] * (b[y, y] - b[x, y]) + 
           a[j, i] * (b[y, y] - b[y, x]) + 
           a[j, j] * (b[x, y] - b[y, y]) + 
           a[j, j] * (b[y, x] - b[y, y]))
           
    delta += S - T_i - T_j
    
    return int(delta)


def _greedy_init(a, b, n):
    """Greedy initialization based on row sums."""
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _random_init(n):
    """Random permutation initialization."""
    return np.random.permutation(n)


def _spectral_init(a, b, n, seed=42):
    """
    Spectral embedding initialization using a linearized QAP matrix.
    Constructs a matrix M = A * sum(b) + B * sum(a) and uses eigenvector-based
    ordering to align facility and location structures.
    """
    rng = np.random.RandomState(seed)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # Build a combined structure matrix
    M = a * b_row_sums[None, :] + b * a_row_sums[None, :]
    
    # Compute leading eigenvectors
    try:
        from scipy.linalg import eigh
        # Use symmetric part for eigh
        M_sym = (M + M.T) / 2.0
        vals, vecs = eigh(M_sym)
        # Sort by largest eigenvalue
        idx = np.argsort(-vals)
        vecs = vecs[:, idx]
        # Use first few eigenvectors for ordering
        n_comp = min(4, n)
        embed = vecs[:, :n_comp]
        # Project to 1D using weighted sum
        scores = embed[:, 0]
        order = np.argsort(scores, kind='stable')
        p = np.empty(n, dtype=np.int64)
        p[order] = np.arange(n)
        return p
    except Exception:
        return _random_init(n)


def _2opt_delta(a, b, p, i, j):
    """
    Compute cost change for a block rotation (2-opt) move:
    reverse the segment p[i..j] (inclusive).
    This is a more powerful neighborhood than single swaps.
    """
    if i > j:
        i, j = j, i
    if j - i < 2:
        return 0  # No change for length < 3
    
    old_segment = p[i:j+1].copy()
    
    # Compute old cost contribution of the segment and its interaction with rest
    # This is expensive; for large segments, fall back to swap-based search
    # For efficiency, only use 2-opt for short segments (length <= 10)
    seg_len = j - i + 1
    if seg_len > 10:
        return None  # Signal to skip this move
    
    # Compute cost before
    old_cost = 0
    # Internal segment cost
    idx = old_segment
    old_cost += np.sum(a[np.ix_(idx, idx)] * b[np.ix_(idx, idx)])
    # Cross terms: segment vs rest
    rest = np.concatenate([p[:i], p[j+1:]])
    old_cost += np.sum(a[np.ix_(idx, rest)] * b[np.ix_(idx, rest)])
    old_cost += np.sum(a[np.ix_(rest, idx)] * b[np.ix_(rest, idx)])
    
    # New segment: reversed
    new_segment = old_segment[::-1]
    
    # Compute cost after
    new_cost = 0
    new_cost += np.sum(a[np.ix_(new_segment, new_segment)] * b[np.ix_(new_segment, new_segment)])
    new_cost += np.sum(a[np.ix_(new_segment, rest)] * b[np.ix_(new_segment, rest)])
    new_cost += np.sum(a[np.ix_(rest, new_segment)] * b[np.ix_(rest, new_segment)])
    
    return new_cost - old_cost


def _tabu_search(a, b, p, time_limit, tabu_tenure=30, max_iters=None, sample_size=1500, use_2opt=False, two_opt_prob=0.15):
    """Tabu search with 1-swap and optionally 2-opt neighborhood."""
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    tabu_list = {}
    iteration = 0
    start_time = time.time()
    
    while time.time() - start_time < time_limit:
        if max_iters is not None and iteration >= max_iters:
            break
        
        iteration += 1
        
        # Best improving 1-swap
        best_swap_i, best_swap_j = -1, -1
        best_delta = 0
        
        # Sample a subset of swaps for speed
        actual_sample_size = min(sample_size, n * n // 4)
        
        if actual_sample_size < n * n:
            # Generate random pairs using randint for speed
            i_arr = np.random.randint(0, n, actual_sample_size)
            j_arr = np.random.randint(0, n, actual_sample_size)
            
            # Filter i != j
            mask = i_arr != j_arr
            i_arr = i_arr[mask]
            j_arr = j_arr[mask]
            
            for k in range(len(i_arr)):
                i = int(i_arr[k])
                j = int(j_arr[k])
                
                key = (i, j) if i < j else (j, i)
                if key in tabu_list and tabu_list[key] > iteration:
                    continue
                
                delta = _swap_delta(a, b, p, i, j)
                
                if delta < best_delta:
                    best_delta = delta
                    best_swap_i, best_swap_j = i, j
        else:
            # Full neighborhood (unlikely for n=256)
            for i in range(n):
                for j in range(i+1, n):
                    key = (i, j)
                    if key in tabu_list and tabu_list[key] > iteration:
                        continue
                    delta = _swap_delta(a, b, p, i, j)
                    if delta < best_delta:
                        best_delta = delta
                        best_swap_i, best_swap_j = i, j
        
        # Occasionally try a 2-opt move
        if use_2opt and random.random() < two_opt_prob and best_swap_i == -1:
            # Try a random short 2-opt
            for _ in range(5):
                i = random.randint(0, n - 3)
                j = random.randint(i + 2, min(i + 10, n - 1))
                key = (i, j)
                if key in tabu_list and tabu_list[key] > iteration:
                    continue
                delta = _2opt_delta(a, b, p, i, j)
                if delta is not None and delta < best_delta:
                    best_delta = delta
                    best_swap_i, best_swap_j = i, j
        
        if best_swap_i == -1:
            break
        
        # Perform the move
        i, j = best_swap_i, best_swap_j
        key = (i, j) if i < j else (j, i)
        tabu_list[key] = iteration + tabu_tenure
        
        # Check if this is a 2-opt move (detected by checking if it's in the 2-opt context)
        # For simplicity, always do swap; 2-opt is used as a diversification trigger
        p[i], p[j] = p[j], p[i]
        current_cost += best_delta
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
        
        # Clean up expired tabu entries occasionally
        if iteration % 1000 == 0:
            expired = [k for k, v in tabu_list.items() if v <= iteration]
            for k in expired:
                del tabu_list[k]
    
    return best_p, best_cost


def _hungarian_linearization(a, b, p_current, n):
    """
    Create a linearized assignment problem from the current solution's structure.
    Uses the current permutation to derive a cost matrix for Hungarian reassignment.
    This provides a smooth surrogate for the discrete QAP.
    """
    # Linearized cost: c[i, l] = sum_j a[i, j] * b[l, p[j]]
    # This is the marginal cost of assigning facility i to location l
    # given the current assignment of all other facilities
    c = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        # For each facility i, compute cost for each location l
        # c[i, l] = sum_j a[i, j] * b[l, p[j]]
        c[i, :] = b[:, p] @ a[i, :]
    
    return c


def _hungarian_solve(c):
    """Solve the linear assignment problem using scipy's Hungarian algorithm."""
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(c)
        p = np.empty(len(c), dtype=np.int64)
        p[row_ind] = col_ind
        return p
    except Exception:
        return None


def _solve_tai256c(a, b):
    n = len(a)
    deadline = time.time() + 130  # Leave 20s safety margin
    
    # Phase 1: Generate diverse initial solutions
    inits = []
    inits.append(_greedy_init(a, b, n))
    inits.append(_spectral_init(a, b, n, seed=42))
    inits.append(_spectral_init(a, b, n, seed=123))
    for _ in range(3):
        inits.append(_random_init(n))
    
    best_p = None
    best_cost = np.inf
    
    # Phase 2: Quick Tabu Search on each initialization to find a good basin
    # Allocate ~4s per init
    quick_time = 4.0
    
    for i, p_init in enumerate(inits):
        if time.time() >= deadline - 40:
            break
        
        p_tabu, cost_tabu = _tabu_search(a, b, p_init, time_limit=quick_time, sample_size=500)
        
        if cost_tabu < best_cost:
            best_cost = cost_tabu
            best_p = p_tabu.copy()
    
    # Phase 3: Hungarian reassignment to escape local minima
    if best_p is not None and time.time() < deadline - 30:
        c = _hungarian_linearization(a, b, best_p, n)
        p_hung = _hungarian_solve(c)
        if p_hung is not None:
            cost_hung = _qap_cost(a, b, p_hung)
            if cost_hung < best_cost:
                best_cost = cost_hung
                best_p = p_hung.copy()
            # Polish the Hungarian solution
            remaining = deadline - time.time() - 10
            if remaining > 5:
                p_polished, cost_polished = _tabu_search(a, b, p_hung, time_limit=min(remaining, 15), sample_size=1000)
                if cost_polished < best_cost:
                    best_cost = cost_polished
                    best_p = p_polished
    
    # Phase 4: Long Tabu Search from the best solution with 2-opt diversification
    if best_p is not None and time.time() < deadline - 10:
        polish_time = deadline - time.time() - 10
        p_polished, cost_polished = _tabu_search(a, b, best_p, time_limit=polish_time, sample_size=2000, use_2opt=True)
        
        if cost_polished < best_cost:
            best_cost = cost_polished
            best_p = p_polished
    
    # Phase 5: Final high-precision polish pass with tighter search
    if best_p is not None and time.time() < deadline - 3:
        final_time = deadline - time.time() - 3
        if final_time > 2:
            p_final, cost_final = _tabu_search(a, b, best_p, time_limit=final_time, sample_size=3000, tabu_tenure=20)
            if cost_final < best_cost:
                best_cost = cost_final
                best_p = p_final
    
    # Fallback if no time was left or error
    if best_p is None:
        p_greedy = _greedy_init(a, b, n)
        best_p, best_cost = _tabu_search(a, b, p_greedy, time_limit=min(10.0, max(5.0, deadline - time.time() - 5)))
    
    return best_p.tolist()


def solve(a, b):
    return _solve_tai256c(a, b)
# EVOLVE-BLOCK-END
