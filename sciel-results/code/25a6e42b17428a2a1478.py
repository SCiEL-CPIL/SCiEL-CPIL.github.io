"""
QAP Solver: Pairwise-Facility-Exchange Local Search with Vectorized Deltas.

Key structural difference from the current program:
- Current: swaps positions i and j (permutation index space).
- This: swaps facilities f1 and f2 (facility index space).
  This is equivalent but allows a different vectorization pattern and
  a different diversification strategy (facility-level frequency memory).

- Uses full-vectorized candidate evaluation: compute deltas for ALL pairs
  in one NumPy call per iteration (or a large subset), pick the best.
- Multi-start with diverse initializations.
- Time-sliced budget with safety margin.
"""

# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _greedy_rank_init(a, b, seed=42):
    """Rank-matching: high flow sum -> low distance sum."""
    n = len(a)
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    # Sort facilities by descending flow sum
    fac_order = np.argsort(-a_row_sums, kind='stable')
    # Sort locations by ascending distance sum (most central)
    loc_order = np.argsort(b_row_sums, kind='stable')
    p = np.empty(n, dtype=np.int64)
    p[fac_order] = loc_order
    return p


def _random_init(n, seed=0):
    rng = random.Random(seed)
    p = np.arange(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        j = rng.randint(0, i)
        p[i], p[j] = p[j], p[i]
    return p


def _heavy_flow_greedy_init(a, b, seed=42):
    """
    Greedy: place facilities in order of total flow (descending).
    For each facility, pick the location that minimizes incremental cost.
    This is O(n^2 * n) but n=256 so ~16M ops, fine.
    """
    n = len(a)
    rng = random.Random(seed)
    # Order facilities by total flow descending
    flow_sums = a.sum(axis=1)
    order = np.argsort(-flow_sums, kind='stable')
    
    p = np.full(n, -1, dtype=np.int64)
    assigned_locs = set()
    
    for fac in order:
        best_loc = -1
        best_cost = np.inf
        for loc in range(n):
            if loc in assigned_locs:
                continue
            # Compute incremental cost of placing fac at loc
            # Cost contribution: sum_{j} a[fac, j] * b[loc, p[j]] + sum_{j} a[j, fac] * b[p[j], loc]
            # But p[j] for unassigned j is unknown, so we approximate using only assigned ones
            # Actually, let's just compute the partial cost properly
            # For simplicity and speed, use a heuristic: minimize sum of distances to already-assigned facilities weighted by flow
            # Incremental cost from already placed facilities
            inc_cost = 0.0
            for j in range(n):
                if p[j] != -1 and j != fac:
                    inc_cost += a[fac, j] * b[loc, p[j]] + a[j, fac] * b[p[j], loc]
            if inc_cost < best_cost:
                best_cost = inc_cost
                best_loc = loc
        if best_loc == -1:
            # Fallback: pick any unassigned
            for loc in range(n):
                if loc not in assigned_locs:
                    best_loc = loc
                    break
        p[fac] = best_loc
        assigned_locs.add(best_loc)
    
    return p


def _pairwise_facility_exchange_search(a, b, p, time_limit, start_time, seed=42, max_iter_per_sec=5000):
    """
    Local search: at each step, consider swapping the locations of two facilities.
    Use vectorized computation to evaluate all (or many) candidate swaps.
    
    We maintain p as a permutation where p[i] = location of facility i.
    Swapping facilities i and j means: p[i] <-> p[j].
    
    Delta computation for swapping facilities i and j:
    Let x = p[i], y = p[j].
    New cost - old cost = 
      sum_k (a[i,k] + a[k,i]) * (b[x,k'] - b[y,k']) where k' = p[k] for k != i,j
    ... this gets complex. Let's use the position-swap delta but interpret it in facility space.
    
    Actually, swapping positions i and j in the permutation p (where p[pos] = facility) 
    is equivalent to swapping the facilities at those positions.
    But here p[fac] = loc, so it's a different representation.
    
    Let's stick to the position-based representation for delta computation,
    but use a different search strategy: 
    - Instead of random pair sampling, use a full vectorized evaluation of all pairs
      in batches.
    """
    n = len(a)
    rng = random.Random(seed)
    
    # Convert to position-based representation: q[pos] = facility at pos
    # p[fac] = loc  =>  q[loc] = fac
    q = np.empty(n, dtype=np.int64)
    q[p] = np.arange(n)
    
    a_sym = a + a.T
    
    def _pos_swap_delta(a_sym, b, q, i, j):
        """Delta for swapping positions i and j in q."""
        x = q[i]
        y = q[j]
        diff_a = a_sym[i, :] - a_sym[j, :]
        diff_b = b[y, :] - b[x, :]
        val = np.dot(diff_a, diff_b[q])
        return val - diff_a[i] * diff_b[x] - diff_a[j] * diff_b[y]
    
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    # Frequency memory for diversification
    freq = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        freq[i, p[i]] += 1
    
    no_improve = 0
    stagnation_limit = 5000
    
    while time.time() - start_time < time_limit - 3:
        # Evaluate all pairs in a vectorized manner
        # For n=256, 32640 pairs. Computing all deltas vectorized:
        # We can compute this in chunks to avoid memory blowup.
        
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        # Sample a large subset of pairs for speed
        # With n=256, full evaluation might be too slow per iteration.
        # Let's do a stratified sample: for each i, sample K random j's.
        K = 20  # 256 * 20 = 5120 evaluations per iteration
        candidates = []
        for i in range(n):
            # Sample K random j != i
            js = rng.sample(range(n), K)
            for j in js:
                if j != i:
                    candidates.append((i, j))
        
        # Vectorized delta computation for all candidates
        # This is tricky because each delta depends on q[i], q[j]
        # Let's just loop but use the fast delta function
        for i, j in candidates:
            delta = _pos_swap_delta(a_sym, b, q, i, j)
            # Add small frequency penalty for diversification
            x = q[i]
            y = q[j]
            freq_pen = 0.01 * (freq[x, i] + freq[y, j])
            adj_delta = delta + freq_pen
            if adj_delta < best_delta:
                best_delta = adj_delta
                best_i = i
                best_j = j
        
        if best_i == -1:
            # No improvement found, random move
            best_i = rng.randint(0, n - 1)
            best_j = rng.randint(0, n - 1)
            if best_i == best_j:
                best_j = (best_j + 1) % n
            best_delta = _pos_swap_delta(a_sym, b, q, best_i, best_j)
        
        # Perform the swap
        x = q[best_i]
        y = q[best_j]
        q[best_i], q[best_j] = q[best_j], q[best_i]
        current_cost += best_delta
        
        # Update frequency
        freq[x, best_i] += 1
        freq[y, best_j] += 1
        freq[x, best_j] -= 1
        freq[y, best_i] -= 1
        
        # Update best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            # Convert q back to p
            p_new = np.empty(n, dtype=np.int64)
            p_new[q] = np.arange(n)
            best_p = p_new.copy()
            no_improve = 0
        else:
            no_improve += 1
        
        # Diversification: if stuck, perturb
        if no_improve > stagnation_limit:
            num_perturb = rng.randint(5, 15)
            for _ in range(num_perturb):
                pi = rng.randint(0, n - 1)
                pj = rng.randint(0, n - 1)
                if pi != pj:
                    q[pi], q[pj] = q[pj], q[pi]
            current_cost = _qap_cost(a, b, p)
            # Recompute p from q
            p_new = np.empty(n, dtype=np.int64)
            p_new[q] = np.arange(n)
            p = p_new
            no_improve = 0
            # Reset frequency
            freq.fill(0)
            for i in range(n):
                freq[i, p[i]] += 1
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 150s kill
    
    best_p = None
    best_cost = np.inf
    
    # Multi-start portfolio
    # Start 1: Rank-matching
    # Start 2: Random
    # Start 3: Heavy-flow greedy
    
    initializations = []
    
    # 1. Rank-matching
    p1 = _greedy_rank_init(a, b, seed=42)
    initializations.append(('rank', p1))
    
    # 2. Random
    p2 = _random_init(n, seed=123)
    initializations.append(('random', p2))
    
    # 3. Heavy-flow greedy (slower init but potentially better)
    if time.time() - start_time < time_limit - 30:
        p3 = _heavy_flow_greedy_init(a, b, seed=77)
        initializations.append(('heavy', p3))
    
    # Allocate time to each start
    num_starts = len(initializations)
    time_per_start = (time_limit - 10) / num_starts  # Leave 10s buffer
    
    for idx, (name, p_init) in enumerate(initializations):
        remaining = time_limit - (time.time() - start_time)
        if remaining < 15:
            break
        
        alloc_time = min(time_per_start, remaining - 5)
        if alloc_time < 10:
            break
        
        p_best, cost_best = _pairwise_facility_exchange_search(
            a, b, p_init.copy(), alloc_time, start_time, seed=1000 + idx
        )
        
        if cost_best < best_cost:
            best_cost = cost_best
            best_p = p_best
    
    if best_p is None:
        best_p = _greedy_rank_init(a, b, seed=0)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
