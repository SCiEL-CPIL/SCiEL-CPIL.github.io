"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cost(a, b, p):
    """Compute the exact QAP cost for permutation p."""
    Bp = b[np.ix_(p, p)]
    return int(np.sum(a * Bp))


def _delta_swap(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j in the assignment p.
    """
    pi = p[i]
    pj = p[j]
    n = len(p)
    
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    mask[j] = False
    
    diff1 = b[pj, p[mask]] - b[pi, p[mask]]
    diff2 = b[p[mask], pj] - b[p[mask], pi]
    
    a_diff_row = a[i, mask] - a[j, mask]
    a_diff_col = a[mask, i] - a[mask, j]
    
    delta_k = np.sum(a_diff_row * diff1) + np.sum(a_diff_col * diff2)
    
    delta_self = (a[i, j] * (b[pj, pi] - b[pi, pj])
                  + a[j, i] * (b[pi, pj] - b[pj, pi])
                  + a[i, i] * (b[pj, pj] - b[pi, pi])
                  + a[j, j] * (b[pi, pi] - b[pj, pj]))
    
    return int(delta_k + delta_self)


def _local_search(a, b, p, deadline, max_passes=50):
    """
    First-improvement local search with 2-opt swaps.
    Returns (p, cost).
    """
    n = len(p)
    cost = _cost(a, b, p)
    
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                return p, cost
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    cost += delta
                    improved = True
        if not improved:
            break
    
    return p, cost


def _best_improvement_pass(a, b, p, deadline):
    """
    One pass of best-improvement: find the best swap, apply it.
    Returns (p, cost, best_delta).
    """
    n = len(p)
    cost = _cost(a, b, p)
    best_delta = 0
    best_i = -1
    best_j = -1
    
    for i in range(n):
        if time.time() > deadline:
            break
        for j in range(i + 1, n):
            delta = _delta_swap(a, b, p, i, j)
            if delta < best_delta:
                best_delta = delta
                best_i = i
                best_j = j
    
    if best_i >= 0:
        p[best_i], p[best_j] = p[best_j], p[best_i]
        cost += best_delta
    
    return p, cost, best_delta


def solve(a, b):
    n = len(a)
    t_start = time.time()
    deadline = t_start + 115.0  # safety margin before 150s kill
    
    # Precompute row sums of a for heuristic
    a_row_sums = a.sum(axis=1)
    b_row_sums = b.sum(axis=1)
    
    # --- Phase 1: Multiple heuristic initializations with local search ---
    best_p = None
    best_cost = np.inf
    
    # Heuristic 1: Rank-matching (heavy facilities to central locations)
    heavy = np.argsort(-a_row_sums, kind="stable")
    central = np.argsort(b_row_sums, kind="stable")
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central
    p1, c1 = _local_search(a, b, p1.copy(), deadline)
    if c1 < best_cost:
        best_cost = c1
        best_p = p1.copy()
    
    # Heuristic 2: Reverse rank-matching
    heavy2 = np.argsort(a_row_sums, kind="stable")
    central2 = np.argsort(-b_row_sums, kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy2] = central2
    p2, c2 = _local_search(a, b, p2.copy(), deadline)
    if c2 < best_cost:
        best_cost = c2
        best_p = p2.copy()
    
    # Heuristic 3: Random with fixed seed
    rng = np.random.default_rng(12345)
    p3 = rng.permutation(n).astype(np.int64)
    p3, c3 = _local_search(a, b, p3, deadline)
    if c3 < best_cost:
        best_cost = c3
        best_p = p3.copy()
    
    # --- Phase 2: Iterated Local Search (ILS) with perturbations ---
    # Use the best solution as starting point for ILS
    current_p = best_p.copy()
    current_cost = best_cost
    
    # Precompute for faster delta evaluation if possible
    # For n=100, each delta is O(n) vectorized ~ 100 ops
    # Full pass is 4950 deltas ~ 500K ops, very fast
    
    ilse_start = time.time()
    # Allocate time for ILS: use remaining time minus buffer
    ilse_deadline = deadline - 5.0
    
    # Number of ILS iterations we can afford
    # Each ILS iteration: perturbation + local search
    # Local search with first improvement: ~1-5 passes, each pass 4950 deltas
    # Let's estimate ~2-5ms per local search pass for n=100
    # So ~10-25ms per ILS iteration
    # In ~100s, we can do ~4000-10000 iterations
    
    iteration = 0
    while time.time() < ilse_deadline:
        iteration += 1
        
        # Perturbation: apply k random swaps (k=2 to 5)
        k = 2 + (iteration % 4)  # cycle through 2,3,4,5
        perturbed_p = current_p.copy()
        for _ in range(k):
            i, j = rng.integers(0, n, size=2)
            if i != j:
                perturbed_p[i], perturbed_p[j] = perturbed_p[j], perturbed_p[i]
        
        # Local search from perturbed solution
        perturbed_p, perturbed_cost = _local_search(a, b, perturbed_p, ilse_deadline)
        
        # Accept if better or equal (to escape plateaus, accept equal with some probability)
        if perturbed_cost < best_cost:
            best_cost = perturbed_cost
            best_p = perturbed_p.copy()
            current_p = perturbed_p
            current_cost = perturbed_cost
        elif perturbed_cost == current_cost:
            # Accept equal-cost solutions to diversify
            if rng.random() < 0.1:
                current_p = perturbed_p
                current_cost = perturbed_cost
        # Otherwise reject, keep current
    
    # --- Phase 3: Final polish with best-improvement on the best solution ---
    if time.time() < deadline - 2.0:
        polish_deadline = deadline - 1.0
        polish_p = best_p.copy()
        polish_cost = best_cost
        # Do a few best-improvement passes
        for _ in range(20):
            if time.time() > polish_deadline:
                break
            polish_p, polish_cost, bd = _best_improvement_pass(a, b, polish_p, polish_deadline)
            if bd == 0:
                break
        if polish_cost < best_cost:
            best_cost = polish_cost
            best_p = polish_p.copy()
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
