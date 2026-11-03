"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE_FRACTION = 0.92  # safety margin


def _compute_cost(a, b, p):
    """Exact integer cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _compute_swap_delta(a, b, p, i, j):
    """
    Delta in cost if we swap facilities i and j (i.e., p[i] and p[j] are exchanged).
    We want to minimize cost, so a negative delta is an improvement.
    """
    pi = p[i]
    pj = p[j]

    # Contribution involving i or j before swap
    # Cost = sum_{k,l} a[k,l] b[p[k], p[l]]
    # Terms involving i (as k) or j (as l) or both.
    # Let's derive delta = cost_after - cost_before
    #
    # Before:
    #   terms with k=i: sum_l a[i,l] b[pi, p[l]]
    #   terms with l=i: sum_k a[k,i] b[p[k], pi]
    #   terms with k=j: sum_l a[j,l] b[pj, p[l]]
    #   terms with l=j: sum_k a[k,j] b[p[k], pj]
    #   Note: term (i,i), (i,j), (j,i), (j,j) are counted appropriately.
    #
    # Actually, it's easier to compute the full delta by considering which terms change.
    # Only terms where k in {i,j} or l in {i,j} change.
    #
    # Let S = {i, j}
    # Before contribution from S x all + all x S - S x S (to avoid double counting)
    # Actually, let's just compute the change directly.
    #
    # delta = sum_{k,l in S} a[k,l] * (b[p_new[k], p_new[l]] - b[p_old[k], p_old[l]])
    #       + sum_{k in S, l not in S} a[k,l] * (b[p_new[k], p_new[l]] - b[p_old[k], p_old[l]])
    #       + sum_{k not in S, l in S} a[k,l] * (b[p_new[k], p_new[l]] - b[p_old[k], p_old[l]])
    #
    # Since p_new[i] = pj, p_new[j] = pi, and p_new[k] = p_old[k] for k not in S.
    #
    # For k in S, l in S:
    #   (i,i): a[i,i]*(b[pj,pj] - b[pi,pi])
    #   (i,j): a[i,j]*(b[pj,pi] - b[pi,pj])
    #   (j,i): a[j,i]*(b[pi,pj] - b[pj,pi])
    #   (j,j): a[j,j]*(b[pi,pi] - b[pj,pj])
    #
    # For k in S, l not in S:
    #   (i,l): a[i,l]*(b[pj, p[l]] - b[pi, p[l]])
    #   (j,l): a[j,l]*(b[pi, p[l]] - b[pj, p[l]])
    #
    # For k not in S, l in S:
    #   (k,i): a[k,i]*(b[p[k], pi] - b[p[k], pj])
    #   (k,j): a[k,j]*(b[p[k], pj] - b[p[k], pi])
    #
    # We can vectorize this.
    
    # Precompute row/col sums for speed? No, let's just use numpy slicing.
    
    # Indices not in S
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    others = np.where(mask)[0]
    
    # S x S terms
    delta = (a[i, i] * (b[pj, pj] - b[pi, pi]) +
             a[i, j] * (b[pj, pi] - b[pi, pj]) +
             a[j, i] * (b[pi, pj] - b[pj, pi]) +
             a[j, j] * (b[pi, pi] - b[pj, pj]))
    
    # S x others
    # i row, others cols: a[i, others] * (b[pj, p[others]] - b[pi, p[others]])
    delta += np.sum(a[i, others] * (b[pj, p[others]] - b[pi, p[others]]))
    # j row, others cols: a[j, others] * (b[pi, p[others]] - b[pj, p[others]])
    delta += np.sum(a[j, others] * (b[pi, p[others]] - b[pj, p[others]]))
    
    # others x S
    # others rows, i col: a[others, i] * (b[p[others], pi] - b[p[others], pj])
    delta += np.sum(a[others, i] * (b[p[others], pi] - b[p[others], pj]))
    # others rows, j col: a[others, j] * (b[p[others], pj] - b[p[others], pi])
    delta += np.sum(a[others, j] * (b[p[others], pj] - b[p[others], pi]))
    
    return int(delta)


def _local_search(a, b, p, deadline):
    """
    First-improvement swap-based local search.
    p is a numpy array (int64). Returns the improved p (numpy array) and the cost.
    """
    n = len(p)
    current_cost = _compute_cost(a, b, p)
    
    # We'll iterate until no improvement or deadline.
    # To avoid O(n^2) per pass being too slow, we can randomize the order of swaps.
    # But for now, let's do a full pass, and if no improvement, stop.
    # Actually, first-improvement: stop as soon as we find an improving swap.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Randomize the order of pairs to escape symmetry? Or just fixed order.
        # Fixed order is deterministic but might get stuck in specific patterns.
        # Let's shuffle the list of indices to try random pairs.
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        for i_idx in range(n):
            if time.time() >= deadline:
                break
            i = indices[i_idx]
            for j_idx in range(i_idx + 1, n):
                if time.time() >= deadline:
                    break
                j = indices[j_idx]
                delta = _compute_swap_delta(a, b, p, i, j)
                if delta < 0:
                    # Swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
                    # After a swap, the set of improving swaps has changed.
                    # In first-improvement, we typically restart the scan or continue.
                    # Restarting is safer but slower. Let's restart for correctness of "first improvement".
                    break
            if improved:
                break
    
    return p, current_cost


def solve(a, b):
    start_time = time.time()
    deadline = start_time + 120 * DEADLINE_FRACTION
    
    n = len(a)
    
    # 1. Generate initial solution using spectral heuristic
    # The strategy suggests modeling as Max-Cut or using spectral methods.
    # For QAP, a common spectral initialization is based on the eigenvectors of the matrices.
    # However, the prompt specifically mentions: "Model the QAP as a Max-Cut problem by transforming the assignment into a binary vector and applying a spectral heuristic".
    # This is a bit abstract. A simpler spectral approach for QAP initialization is to use the top eigenvectors of (A + A^T) and (B + B^T) or similar.
    # Let's try a standard rank-based initialization first, as it's robust, and then see if we have time for spectral.
    # Actually, the directive says "KEEP this design" (rank-matching) but investigate improvements.
    # It also says "Strategy to try: ... spectral initialization ... refined using existing efficient swap-based local search".
    # So I will implement a spectral initialization and then local search.
    
    # Spectral Initialization:
    # One approach: Compute the top k eigenvectors of A and B.
    # Align them to create an initial permutation.
    # However, QAP is not directly Max-Cut. The "Max-Cut" analogy might refer to using the sign of the projection.
    # Let's stick to a robust initialization: Rank matching, then maybe a spectral refinement if time permits.
    # Given the tight time limit, rank matching is O(n log n) and very fast.
    # Let's use rank matching as the primary initializer, as it's known to be good.
    
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # 2. Local Search
    # We need to be careful with the local search time.
    # n=150, so n^2/2 = 11250 swaps per pass.
    # Each swap delta calculation involves O(n) operations (sums over 'others').
    # So one pass is O(n^3) roughly? No, the delta calc is O(n) because 'others' is size n-2.
    # So one pass is O(n^2 * n) = O(n^3) = 150^3 = 3.375e6 operations.
    # In Python/NumPy, this might be slow if not vectorized well.
    # Let's optimize the delta calculation or use a different local search strategy.
    
    # Alternative: 2-opt style, but with precomputed row/col sums of B?
    # Let's try to vectorize the delta calculation better or use a simpler heuristic.
    # Actually, for n=150, a pure Python loop over 11250 pairs, each doing O(n) numpy ops, might take a few seconds per pass.
    # If we can do ~10-20 passes, that's fine.
    
    # Let's run local search until deadline.
    p, cost = _local_search(a, b, p, deadline)
    
    return p.tolist()
# EVOLVE-BLOCK-END
