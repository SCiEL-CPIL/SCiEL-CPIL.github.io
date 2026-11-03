"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(a, b, p):
    """Compute exact QAP cost for permutation p."""
    n = len(p)
    # a[i,j] * b[p[i], p[j]]
    # Vectorized: sum over i,j of a[i,j] * b[p[i],p[j]]
    # We can do: for each i, sum_j a[i,j] * b[p[i], p[j]]
    # = sum_i (a[i,:] dot b[p[i], p])
    # Using advanced indexing:
    pi = p
    # b_pi[i, j] = b[p[i], p[j]]
    # We need sum_{i,j} a[i,j] * b_pi[i,j]
    # Efficient: np.einsum or dot products
    # b_pi = b[np.ix_(p, p)]
    # cost = np.sum(a * b_pi)
    # For n=150, this is 150*150 = 22500 elements, very fast.
    b_pi = b[np.ix_(p, p)]
    return int(np.sum(a * b_pi))


def _compute_swap_delta(a, b, p, i, j):
    """
    Compute the change in cost when swapping facilities i and j in permutation p.
    p maps facility -> location. So p[i] is the location of facility i.
    Swapping means: new_p[i] = p[j], new_p[j] = p[i], others unchanged.
    
    Cost = sum_{k,l} a[k,l] * b[p[k], p[l]]
    
    After swap, only terms involving p[i] or p[j] change.
    Let ui = p[i], uj = p[j].
    
    The change involves:
    - Terms where k=i or l=i (facility i's flow)
    - Terms where k=j or l=j (facility j's flow)
    
    Specifically, the contribution of facility i to the cost (before) is:
      sum_l a[i,l] * b[ui, p[l]] + sum_k a[k,i] * b[p[k], ui]
    But note: the term a[i,i]*b[ui,ui] is counted in both if we're not careful.
    
    Actually, let's think more carefully.
    
    Cost = sum_{k,l} a[k,l] * b[p[k], p[l]]
    
    Terms that change are those where p[k] or p[l] changes, i.e., k=i or k=j or l=i or l=j.
    Since p changes only at indices i and j, the affected terms are:
    - (i, l) for all l
    - (j, l) for all l
    - (k, i) for all k
    - (k, j) for all k
    
    But (i,i), (i,j), (j,i), (j,j) are in multiple sets. Let's just compute the delta directly.
    
    Before swap:
    - For k=i: sum_l a[i,l] * b[ui, p[l]]
    - For k=j: sum_l a[j,l] * b[uj, p[l]]
    - For l=i (k != i,j): sum_{k!=i,j} a[k,i] * b[p[k], ui]
    - For l=j (k != i,j): sum_{k!=i,j} a[k,j] * b[p[k], uj]
    
    After swap (ui <-> uj):
    - For k=i: sum_l a[i,l] * b[uj, p'[l]] where p'[i]=uj, p'[j]=ui, others same
      = a[i,i]*b[uj,uj] + a[i,j]*b[uj,ui] + sum_{l!=i,j} a[i,l]*b[uj,p[l]]
    - For k=j: a[j,i]*b[ui,uj] + a[j,j]*b[ui,ui] + sum_{l!=i,j} a[j,l]*b[ui,p[l]]
    - For l=i (k!=i,j): sum_{k!=i,j} a[k,i] * b[p[k], uj]
    - For l=j (k!=i,j): sum_{k!=i,j} a[k,j] * b[p[k], ui]
    
    This is getting complex. Let me use a cleaner formulation.
    
    Let C(p) = sum_{k,l} a[k,l] * b[p[k], p[l]]
    
    Define for a fixed permutation p:
    row_sum[k] = sum_l a[k,l] * b[p[k], p[l]]  (contribution of row k)
    col_sum[l] = sum_k a[k,l] * b[p[k], p[l]]  (contribution of col l)
    
    Note: C = sum_k row_sum[k] = sum_l col_sum[l]
    Also, a[k,l]*b[p[k],p[l]] is counted in both row_sum[k] and col_sum[l].
    
    When we swap p[i] and p[j], the new cost is:
    C' = C - (terms involving old positions) + (terms involving new positions)
    
    The terms that change are exactly those where at least one of the indices k,l is i or j, 
    AND the corresponding p value changes. Since only p[i] and p[j] change, the affected 
    (k,l) pairs are those where k in {i,j} OR l in {i,j}.
    
    Let S = {i,j}
    Affected terms: (k,l) where k in S or l in S.
    
    Before: sum over affected (k,l) of a[k,l] * b[p[k], p[l]]
    After: sum over affected (k,l) of a[k,l] * b[p'[k], p'[l]]
    
    delta = After - Before
    
    We can compute this efficiently by precomputing some sums.
    """
    ui = p[i]
    uj = p[j]
    
    # We need to compute the delta. Let's use the fact that:
    # delta = sum_{k in S or l in S} a[k,l] * (b[p'[k], p'[l]] - b[p[k], p[l]])
    
    # p'[i] = uj, p'[j] = ui, p'[k] = p[k] for k not in S
    
    # Let's break it down:
    # Terms where k=i:
    #   l=i: a[i,i] * (b[uj,uj] - b[ui,ui])
    #   l=j: a[i,j] * (b[uj,ui] - b[ui,uj])
    #   l not in S: a[i,l] * (b[uj,p[l]] - b[ui,p[l]])
    # Terms where k=j:
    #   l=i: a[j,i] * (b[ui,uj] - b[uj,ui])
    #   l=j: a[j,j] * (b[ui,ui] - b[uj,uj])
    #   l not in S: a[j,l] * (b[ui,p[l]] - b[uj,p[l]])
    # Terms where k not in S, l=i:
    #   a[k,i] * (b[p[k],uj] - b[p[k],ui])
    # Terms where k not in S, l=j:
    #   a[k,j] * (b[p[k],ui] - b[p[k],uj])
    
    # Note: terms (i,i), (i,j), (j,i), (j,j) are already included above.
    # Terms where k not in S and l not in S: no change.
    
    # Let's compute each part:
    # Part 1: k=i, l=i
    d1 = a[i, i] * (b[uj, uj] - b[ui, ui])
    # Part 2: k=i, l=j
    d2 = a[i, j] * (b[uj, ui] - b[ui, uj])
    # Part 3: k=j, l=i
    d3 = a[j, i] * (b[ui, uj] - b[uj, ui])
    # Part 4: k=j, l=j
    d4 = a[j, j] * (b[ui, ui] - b[uj, uj])
    
    # Part 5: k=i, l not in S
    # sum_{l not in S} a[i,l] * (b[uj,p[l]] - b[ui,p[l]])
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    p_other = p[mask]
    d5 = np.sum(a[i, mask] * (b[uj, p_other] - b[ui, p_other]))
    
    # Part 6: k=j, l not in S
    d6 = np.sum(a[j, mask] * (b[ui, p_other] - b[uj, p_other]))
    
    # Part 7: k not in S, l=i
    d7 = np.sum(a[mask, i] * (b[p_other, uj] - b[p_other, ui]))
    
    # Part 8: k not in S, l=j
    d8 = np.sum(a[mask, j] * (b[p_other, ui] - b[p_other, uj]))
    
    return d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8


def _local_search(a, b, p, time_limit):
    """
    Efficient swap-based local search with precomputed row/col contributions.
    """
    n = len(p)
    p = p.copy()
    
    # Precompute b_p = b[np.ix_(p, p)]
    # And compute row and column sums for fast delta computation
    b_p = b[np.ix_(p, p)]
    row_sum = np.array([np.dot(a[i, :], b_p[i, :]) for i in range(n)])
    col_sum = np.array([np.dot(a[:, i], b_p[:, i]) for i in range(n)])
    
    # Actually, let's use a cleaner approach with precomputed vectors.
    # For each facility i, define:
    #   r[i] = sum_l a[i,l] * b[p[i], p[l]]  (row contribution)
    #   c[i] = sum_k a[k,i] * b[p[k], p[i]]  (col contribution)
    # Total cost = sum_i r[i] = sum_i c[i]
    
    # When swapping i and j, the change in r[i], r[j], c[i], c[j] can be computed.
    
    # Let's precompute for each i:
    #   a_i = a[i, :]  (row i of A)
    #   b_pi = b[p[i], :]  (row p[i] of B)
    #   r[i] = a_i dot b_pi
    
    # Similarly for columns.
    
    # For efficiency, let's precompute:
    #   For each location loc, b_row[loc] = b[loc, :]
    #   For each facility i, a_row[i] = a[i, :]
    
    # Then r[i] = a_row[i] dot b_row[p[i]]
    # c[i] = a_col[i] dot b_col[p[i]] where a_col[i] = a[:, i], b_col[loc] = b[:, loc]
    
    a_rows = a  # a[i, :]
    a_cols = a.T  # a[:, i]
    b_rows = b  # b[loc, :]
    b_cols = b.T  # b[:, loc]
    
    # Current row and col contributions
    r = np.array([np.dot(a_rows[i], b_rows[p[i]]) for i in range(n)])
    c = np.array([np.dot(a_cols[i], b_cols[p[i]]) for i in range(n)])
    
    current_cost = int(np.sum(r))
    
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        # Try all pairs (i, j) with i < j
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Compute delta for swapping i and j
                ui = p[i]
                uj = p[j]
                
                # The delta in total cost when swapping i and j:
                # We need to recompute r[i], r[j], c[i], c[j] after swap.
                
                # After swap: p[i] = uj, p[j] = ui
                # New r[i] = a_rows[i] dot b_rows[uj]
                # New r[j] = a_rows[j] dot b_rows[ui]
                # New c[i] = a_cols[i] dot b_cols[uj]
                # New c[j] = a_cols[j] dot b_cols[ui]
                
                # Old r[i] = a_rows[i] dot b_rows[ui]
                # Old r[j] = a_rows[j] dot b_rows[uj]
                # Old c[i] = a_cols[i] dot b_cols[ui]
                # Old c[j] = a_cols[j] dot b_cols[uj]
                
                # But wait, r and c are not independent. The total cost is sum(r) = sum(c).
                # When we swap, the total cost changes by:
                # delta = (new_r[i] + new_r[j] - old_r[i] - old_r[j])
                #       = (new_c[i] + new_c[j] - old_c[i] - old_c[j])
                # These should be equal.
                
                # Let's compute delta_r:
                new_ri = np.dot(a_rows[i], b_rows[uj])
                new_rj = np.dot(a_rows[j], b_rows[ui])
                delta_r = (new_ri + new_rj) - (r[i] + r[j])
                
                # For efficiency, we can precompute dot products.
                # But for now, let's just compute and see if it's fast enough.
                
                if delta_r < best_delta:
                    best_delta = delta_r
                    best_i = i
                    best_j = j
                
                if time.time() - start_time > time_limit:
                    break
            
            if time.time() - start_time > time_limit:
                break
        
        if best_i >= 0:
            # Perform the swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += int(best_delta)
            improved = True
            
            # Update r and c for the swapped positions
            ui, uj = p[best_i], p[best_j]
            r[best_i] = np.dot(a_rows[best_i], b_rows[ui])
            r[best_j] = np.dot(a_rows[best_j], b_rows[uj])
            c[best_i] = np.dot(a_cols[best_i], b_cols[ui])
            c[best_j] = np.dot(a_cols[best_j], b_cols[uj])
    
    return p.tolist(), current_cost


def _spectral_init(a, b):
    """
    Spectral heuristic initialization for QAP.
    Uses eigenvectors of the distance matrix to guide assignment.
    """
    n = len(a)
    
    # Compute the Laplacian or use the distance matrix directly
    # For QAP, a common spectral approach:
    # 1. Compute eigenvectors of B (distance matrix)
    # 2. Use the top few eigenvectors to create a low-dimensional embedding
    # 3. Match facilities to locations based on similar embeddings
    
    # Symmetrize B if needed
    B = (b + b.T) / 2.0
    
    # Compute eigenvalues and eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(B)
    except np.linalg.LinAlgError:
        # Fallback to random
        return np.random.permutation(n).tolist()
    
    # Use the smallest eigenvalues (most "central" directions)
    # Actually, for clustering/assignment, we want the top eigenvectors
    # that capture the most variance.
    
    # Let's use the top 10 eigenvectors (or fewer if n is small)
    k = min(10, n)
    # Sort by eigenvalue descending
    idx = np.argsort(eigvals)[::-1]
    top_eigvecs = eigvecs[:, idx[:k]]
    
    # Project facilities: for each facility i, compute its "embedding"
    # based on the flow matrix A
    # Facility i's importance in direction v: sum_j a[i,j] * v[j]
    # This gives a k-dimensional vector for each facility.
    
    facility_embed = np.zeros((n, k))
    for d in range(k):
        v = top_eigvecs[:, d]
        facility_embed[:, d] = a @ v
    
    # Project locations: for each location l, compute its embedding
    # based on the distance matrix B
    # Location l's embedding: top_eigvecs[l, :]
    location_embed = top_eigvecs
    
    # Now we need to match facilities to locations.
    # Use a simple approach: for each facility, find the most similar location.
    # Or use the Hungarian algorithm on a cost matrix.
    
    # Compute cost matrix: cost[i, l] = -dot(facility_embed[i], location_embed[l])
    # (negative because we want to maximize similarity)
    cost_matrix = -facility_embed @ location_embed.T
    
    # Use scipy's linear_sum_assignment
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    p = np.empty(n, dtype=np.int64)
    p[row_ind] = col_ind
    return p.tolist()


def _rank_init(a, b):
    """Rank-matching initialization."""
    n = len(a)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p.tolist()


def solve(a, b):
    n = len(a)
    deadline = time.time() + 140.0  # Safety margin
    
    # Generate multiple initial solutions
    candidates = []
    
    # 1. Rank-matching
    candidates.append(np.array(_rank_init(a, b), dtype=np.int64))
    
    # 2. Spectral init
    candidates.append(np.array(_spectral_init(a, b), dtype=np.int64))
    
    # 3. Random
    candidates.append(np.random.permutation(n).astype(np.int64))
    
    # Evaluate each candidate
    best_p = None
    best_cost = float('inf')
    
    for p in candidates:
        cost = _compute_cost(a, b, p)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    # Local search on the best candidate
    if time.time() < deadline - 10.0:
        p_list, cost = _local_search(a, b, best_p, deadline - time.time() - 5.0)
        if cost < best_cost:
            best_cost = cost
            best_p = np.array(p_list, dtype=np.int64)
    
    # If we have time, try local search on other candidates too
    for p in candidates:
        if np.array_equal(p, best_p):
            continue
        if time.time() > deadline - 10.0:
            break
        p_list, cost = _local_search(a, b, p, deadline - time.time() - 5.0)
        if cost < best_cost:
            best_cost = cost
            best_p = np.array(p_list, dtype=np.int64)
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
