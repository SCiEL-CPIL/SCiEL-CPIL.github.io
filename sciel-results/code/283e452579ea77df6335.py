"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_cost(a, b, p):
    """Compute the QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    n = len(p)
    # Efficient: sum_i a[i, p[i]] * b[p[i], p[i]] for diagonal + off-diagonal
    # Actually the full sum is: sum_{i,j} a[i,j] * b[p[i], p[j]]
    # We can compute this as: for each pair (i,j), a[i,j]*b[p[i],p[j]]
    # Vectorized: create permuted b
    pb = b[np.ix_(p, p)]
    return int(np.sum(a * pb))


def _swap_delta(a, b, p, i, j):
    """Compute the change in cost when swapping positions i and j in p.
    
    p is the assignment: facility k is at location p[k].
    Swapping p[i] and p[j] means facility i goes to location p[j] and facility j goes to location p[i].
    
    The cost is sum_{k,l} a[k,l] * b[p[k], p[l]].
    
    When we swap p[i] and p[j], the affected terms are those where k=i or l=i or k=j or l=j.
    Let's denote old_pi = p[i], old_pj = p[j].
    
    Terms involving i or j:
    - For k=i, l=j: a[i,j] * b[old_pi, old_pj] -> a[i,j] * b[old_pj, old_pi]
    - For k=j, l=i: a[j,i] * b[old_pj, old_pi] -> a[j,i] * b[old_pi, old_pj]
    - For k=i, l=i: a[i,i] * b[old_pi, old_pi] -> a[i,i] * b[old_pj, old_pj]
    - For k=j, l=j: a[j,j] * b[old_pj, old_pj] -> a[j,j] * b[old_pi, old_pi]
    - For k=i, l=m (m != i,j): a[i,m] * b[old_pi, p[m]] -> a[i,m] * b[old_pj, p[m]]
    - For k=m, l=i (m != i,j): a[m,i] * b[p[m], old_pi] -> a[m,i] * b[p[m], old_pj]
    - For k=j, l=m (m != i,j): a[j,m] * b[old_pj, p[m]] -> a[j,m] * b[old_pi, p[m]]
    - For k=m, l=j (m != i,j): a[m,j] * b[p[m], old_pj] -> a[m,j] * b[p[m], old_pi]
    
    Delta = new_cost - old_cost for these terms.
    """
    old_pi = p[i]
    old_pj = p[j]
    
    # Direct terms
    delta = 0
    # k=i, l=i
    delta += a[i, i] * (b[old_pj, old_pj] - b[old_pi, old_pi])
    # k=j, l=j
    delta += a[j, j] * (b[old_pi, old_pi] - b[old_pj, old_pj])
    # k=i, l=j
    delta += a[i, j] * (b[old_pj, old_pi] - b[old_pi, old_pj])
    # k=j, l=i
    delta += a[j, i] * (b[old_pi, old_pj] - b[old_pj, old_pi])
    
    # Terms with other indices m
    # For k=i, l=m (m != i,j): a[i,m] * (b[old_pj, p[m]] - b[old_pi, p[m]])
    # For k=m, l=i (m != i,j): a[m,i] * (b[p[m], old_pj] - b[p[m], old_pi])
    # For k=j, l=m (m != i,j): a[j,m] * (b[old_pi, p[m]] - b[old_pj, p[m]])
    # For k=m, l=j (m != i,j): a[m,j] * (b[p[m], old_pi] - b[p[m], old_pj])
    
    # We can compute these as:
    # sum_{m != i,j} a[i,m] * (b[old_pj, p[m]] - b[old_pi, p[m]])
    # + sum_{m != i,j} a[m,i] * (b[p[m], old_pj] - b[p[m], old_pi])
    # + sum_{m != i,j} a[j,m] * (b[old_pi, p[m]] - b[old_pj, p[m]])
    # + sum_{m != i,j} a[m,j] * (b[p[m], old_pi] - b[p[m], old_pj])
    
    # Let's compute using vectorized operations
    # Create a mask for m != i, j
    mask = np.ones(len(p), dtype=bool)
    mask[i] = False
    mask[j] = False
    
    # Sum over m of a[i,m] * (b[old_pj, p[m]] - b[old_pi, p[m]])
    delta += np.sum(a[i, mask] * (b[old_pj, p[mask]] - b[old_pi, p[mask]]))
    # Sum over m of a[m,i] * (b[p[m], old_pj] - b[p[m], old_pi])
    delta += np.sum(a[mask, i] * (b[p[mask], old_pj] - b[p[mask], old_pi]))
    # Sum over m of a[j,m] * (b[old_pi, p[m]] - b[old_pj, p[m]])
    delta += np.sum(a[j, mask] * (b[old_pi, p[mask]] - b[old_pj, p[mask]]))
    # Sum over m of a[m,j] * (b[p[m], old_pi] - b[p[m], old_pj])
    delta += np.sum(a[mask, j] * (b[p[mask], old_pi] - b[p[mask], old_pj]))
    
    return int(delta)


def _local_search_2swap(a, b, p, max_iter=None, time_limit=None):
    """Perform 2-swap local search. p is a numpy array."""
    n = len(p)
    p = p.copy()
    current_cost = _compute_cost(a, b, p)
    
    start_time = time.time()
    iter_count = 0
    improved = True
    
    while improved:
        improved = False
        iter_count += 1
        if max_iter is not None and iter_count > max_iter:
            break
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Try all pairs
        for i in range(n):
            for j in range(i + 1, n):
                delta = _swap_delta(a, b, p, i, j)
                if delta < 0:
                    # Accept the swap
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
    
    return p, current_cost


def _rank_matching_init(a, b):
    """Rank-matching initialization."""
    n = len(a)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _spectral_init(a, b):
    """Spectral initialization using eigenvectors."""
    n = len(a)
    # Symmetrize a
    a_sym = (a + a.T) / 2.0
    # Compute eigenvectors
    try:
        w_a, v_a = np.linalg.eigh(a_sym)
        w_b, v_b = np.linalg.eigh(b)
        # Sort by eigenvalue
        idx_a = np.argsort(-w_a)
        idx_b = np.argsort(w_b)
        v_a = v_a[:, idx_a]
        v_b = v_b[:, idx_b]
        # Match facilities to locations based on eigenvector ordering
        # For each facility i, compute a score based on its row in v_a
        # and match to locations based on v_b
        # Use the top few eigenvectors
        k = min(10, n)
        scores_a = v_a[:k, :]  # shape (k, n)
        scores_b = v_b[:k, :]  # shape (k, n)
        
        # For each facility i, compute a vector in R^k
        # For each location j, compute a vector in R^k
        # Match facility i to location j that minimizes distance
        # This is an assignment problem; use greedy or Hungarian
        
        # Simpler: rank facilities by their projection on the top eigenvector
        # and rank locations similarly
        rank_a = np.argsort(-v_a[0, :])  # facilities ordered by first eigenvector
        rank_b = np.argsort(v_b[0, :])   # locations ordered by first eigenvector
        
        p = np.empty(n, dtype=np.int64)
        p[rank_a] = rank_b
        return p
    except Exception:
        return _rank_matching_init(a, b)


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    # Get initial solution using spectral method
    p = _spectral_init(a, b)
    current_cost = _compute_cost(a, b, p)
    
    # Local search with time budget
    time_for_ls = min(100.0, deadline - (time.time() - start_time))
    if time_for_ls > 1.0:
        p, current_cost = _local_search_2swap(a, b, p, time_limit=time_for_ls)
    
    # Check if we have time for another round
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        # Try a perturbation and local search again
        p2 = p.copy()
        # Random swap perturbation
        idx = np.random.choice(n, size=2, replace=False)
        p2[idx[0]], p2[idx[1]] = p2[idx[1]], p2[idx[0]]
        cost2 = _compute_cost(a, b, p2)
        if cost2 < current_cost:
            p = p2
            current_cost = cost2
            p, current_cost = _local_search_2swap(a, b, p, time_limit=deadline - (time.time() - start_time) - 1.0)
    
    return p.tolist()
# EVOLVE-BLOCK-END
