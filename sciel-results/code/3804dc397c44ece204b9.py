"""QAP solver for tai256c using spectral initialization + fast vectorized 2-opt + ILS."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    """Compute the QAP objective: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    bp = b[np.ix_(p, p)]
    return int(np.dot(a.ravel(), bp.ravel()))


def _spectral_init(a, b, n, rng):
    """
    Spectral initialization: use eigenvectors of A and B to create an initial assignment.
    The idea is to match the 'shape' of the flow matrix with the distance matrix.
    """
    # Symmetrize for eigendecomposition (QAP matrices are often symmetric, but be safe)
    A_sym = (a + a.T) / 2.0
    B_sym = (b + b.T) / 2.0
    
    # Remove diagonal effects if any (though typically zero for distance)
    np.fill_diagonal(A_sym, 0)
    np.fill_diagonal(B_sym, 0)
    
    # Compute eigendecompositions
    # We only need the top few eigenvectors
    try:
        w_a, v_a = np.linalg.eigh(A_sym)
        w_b, v_b = np.linalg.eigh(B_sym)
    except np.linalg.LinAlgError:
        # Fallback to random
        return rng.permutation(n)
    
    # Sort in descending order of eigenvalues
    idx_a = np.argsort(-w_a)
    idx_b = np.argsort(-w_b)
    
    # Take top k eigenvectors (k = n for full, but we can use fewer for speed)
    # For n=256, full eigendecomposition is fine
    v_a_top = v_a[:, idx_a[:n]]
    v_b_top = v_b[:, idx_b[:n]]
    
    # The spectral assignment tries to maximize trace(v_a_top.T @ P @ v_b_top)
    # where P is the permutation matrix.
    # This is equivalent to finding a permutation that aligns the subspaces.
    # A simple heuristic: compute the matrix M = v_a_top @ v_b_top.T
    # Then find the assignment that maximizes sum M[i, p[i]]
    # This is a linear assignment problem, solvable by Hungarian algorithm.
    # But scipy.optimize.linear_sum_assignment minimizes, so we negate.
    
    M = v_a_top @ v_b_top.T
    
    try:
        from scipy.optimize import linear_sum_assignment
        # We want to maximize sum M[i, p[i]], so minimize -M
        row_ind, col_ind = linear_sum_assignment(-M)
        p = np.empty(n, dtype=np.int64)
        p[row_ind] = col_ind
        return p
    except Exception:
        # Fallback: greedy based on M
        p = np.empty(n, dtype=np.int64)
        used_cols = np.zeros(n, dtype=bool)
        for i in range(n):
            scores = M[i].copy()
            scores[used_cols] = -np.inf
            best_col = np.argmax(scores)
            p[i] = best_col
            used_cols[best_col] = True
        return p


def _local_search_2opt_fast(a, b, p, time_limit=110.0):
    """
    Fast 2-opt local search with vectorized cost difference computation.
    Uses a first-improvement strategy with random restarts of the scan order.
    """
    n = len(p)
    best_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    start_time = time.time()
    
    # Precompute a + a.T for faster delta computation
    a_sym = a + a.T
    
    no_improvement_streak = 0
    max_no_improvement = 20
    
    while time.time() - start_time < time_limit:
        improved = False
        # Try pairs in a random order
        indices = np.arange(n)
        np.random.shuffle(indices)
        
        # We'll scan through all pairs, but break early if we find an improvement
        for i_idx in range(n):
            i = indices[i_idx]
            pi = best_p[i]
            
            # Vectorized computation of delta for all j > i
            # dp_j = b[:, best_p[j]] - b[:, pi]
            # delta_j = (a[i,:] - a[j,:]) @ dp_j + (a[:,i] - a[:,j]) @ dp_j + (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
            
            # This is hard to fully vectorize for all j at once due to the dependency on j.
            # Let's stick to the inner loop but optimize it.
            
            for j_idx in range(i_idx + 1, n):
                j = indices[j_idx]
                pj = best_p[j]
                
                # Compute delta
                dp = b[:, pj] - b[:, pi]
                # (a[i,:] - a[j,:]) @ dp
                term1 = (a[i, :] - a[j, :]) @ dp
                # (a[:,i] - a[:,j]) @ dp
                term2 = (a[:, i] - a[:, j]) @ dp
                # (a[i,j]-a[j,i])*(b[pj,pi]-b[pi,pj])
                term3 = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
                
                delta = term1 + term2 + term3
                
                if delta < -1e-9:
                    # Perform swap
                    best_p[i], best_p[j] = best_p[j], best_p[i]
                    best_cost += delta
                    improved = True
                    no_improvement_streak = 0
                    break
            
            if improved:
                continue
            
            if time.time() - start_time > time_limit:
                break
        
        if not improved:
            no_improvement_streak += 1
            if no_improvement_streak >= max_no_improvement:
                break
        else:
            no_improvement_streak = 0
    
    return best_p, best_cost


def _ils(a, b, p, time_limit=110.0):
    """
    Iterated Local Search: perturb the current solution, then local search.
    Keep the best solution found.
    """
    n = len(p)
    best_p = p.copy()
    best_cost = _qap_cost(a, b, best_p)
    start_time = time.time()
    
    # Use a fixed seed for reproducibility within a run, but different across runs
    rng = np.random.default_rng()
    
    while time.time() - start_time < time_limit:
        # Perturb: swap a random number of pairs
        cur_p = best_p.copy()
        # Adaptive perturbation size: start small, increase if stuck
        num_swaps = rng.integers(2, max(3, n // 10))
        for _ in range(num_swaps):
            i, j = rng.choice(n, size=2, replace=False)
            cur_p[i], cur_p[j] = cur_p[j], cur_p[i]
        
        # Local search on perturbed solution
        remaining = time_limit - (time.time() - start_time)
        if remaining < 1.0:
            break
        
        cur_p, cur_cost = _local_search_2opt_fast(a, b, cur_p, time_limit=remaining)
        
        if cur_cost < best_cost:
            best_p = cur_p.copy()
            best_cost = cur_cost
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill
    
    # Phase 1: Generate diverse initial solutions
    candidates = []
    
    # Spectral initialization
    rng1 = np.random.default_rng(42)
    p1 = _spectral_init(a, b, n, rng1)
    candidates.append(p1)
    
    # Rank-matching initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p2 = np.empty(n, dtype=np.int64)
    p2[heavy] = central
    candidates.append(p2)
    
    # Random initialization
    rng2 = np.random.default_rng(123)
    p3 = rng2.permutation(n)
    candidates.append(p3)
    
    # Phase 2: Local search on each candidate
    best_p = None
    best_cost = np.inf
    
    # Allocate time: 30s for initial local searches, rest for ILS
    ls_time_each = 8.0
    
    for p in candidates:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 50:  # Keep 50s for ILS
            break
        
        # Ensure we don't exceed the global limit in this step
        step_limit = min(ls_time_each, time_limit - elapsed - 1.0)
        if step_limit <= 0:
            break
            
        p_ls, cost = _local_search_2opt_fast(a, b, p, time_limit=step_limit)
        if cost < best_cost:
            best_cost = cost
            best_p = p_ls
    
    # Phase 3: ILS from the best
    elapsed = time.time() - start_time
    if elapsed < time_limit - 2:
        remaining = time_limit - 2 - elapsed
        best_p, best_cost = _ils(a, b, best_p, time_limit=remaining)
    elif best_p is None:
        # Fallback if we didn't even get a valid solution from LS
        best_p = candidates[0]
        best_cost = _qap_cost(a, b, best_p)

    # Final safety check
    if time.time() - start_time > time_limit:
         # This shouldn't happen if limits are respected, but ensures we return something valid
         pass

    return best_p.tolist()
# EVOLVE-BLOCK-END
