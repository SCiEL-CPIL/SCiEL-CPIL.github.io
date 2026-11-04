"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cost(a, b, p):
    """Exact QAP cost: sum_{i,j} a[i,j] * b[p[i], p[j]]."""
    return int(np.sum(a * b[np.ix_(p, p)]))


def _delta_swap(a, b, p, i, j):
    """
    Compute the change in cost if we swap facilities i and j.
    p is the current assignment array (indices 0..n-1 -> locations).
    Facility i is at location p[i], facility j at p[j].
    """
    li = p[i]
    lj = p[j]
    # delta = sum_k [a[i,k]*(b[lj,k] - b[li,k]) + a[j,k]*(b[li,k] - b[lj,k])
    #         + a[k,i]*(b[lj,k] - b[li,k]) + a[k,j]*(b[li,k] - b[lj,k])
    #         - a[i,j]*(b[li,lj] - b[li,li]) - a[j,i]*(b[lj,li] - b[lj,lj])
    # Simplified:
    # For k != i, j:
    #   a[i,k]*(b[lj,k]-b[li,k]) + a[k,i]*(b[lj,k]-b[li,k])
    #   + a[j,k]*(b[li,k]-b[lj,k]) + a[k,j]*(b[li,k]-b[lj,k])
    # For k == i:
    #   a[i,i]*(b[lj,i]-b[li,i]) + a[i,i]*(b[lj,i]-b[li,i]) -> but p[i] changes from li to lj
    #   Actually, let's use the standard formula:
    #   delta = (a[i,:][j] + a[:,i][j] - a[i,j] - a[j,i]) * (b[li, lj] - b[li, li])
    #           + (a[j,:][i] + a[:,j][i] - a[i,j] - a[j,i]) * (b[lj, li] - b[lj, lj])
    #           + sum_{k != i,j} (a[i,k] + a[k,i] - a[j,k] - a[k,j]) * (b[lj,k] - b[li,k])
    # This is still O(n). Let's precompute row sums or use vectorization.
    
    # Vectorized O(n) delta calculation:
    # Let d[k] = b[lj, k] - b[li, k]
    # delta = sum_k (a[i,k] + a[k,i] - a[j,k] - a[k,j]) * d[k]
    #        - a[i,j] * (b[li, lj] - b[li, li])  # correction for k=j term in the sum where a[i,j] was included
    #        - a[j,i] * (b[lj, li] - b[lj, lj])  # correction for k=i term
    # Wait, the sum includes k=i and k=j.
    # At k=i: term is (a[i,i] + a[i,i] - a[j,i] - a[i,j]) * (b[lj, i] - b[li, i])
    #   But actually when k=i, the location of facility i is changing from li to lj.
    #   The term a[i,k] becomes a[i,i] * b[lj, i] instead of a[i,i] * b[li, i]. Change: a[i,i]*(b[lj,i]-b[li,i]).
    #   The term a[k,i] = a[i,i] * b[lj, i] instead of a[i,i] * b[li, i]. Change: a[i,i]*(b[lj,i]-b[li,i]).
    #   The term a[j,k] = a[j,i] * b[li, i] instead of a[j,i] * b[lj, i]. Change: a[j,i]*(b[li,i]-b[lj,i]).
    #   The term a[k,j] = a[i,j] * b[li, i] instead of a[i,j] * b[lj, i]. Change: a[i,j]*(b[li,i]-b[lj,i]).
    #   Total for k=i: (2*a[i,i] - a[j,i] - a[i,j]) * (b[lj,i] - b[li,i])
    #   But d[i] = b[lj,i] - b[li,i]. So the formula sum_k (...) * d[k] gives:
    #   (a[i,i] + a[i,i] - a[j,i] - a[i,j]) * d[i] = (2*a[i,i] - a[j,i] - a[i,j]) * d[i]. Correct.
    # At k=j: term is (a[i,j] + a[j,i] - a[j,j] - a[j,j]) * (b[lj, j] - b[li, j])
    #   d[j] = b[lj, j] - b[li, j].
    #   Actual change:
    #   a[i,j] * b[lj, j] instead of a[i,j] * b[li, j]: change a[i,j]*(b[lj,j]-b[li,j]) = a[i,j]*d[j]
    #   a[j,i] * b[lj, j] instead of a[j,i] * b[li, j]: change a[j,i]*d[j]
    #   a[j,j] * b[li, j] instead of a[j,j] * b[lj, j]: change a[j,j]*(b[li,j]-b[lj,j]) = -a[j,j]*d[j]
    #   a[j,j] * b[li, j] instead of a[j,j] * b[lj, j]: change -a[j,j]*d[j]
    #   Total: (a[i,j] + a[j,i] - 2*a[j,j]) * d[j].
    #   Formula gives: (a[i,j] + a[j,i] - a[j,j] - a[j,j]) * d[j] = (a[i,j] + a[j,i] - 2*a[j,j]) * d[j]. Correct.
    # So the vectorized formula works directly.
    
    d = b[lj, :] - b[li, :]
    coeffs = a[i, :] + a[:, i] - a[j, :] - a[:, j]
    return int(np.dot(coeffs, d))


def _local_search(a, b, p, max_iters, deadline):
    """First-improvement local search with swap moves."""
    p = np.array(p, dtype=np.int64)
    n = len(p)
    current_cost = _cost(a, b, p)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if time.time() > deadline:
            break
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                delta = _delta_swap(a, b, p, i, j)
                if delta < 0:
                    p[i], p[j] = p[j], p[i]
                    current_cost += delta
                    improved = True
    return p, current_cost


def _simulated_annealing(a, b, p, deadline, T0=1000.0, T_end=1.0, cooling=0.999):
    """Simulated annealing with swap moves."""
    p = np.array(p, dtype=np.int64)
    n = len(p)
    current_cost = _cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    T = T0
    rng = np.random.default_rng(42)
    
    while T > T_end:
        if time.time() > deadline:
            break
        # Pick random pair
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        if i == j:
            continue
        delta = _delta_swap(a, b, p, i, j)
        new_cost = current_cost + delta
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            p[i], p[j] = p[j], p[i]
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        T *= cooling
    return best_p, best_cost


def _spectral_initialization(a, b):
    """
    Spectral initialization: use eigenvectors of the matrices to create an initial assignment.
    """
    n = len(a)
    # Center the matrices
    a_c = a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + a.mean()
    b_c = b - b.mean(axis=1, keepdims=True) - b.mean(axis=0, keepdims=True) + b.mean()
    
    # Get top eigenvectors
    try:
        w_a, v_a = np.linalg.eigh(a_c)
        w_b, v_b = np.linalg.eigh(b_c)
        # Use top k eigenvectors
        k = min(10, n)
        u_a = v_a[:, -k:]  # top k eigenvectors of a
        u_b = v_b[:, -k:]  # top k eigenvectors of b
        
        # Project into k-dimensional space
        x_a = u_a.T @ a_c @ u_a
        x_b = u_b.T @ b_c @ u_b
        
        # Get eigenvectors of the projected matrices
        w_pa, v_pa = np.linalg.eigh(x_a)
        w_pb, v_pb = np.linalg.eigh(x_b)
        
        # Sort by eigenvalue
        idx_a = np.argsort(w_pa)[::-1]
        idx_b = np.argsort(w_pb)[::-1]
        v_pa = v_pa[:, idx_a]
        v_pb = v_pb[:, idx_b]
        
        # Map facilities to locations based on similar eigenvector structure
        # For each facility, find the most similar location
        # Use the top eigenvector components
        p = np.empty(n, dtype=np.int64)
        used = np.zeros(n, dtype=bool)
        # Sort facilities by their projection onto the top eigenvector
        fac_scores = v_pa[0, :]
        loc_scores = v_pb[0, :]
        fac_order = np.argsort(-fac_scores)
        loc_order = np.argsort(loc_scores)
        for fi, li in zip(fac_order, loc_order):
            p[fi] = li
        return p
    except:
        # Fallback to rank-matching
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
        return p


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial solution: spectral + rank-matching hybrid
    p_spectral = _spectral_initialization(a, b)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p_rank = np.empty(n, dtype=np.int64)
    p_rank[heavy] = central
    
    cost_spec = _cost(a, b, p_spectral)
    cost_rank = _cost(a, b, p_rank)
    
    if cost_spec < cost_rank:
        p = p_spectral
    else:
        p = p_rank
    
    # Local search phase 1
    if time.time() < deadline - 20.0:
        p, cost = _local_search(a, b, p, max_iters=50, deadline=deadline - 20.0)
    
    # Simulated annealing for escape from local minima
    if time.time() < deadline - 5.0:
        p, cost = _simulated_annealing(a, b, p, deadline=deadline - 5.0, T0=500.0, T_end=1.0, cooling=0.9995)
    
    # Final local search polish
    if time.time() < deadline - 1.0:
        p, cost = _local_search(a, b, p, max_iters=100, deadline=deadline - 0.5)
    
    return p.tolist()
# EVOLVE-BLOCK-END
