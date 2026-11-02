"""Matrix multiplication 3x3x5: tensor decomposition search.

Strategy:
- Start from the schoolbook decomposition (rank 45).
- Attempt to find a lower-rank exact decomposition using:
  1. A structured local search with column-replacement and greedy updates.
  2. A smooth optimization approach (Nelder-Mead) with rounding.
- Since exact integer/half-integer solutions are hard to find via floating-point
  optimization, we use a two-phase approach:
    1. Structured local search for sparse, low-rank solutions with small integer entries.
    2. If no improvement is found, return the schoolbook solution.

Given the known tight bounds for small matrix multiplication, rank 45 is likely
minimal for 3x3x5, but we attempt a brief search to confirm or find a surprise.
"""

import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]


def _build_tensor_indices(n, m, p):
    """Return list of (u_idx, v_idx, w_idx) for each (i,j,k)."""
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                triples.append((u_idx, v_idx, w_idx))
    return triples


def _compute_residual(U, V, W, n, m, p, rank):
    """Compute the residual tensor R where R[i,j,k] = 1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    R = np.ones((n, m, p), dtype=np.float64)
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                u_val = u_col[i * m + j]
                if u_val != 0:
                    v_slice = v_col[j * p:(j + 1) * p]
                    w_slice = w_col[i::n]  # W[k*n+i] for k=0..p-1
                    R[i, j, :] -= u_val * v_slice * w_slice
    return R


def _verify_solution(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact."""
    R = _compute_residual(U, V, W, n, m, p, rank)
    return np.max(np.abs(R)) < 1e-9


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the tensor defined by U, V, W exactly equals the matrix
    multiplication tensor.
    
    U shape: (n*m, rank)
    V shape: (m*p, rank)
    W shape: (p*n, rank)
    
    The multiplication tensor T[u, v, w] = 1 if and only if:
      u = i*m + j, v = j*p + k, w = k*n + i for some i, j, k
    and 0 otherwise.
    
    We check: sum_r U[u,r] * V[v,r] * W[w,r] == T[u,v,w] for all u,v,w.
    """
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Build the expected tensor T as a 3D array
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Compute the tensor from the decomposition
    T_decomp = np.einsum('ur,vr,wr->uvw', U_arr, V_arr, W_arr)
    
    # Check exact match (with small tolerance for floating point)
    if not np.allclose(T_decomp, T, atol=1e-9, rtol=0):
        return False
    
    # Also check that entries are integer or half-integer
    for matrix in [U_arr, V_arr, W_arr]:
        for row in matrix:
            for val in row:
                if val != 0:
                    # Check if val is integer or half-integer
                    if abs(val * 2 - round(val * 2)) > 1e-9:
                        return False
    
    return True


def _schoolbook_decomposition(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _structured_local_search(n, m, p, rank, rng, deadline):
    """
    Attempt to find an exact integer decomposition of given rank using
    structured local search with greedy updates.
    Returns (U,V,W) as numpy int arrays or None.
    """
    N = n * m
    P = m * p
    Q = p * n
    
    max_attempts = 30
    max_iters_per_attempt = 150
    
    for attempt in range(max_attempts):
        if time.time() >= deadline:
            break

        # Random initialisation with small integer entries
        U = rng.integers(-2, 3, size=(N, rank))
        V = rng.integers(-2, 3, size=(P, rank))
        W = rng.integers(-2, 3, size=(Q, rank))

        for _iter in range(max_iters_per_attempt):
            if time.time() >= deadline:
                break

            # Compute residual
            R = _compute_residual(U, V, W, n, m, p, rank)
            err = np.abs(R)
            max_err = np.max(err)

            if max_err < 1e-9:
                # Found exact solution!
                return (U.astype(int), V.astype(int), W.astype(int))

            # Simple greedy update: for each rank r, try to adjust one entry
            # to reduce the largest residual.
            ri, rj, rk = np.unravel_index(np.argmax(err), err.shape)
            target = 1.0
            current = 0.0
            for r in range(rank):
                current += U[ri * m + rj, r] * V[rj * p + rk, r] * W[rk * n + ri, r]

            deficit = target - current
            if abs(deficit) < 1e-12:
                continue

            # Try adjusting U[ri*m+rj, r] for a random r
            r = rng.integers(0, rank)
            uv = U[ri * m + rj, r]
            vv = V[rj * p + rk, r]
            ww = W[rk * n + ri, r]
            old_contrib = uv * vv * ww
            if vv * ww == 0:
                continue
            # Solve: (uv + delta) * vv * ww = old_contrib + deficit
            # delta = deficit / (vv * ww)
            delta = deficit / (vv * ww)
            new_val = uv + delta
            # Round to nearest integer
            rounded = int(round(new_val))
            if abs(rounded - uv) <= 3:  # small step
                U[ri * m + rj, r] = rounded
                # Verify improvement
                new_current = 0.0
                for r2 in range(rank):
                    new_current += U[ri * m + rj, r2] * V[rj * p + rk, r2] * W[rk * n + ri, r2]
                if abs(new_current - target) < abs(current - target):
                    pass  # keep it
                else:
                    U[ri * m + rj, r] = rounded - 1 if rounded > uv else rounded + 1

        # Try one more pass
        R = _compute_residual(U, V, W, n, m, p, rank)
        if np.max(np.abs(R)) < 1e-9:
            return (U.astype(int), V.astype(int), W.astype(int))

    return None


def _nelder_mead_search(n, m, p, target_rank, time_budget, seed=42):
    """
    Attempt to find a decomposition of exactly target_rank using Nelder-Mead
    optimization on the squared error, followed by rounding to half-integers
    and verification.
    """
    import numpy as np
    
    start_time = time.time()
    
    # Build target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    n_u = n * m
    n_v = m * p
    n_w = p * n
    
    def loss_function(params, rank):
        U = params[:n_u * rank].reshape(n_u, rank)
        V = params[n_u * rank:n_u * rank + n_v * rank].reshape(n_v, rank)
        W = params[n_u * rank + n_v * rank:].reshape(n_w, rank)
        T_rec = np.zeros((n_u, n_v, n_w))
        for r in range(rank):
            T_rec += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        diff = T_rec - T
        return np.sum(diff * diff)
    
    total_params = (n_u + n_v + n_w) * target_rank
    
    # Try a few random restarts
    for restart in range(2):
        if time.time() - start_time > time_budget:
            break
        
        np_rng = np.random.RandomState(seed + restart * 1000)
        params0 = np_rng.randn(total_params) * 0.5
        
        try:
            from scipy.optimize import minimize
            result = minimize(
                loss_function,
                params0,
                args=(target_rank,),
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if result.fun < 1e-4:
                U = result.x[:n_u * target_rank].reshape(n_u, target_rank)
                V = result.x[n_u * target_rank:n_u * target_rank + n_v * target_rank].reshape(n_v, target_rank)
                W = result.x[n_u * target_rank + n_v * target_rank:].reshape(n_w, target_rank)
                
                # Round to half-integers
                U_rounded = np.round(U * 2.0) / 2.0
                V_rounded = np.round(V * 2.0) / 2.0
                W_rounded = np.round(W * 2.0) / 2.0
                
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass
    
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 45
    
    # Get the schoolbook solution as fallback
    best_result = _schoolbook_decomposition(n, m, p)
    
    # Attempt to find a lower-rank solution
    # Allocate time for search
    search_time_budget = min(60, deadline - start_time - 5)
    
    if search_time_budget > 5:
        # First try structured local search for ranks just below schoolbook
        for rank in range(schoolbook_rank - 1, max(35, schoolbook_rank - 8), -1):
            if time.time() > deadline - 10:
                break
            
            remaining_time = deadline - 10 - time.time()
            if remaining_time < 5:
                break
            
            rng = np.random.default_rng(42 + rank)
            result = _structured_local_search(n, m, p, rank, rng, remaining_time)
            if result is not None:
                U, V, W = result
                best_result = {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
                break
        
        # If structured local search didn't find anything, try Nelder-Mead
        if best_result is _schoolbook_decomposition(n, m, p):
            remaining_time = deadline - 5 - time.time()
            if remaining_time > 5:
                for rank in range(schoolbook_rank - 1, max(38, schoolbook_rank - 5), -1):
                    if time.time() > deadline - 5:
                        break
                    
                    remaining_time = deadline - 5 - time.time()
                    if remaining_time < 5:
                        break
                    
                    result = _nelder_mead_search(n, m, p, rank, remaining_time, seed=42)
                    if result is not None:
                        best_result = result
                        break
    
    return best_result
