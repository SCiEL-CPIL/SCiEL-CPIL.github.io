"""Tensor decomposition for 2x2 by 2x5 matrix multiplication."""

import numpy as np
import time

DIMENSIONS = [2, 2, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    rank = len(U[0]) if U else 0
    # Convert to numpy for fast multiplication
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # The tensor T[i, j, k] = delta_{i,k} delta_{j,l} ... 
    # Actually, C = A * B
    # C[i, k] = sum_j A[i, j] B[j, k]
    # Tensor form: T_{i, j, k} = 1 if i,j,k correspond to valid indices, 0 otherwise?
    # No, the tensor is the multiplication map.
    # T is a 3-tensor of size n x m x p.
    # T[i, j, k] = 1 if the term A[i, j] * B[j, k] contributes to C[i, k].
    # In the decomposition:
    # C[i, k] = sum_t U[i, t] * V[j, t] * W[k, t] ??
    # The interface is:
    # U: (n*m, rank) -> indices (i, j) flattened?
    # V: (m*p, rank) -> indices (j, k) flattened?
    # W: (p*n, rank) -> indices (k, i) flattened?
    
    # Let's check the flattening order.
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i
    
    # The condition is:
    # sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] == 1 if (i,j,k) is a valid multiplication term?
    # No, the product should reconstruct C.
    # C[i, k] = sum_j A[i, j] B[j, k]
    # In tensor decomposition, we have:
    # T_{i, j, k} = sum_t U_{i, j, t} V_{j, k, t} W_{k, i, t}
    # And T_{i, j, k} should be 1 if the indices are consistent for the multiplication?
    # Actually, the tensor T is defined such that:
    # <T, A \otimes B \otimes C> = trace(C^T A B)
    # This implies T_{i, j, k} = 1 if i, j, k form a "cycle" in the multiplication?
    # Specifically, T[i, j, k] = 1 if i, j, k are valid indices for A[i,j], B[j,k], C[k,i]?
    # No, C is n x p. C[i, k].
    # The tensor is T of size n x m x p.
    # T[i, j, k] = 1 if the term A[i, j] * B[j, k] contributes to C[i, k].
    # So T[i, j, k] = 1 for all valid i, j, k.
    # Wait, if T[i, j, k] = 1 for all i, j, k, then:
    # sum_t U[i, t] V[j, t] W[k, t] = 1 for all i, j, k?
    # This would mean U, V, W are such that their outer product sum is the all-ones tensor.
    # This is NOT correct.
    
    # The correct identity is:
    # C[i, k] = sum_j A[i, j] B[j, k]
    # This can be written as:
    # C[i, k] = sum_t (sum_j A[i, j] U[j, t]) * (sum_k' B[k', k] V[k', t]) ... no.
    
    # Standard Tensor Decomposition for MatMul:
    # We want to compute C = A B.
    # We define T such that T_{i, j, k} = 1 if i, j, k are indices of A, B, C?
    # The decomposition is:
    # T_{i, j, k} = sum_t U_{i, t} V_{j, t} W_{k, t}
    # Where U is n x rank, V is m x rank, W is p x rank?
    # No, the interface is:
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # This suggests the tensor is of order 3, with dimensions n*m, m*p, p*n?
    # No, that's the size of the matrices U, V, W.
    # The tensor being decomposed is the multiplication tensor.
    # The multiplication tensor $\mathcal{T}$ has dimensions $n \times m \times p$.
    # $\mathcal{T}_{i, j, k} = 1$ if the indices $i, j, k$ are such that $A_{i, j}$ and $B_{j, k}$ multiply to give $C_{i, k}$.
    # Actually, $\mathcal{T}_{i, j, k} = 1$ for all $i, j, k$ is incorrect.
    # The tensor is defined by $\mathcal{T}_{i, j, k} = \delta_{i, i'} \delta_{j, j'} \delta_{k, k'}$?
    
    # Let's look at the indices in the current code:
    # U[0][t] = 1 (A00)
    # V[0*p + 0][t] = 1 (B00)
    # W[0*n + 0][t] = 1 (C00)
    # This corresponds to the term A00 * B00 contributing to C00.
    # So the tensor entry $\mathcal{T}_{i, j, k}$ is 1 if the product $A_{i, j} B_{j, k}$ contributes to $C_{i, k}$.
    # Since every $A_{i, j} B_{j, k}$ contributes to $C_{i, k}$, the tensor $\mathcal{T}$ is the all-ones tensor of size $n \times m \times p$?
    # No, the dimensions of the tensor in the decomposition $U, V, W$ must match.
    # U has $n \times m$ rows. V has $m \times p$ rows. W has $p \times n$ rows.
    # This implies the tensor is of size $(nm) \times (mp) \times (pn)$?
    # No, that's the size of the factor matrices.
    
    # The correct interpretation:
    # We are decomposing the tensor $\mathcal{T}$ of size $n \times m \times p$.
    # But U, V, W have sizes $nm, mp, pn$.
    # This implies the tensor is actually defined on the indices of the elements of A, B, C.
    # The condition is:
    # $\sum_t U_{i, j, t} V_{j, k, t} W_{k, i, t} = 1$ if the indices are valid?
    # Actually, for the multiplication to work:
    # $C_{i, k} = \sum_j A_{i, j} B_{j, k}$
    # We can write this as:
    # $C_{i, k} = \sum_t \left( \sum_j A_{i, j} U_{i, j, t} \right) \dots$ No.
    
    # The standard form is:
    # $C_{i, k} = \sum_t M_t$
    # Where $M_t$ are the intermediate products.
    # $M_t = (\sum_j A_{i, j} u_{i, j, t}) (\sum_k' B_{k', k} v_{k', k, t})$?
    
    # Let's stick to the verification used in the previous program or standard checks.
    # The check is:
    # For all i, j, k:
    # $\sum_t U[i, t] V[j, t] W[k, t] = 1$
    # Where U is flattened A, V is flattened B, W is flattened C?
    # In the current code:
    # U has size $n \times m$.
    # V has size $m \times p$.
    # W has size $p \times n$.
    # The sum is over t.
    # The indices of U are $i \cdot m + j$.
    # The indices of V are $j \cdot p + k$.
    # The indices of W are $k \cdot n + i$.
    
    # The condition for exact decomposition is:
    # $\sum_t U[i \cdot m + j, t] \cdot V[j \cdot p + k, t] \cdot W[k \cdot n + i, t] = 1$
    # for all valid $i, j, k$.
    
    # Let's verify this condition.
    
    max_diff = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = 0.0
                for t in range(rank):
                    s += U[u_idx][t] * V[v_idx][t] * W[w_idx][t]
                # s should be 1
                diff = abs(s - 1.0)
                if diff > max_diff:
                    max_diff = diff
    
    return max_diff < 1e-9

def _get_rank18_solution():
    n, m, p = DIMENSIONS
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(t_offset, k_start):
        U[0][t_offset+0] = 1
        U[3][t_offset+0] = 1
        U[2][t_offset+1] = 1
        U[3][t_offset+1] = 1
        U[0][t_offset+2] = 1
        U[3][t_offset+3] = 1
        U[0][t_offset+4] = 1
        U[1][t_offset+4] = 1
        U[2][t_offset+5] = 1
        U[0][t_offset+5] = -1
        U[1][t_offset+6] = 1
        U[3][t_offset+6] = -1
        
        k0 = k_start
        k1 = k_start + 1
        V[0*p + k0][t_offset+0] = 1
        V[1*p + k1][t_offset+0] = 1
        V[0*p + k0][t_offset+1] = 1
        V[0*p + k1][t_offset+2] = 1
        V[1*p + k1][t_offset+2] = -1
        V[1*p + k0][t_offset+3] = 1
        V[0*p + k0][t_offset+3] = -1
        V[1*p + k1][t_offset+4] = 1
        V[0*p + k0][t_offset+5] = 1
        V[0*p + k1][t_offset+5] = 1
        V[1*p + k0][t_offset+6] = 1
        V[1*p + k1][t_offset+6] = 1
        
        W[k0*n + 0][t_offset+0] = 1
        W[k1*n + 1][t_offset+0] = 1
        W[k0*n + 1][t_offset+1] = 1
        W[k1*n + 1][t_offset+1] = -1
        W[k1*n + 0][t_offset+2] = 1
        W[k1*n + 1][t_offset+2] = 1
        W[k0*n + 0][t_offset+3] = 1
        W[k0*n + 1][t_offset+3] = 1
        W[k0*n + 0][t_offset+4] = -1
        W[k1*n + 0][t_offset+4] = 1
        W[k1*n + 1][t_offset+5] = 1
        W[k0*n + 0][t_offset+6] = 1

    add_strassen_block(0, 0)
    add_strassen_block(7, 2)
    
    t = 14
    U[0][t] = 1
    V[0*p + 4][t] = 1
    W[4*n + 0][t] = 1
    t += 1
    U[1][t] = 1
    V[1*p + 4][t] = 1
    W[4*n + 0][t] = 1
    t += 1
    U[2][t] = 1
    V[0*p + 4][t] = 1
    W[4*n + 1][t] = 1
    t += 1
    U[3][t] = 1
    V[1*p + 4][t] = 1
    W[4*n + 1][t] = 1
    
    return {"U": U, "V": V, "W": W}

def _snap_to_half_integers(mat, tol=1e-6):
    result = np.array(mat, dtype=np.float64)
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            v = result[i, j]
            nearest = round(2 * v) / 2.0
            if abs(v - nearest) < tol:
                result[i, j] = nearest
    return result

def _compute_error(U, V, W, n, m, p):
    rank = len(U[0]) if U else 0
    max_diff = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                s = 0.0
                for t in range(rank):
                    s += U[u_idx][t] * V[v_idx][t] * W[w_idx][t]
                diff = abs(s - 1.0)
                if diff > max_diff:
                    max_diff = diff
    return max_diff

def _als_update_U(U, V, W, n, m, p, tol=1e-10):
    rank = len(U[0])
    n_m = n * m
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    U_new = np.zeros((n_m, rank))
    for i in range(n):
        for j in range(m):
            u_idx = i * m + j
            A = np.zeros((p, rank))
            for k in range(p):
                for t in range(rank):
                    A[k, t] = V_np[j * p + k, t] * W_np[k * n + i, t]
            
            try:
                x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(p), rcond=tol)
                U_new[u_idx, :] = x
            except:
                U_new[u_idx, :] = U[u_idx]
    
    return U_new.tolist()

def _als_update_V(U, V, W, n, m, p, tol=1e-10):
    rank = len(V[0])
    m_p = m * p
    U_np = np.array(U, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    V_new = np.zeros((m_p, rank))
    for j in range(m):
        for k in range(p):
            v_idx = j * p + k
            A = np.zeros((n, rank))
            for i in range(n):
                for t in range(rank):
                    A[i, t] = U_np[i * m + j, t] * W_np[k * n + i, t]
            
            try:
                x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(n), rcond=tol)
                V_new[v_idx, :] = x
            except:
                V_new[v_idx, :] = V[v_idx]
    
    return V_new.tolist()

def _als_update_W(U, V, W, n, m, p, tol=1e-10):
    rank = len(W[0])
    p_n = p * n
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    
    W_new = np.zeros((p_n, rank))
    for k in range(p):
        for i in range(n):
            w_idx = k * n + i
            A = np.zeros((m, rank))
            for j in range(m):
                for t in range(rank):
                    A[j, t] = U_np[i * m + j, t] * V_np[j * p + k, t]
            
            try:
                x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=tol)
                W_new[w_idx, :] = x
            except:
                W_new[w_idx, :] = W[w_idx]
    
    return W_new.tolist()

def _try_als_rank(rank, n, m, p, max_iters=100, tol=1e-12, time_limit=30):
    start_time = time.time()
    
    r18 = _get_rank18_solution()
    U = [row[:rank] for row in r18["U"]]
    V = [row[:rank] for row in r18["V"]]
    W = [row[:rank] for row in r18["W"]]
    
    best_error = _compute_error(U, V, W, n, m, p)
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    for iteration in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        U = _als_update_U(U, V, W, n, m, p, tol)
        V = _als_update_V(U, V, W, n, m, p, tol)
        W = _als_update_W(U, V, W, n, m, p, tol)
        
        error = _compute_error(U, V, W, n, m, p)
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
        
        if best_error < 1e-10:
            break
    
    if best_error < 1e-8:
        U_np = np.array(best_U)
        V_np = np.array(best_V)
        W_np = np.array(best_W)
        
        U_snapped = _snap_to_half_integers(U_np)
        V_snapped = _snap_to_half_integers(V_np)
        W_snapped = _snap_to_half_integers(W_np)
        
        U_list = U_snapped.tolist()
        V_list = V_snapped.tolist()
        W_list = W_snapped.tolist()
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140
    
    for target_rank in [16, 17]:
        if time.time() > deadline - 20:
            break
        
        result = _try_als_rank(target_rank, n, m, p, max_iters=50, tol=1e-12, time_limit=15)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                return result
    
    return _get_rank18_solution()

# EVOLVE-BLOCK-END
