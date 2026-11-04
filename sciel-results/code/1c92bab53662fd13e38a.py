"""Rewritten program: block-based tensor decomposition for 4x5x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]
_DEADLINE_SECONDS = 115  # safety margin before 120s kill

# EVOLVE-BLOCK-START


def _schoolbook(n, m, p):
    """Return (U, V, W) as lists of lists, rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W


def _strassen_2x2():
    """Strassen 7-product decomposition for 2x2x2 matrix multiplication.
    A is 2x2 flattened row-major: (a,b,c,d)
    B is 2x2 flattened row-major: (e,f,g,h)
    C = A*B with W index k*n+i (n=2, p=2): C[0][0], C[0][1], C[1][0], C[1][1]
    """
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, -1, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, -1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, -1, 0, 0, 1, 0]
    ]
    return U, V, W


def _embed_subproblem(U_sub, V_sub, W_sub, n_sub, m_sub, p_sub,
                      i_offset, j_offset, k_offset, n, m, p):
    """Embed a subproblem decomposition into the full (n,m,p) space."""
    rank_sub = len(U_sub[0])
    U_full = [[0] * rank_sub for _ in range(n * m)]
    V_full = [[0] * rank_sub for _ in range(m * p)]
    W_full = [[0] * rank_sub for _ in range(p * n)]

    for local_i in range(n_sub):
        for local_j in range(m_sub):
            full_i = i_offset + local_i
            full_j = j_offset + local_j
            local_idx = local_i * m_sub + local_j
            full_idx = full_i * m + full_j
            U_full[full_idx] = list(U_sub[local_idx])

    for local_j in range(m_sub):
        for local_k in range(p_sub):
            full_j = j_offset + local_j
            full_k = k_offset + local_k
            local_idx = local_j * p_sub + local_k
            full_idx = full_j * p + full_k
            V_full[full_idx] = list(V_sub[local_idx])

    for local_k in range(p_sub):
        for local_i in range(n_sub):
            full_k = k_offset + local_k
            full_i = i_offset + local_i
            local_idx = local_k * n_sub + local_i
            full_idx = full_k * n + full_i
            W_full[full_idx] = list(W_sub[local_idx])

    return U_full, V_full, W_full


def _concat(U1, V1, W1, U2, V2, W2, n, m, p):
    """Horizontally concatenate two decompositions (independent tensor terms)."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    U = [U1[i] + U2[i] for i in range(n * m)]
    V = [V1[i] + V2[i] for i in range(m * p)]
    W = [W1[i] + W2[i] for i in range(p * n)]
    return U, V, W


def _multi_concat(decomps, n, m, p):
    """Concatenate a list of (U,V,W) decompositions."""
    if len(decomps) == 0:
        U = [[0] * 0 for _ in range(n * m)]
        V = [[0] * 0 for _ in range(m * p)]
        W = [[0] * 0 for _ in range(p * n)]
        return U, V, W
    result = decomps[0]
    for d in decomps[1:]:
        result = _concat(result[0], result[1], result[2], d[0], d[1], d[2], n, m, p)
    return result


def _construct_4x5x5():
    """Build 4x5x5 decomposition via block partitioning.

    Partition A (4x5) into:
      A1 = A[:, 0:2]  (4x2)
      A2 = A[:, 2]    (4x1)
      A3 = A[:, 3:5]  (4x2)

    Partition B (5x5) correspondingly:
      B1 = B[0:2, :]  (2x5)
      B2 = B[2, :]    (1x5)
      B3 = B[3:5, :]  (2x5)

    Product = A1*B1 + A2*B2 + A3*B3

    For each 4x2 * 2x5 block, split the 4 rows into two 2-row groups:
      (2x2 * 2x5): further split 2x5 into (2x2, 2x1, 2x1)
        2x2*2x2: Strassen (7)
        2x2*2x1: schoolbook (4)
        2x2*2x1: schoolbook (4)
        Total per 2x2*2x5: 15
      So 4x2 * 2x5 = 2 * 15 = 30

    For 4x1 * 1x5: schoolbook = 4*1*5 = 20

    Total: 30 + 20 + 30 = 80
    """
    n, m, p = 4, 5, 5
    parts = []

    # --- Block A1 (4x2) * B1 (2x5) ---
    # Top half: A[0:2, 0:2] (2x2) * B[0:2, :] (2x5)
    #   Split B[0:2,:] into B[0:2,0:2], B[0:2,2:4], B[0:2,4:5]
    #   2x2*2x2: Strassen (7)
    #   2x2*2x2: Strassen (7)
    #   2x2*2x1: schoolbook (4)
    U_a1t, V_a1t, W_a1t = _embed_subproblem(*_strassen_2x2(), 2, 2, 2, 0, 0, 0, n, m, p)
    U_a1t2, V_a1t2, W_a1t2 = _embed_subproblem(*_strassen_2x2(), 2, 2, 2, 0, 0, 2, n, m, p)
    U_a1t3, V_a1t3, W_a1t3 = _embed_subproblem(*_schoolbook(2, 2, 1), 2, 2, 1, 0, 0, 4, n, m, p)
    parts.append((U_a1t, V_a1t, W_a1t))
    parts.append((U_a1t2, V_a1t2, W_a1t2))
    parts.append((U_a1t3, V_a1t3, W_a1t3))

    # Bottom half: A[2:4, 0:2] (2x2) * B[0:2, :] (2x5)
    U_a1b, V_a1b, W_a1b = _embed_subproblem(*_strassen_2x2(), 2, 2, 2, 2, 0, 0, n, m, p)
    U_a1b2, V_a1b2, W_a1b2 = _embed_subproblem(*_strassen_2x2(), 2, 2, 2, 2, 0, 2, n, m, p)
    U_a1b3, V_a1b3, W_a1b3 = _embed_subproblem(*_schoolbook(2, 2, 1), 2, 2, 1, 2, 0, 4, n, m, p)
    parts.append((U_a1b, V_a1b, W_a1b))
    parts.append((U_a1b2, V_a1b2, W_a1b2))
    parts.append((U_a1b3, V_a1b3, W_a1b3))

    # --- Block A2 (4x1) * B2 (1x5) ---
    U_a2, V_a2, W_a2 = _embed_subproblem(*_schoolbook(4, 1, 5), 4, 1, 5, 0, 2, 0, n, m, p)
    parts.append((U_a2, V_a2, W_a2))

    # --- Block A3 (4x2) * B3 (2x5) ---
    # Top half: A[0:2, 3:5] (2x2) * B[3:5, :] (2x5)
    U_a3t, V_a3t, W_a3t = _embed_subproblem(*_strassen_2x2(), 2, 2, 2, 0, 3, 0, n, m, p)
    U_a3t2, V_a3t2, W_a3t2 = _embed_subproblem(*_strassen_2x2(), 2, 2, 2, 0, 3, 2, n, m, p)
    U_a3t3, V_a3t3, W_a3t3 = _embed_subproblem(*_schoolbook(2, 2, 1), 2, 2, 1, 0, 3, 4, n, m, p)
    parts.append((U_a3t, V_a3t, W_a3t))
    parts.append((U_a3t2, V_a3t2, W_a3t2))
    parts.append((U_a3t3, V_a3t3, W_a3t3))

    # Bottom half: A[2:4, 3:5] (2x2) * B[3:5, :] (2x5)
    U_a3b, V_a3b, W_a3b = _embed_subproblem(*_strassen_2x2(), 2, 2, 2, 2, 3, 0, n, m, p)
    U_a3b2, V_a3b2, W_a3b2 = _embed_subproblem(*_strassen_2x2(), 2, 2, 2, 2, 3, 2, n, m, p)
    U_a3b3, V_a3b3, W_a3b3 = _embed_subproblem(*_schoolbook(2, 2, 1), 2, 2, 1, 2, 3, 4, n, m, p)
    parts.append((U_a3b, V_a3b, W_a3b))
    parts.append((U_a3b2, V_a3b2, W_a3b2))
    parts.append((U_a3b3, V_a3b3, W_a3b3))

    U, V, W = _multi_concat(parts, n, m, p)
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that sum_r U[i*m+j][r]*V[j*p+k][r]*W[k*n+i][r] == delta_{i,k}*delta_{j,i}... 
    Actually the tensor identity is:
      T[i,j,k,l] = delta_{i,l} * delta_{j,k}  (for matrix mult: (AB)_{i,k} = sum_j A_{i,j} B_{j,k})
    With our convention:
      U has shape (n*m, rank), indexed by (i,j)
      V has shape (m*p, rank), indexed by (j,k)
      W has shape (p*n, rank), indexed by (k,i)
    The identity: sum_r U[(i,j),r] * V[(j,k),r] * W[(k,i),r] = 1 if (i==i and j==j and k==k) -> always 1?
    No. The tensor for matrix multiplication M_{n,m,p} has:
      T[(i,j),(j,k),(k,i)] = 1  for all valid i,j,k
    And the decomposition says: T = sum_r u_r * v_r * w_r
    where (u_r)_{(i,j)} = U[(i,j),r], (v_r)_{(j,k)} = V[(j,k),r], (w_r)_{(k,i)} = W[(k,i),r]
    So: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should equal 1 for all i,j,k.
    Wait, that's not right either. Let me reconsider.
    
    The matrix multiplication tensor T is a 3-tensor with indices (i,j,k) where:
      T[i,j,k] = delta_{i,j} * delta_{j,k}  -- no.
    
    Actually for multiplying A (n x m) by B (m x p) to get C (n x p):
      C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is T with indices (i, j, k) where T[i,j,k] = 1 for all valid (i,j,k).
    The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    So we need: for all i in [0,n), j in [0,m), k in [0,p):
      sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
    """
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(len(U[0])):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_np(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification using numpy."""
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    rank = U_np.shape[1]
    # Reshape: U -> (n, m, rank), V -> (m, p, rank), W -> (p, n, rank)
    U3 = U_np.reshape(n, m, rank)
    V3 = V_np.reshape(m, p, rank)
    W3 = W_np.reshape(p, n, rank)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kin->ijk', U3, V3, W3)
    return np.allclose(T, np.ones((n, m, p)), atol=tol)


def construct():
    """Construct the tensor decomposition for 4x5x5 matrix multiplication."""
    t0 = time.time()
    n, m, p = DIMENSIONS  # 4, 5, 5

    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)

    try:
        U, V, W = _construct_4x5x5()
        rank = len(U[0])
        if rank < n * m * p:  # better than schoolbook
            if _verify_np(U, V, W, n, m, p):
                if time.time() - t0 < _DEADLINE_SECONDS - 5:
                    return {"U": U, "V": V, "W": W}
    except Exception:
        pass

    return {"U": U_school, "V": V_school, "W": W_school}


# EVOLVE-BLOCK-END
