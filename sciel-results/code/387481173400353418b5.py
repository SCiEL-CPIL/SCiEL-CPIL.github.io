"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s wall-clock

    # Build slice-method decomposition
    result = _build_slice_method(n, m, p)
    
    if result is not None:
        U, V, W, rank = result
        if _verify_fast(U, V, W, n, m, p):
            # Try to reduce rank by identifying linear dependencies
            U, V, W, rank = _try_reduce_rank(U, V, W, rank, n, m, p, deadline)
            if _verify_fast(U, V, W, n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            else:
                U, V, W, rank = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook
    rank_school = n * m * p
    U_fb = [[0] * rank_school for _ in range(n * m)]
    V_fb = [[0] * rank_school for _ in range(m * p)]
    W_fb = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    return {"U": U_fb, "V": V_fb, "W": W_fb}


def _try_reduce_rank(U, V, W, rank, n, m, p, deadline):
    """
    Try to reduce rank by checking if any rank-1 tensor term lies in the span
    of the others. If T_r = sum_{c != r} alpha_c * T_c, then we can rescale
    the remaining terms to absorb T_r, reducing rank by 1.
    Multiple passes to catch cascading dependencies.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    current_rank = rank
    removed = set()
    
    # Flatten each rank-1 tensor into a vector of size n*m*p
    def tensor_to_vec(u, v, w):
        """Flatten U[i,j]*V[j,k]*W[k,i] into a flat vector."""
        vec = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    vec[(i * m + j) * p + k] = u[i * m + j] * v[j * p + k] * w[k * n + i]
        return vec
    
    # Build all rank-1 tensor vectors
    tensors = []
    for c in range(rank):
        tensors.append(tensor_to_vec(U[:, c], V[:, c], W[:, c]))
    
    tensors = np.array(tensors)  # shape (rank, n*m*p)
    
    # Multiple passes to find dependencies
    for pass_num in range(10):
        if time.time() > deadline - 10:
            break
        found = False
        for r in range(rank):
            if r in removed:
                continue
            if time.time() > deadline - 10:
                break
            
            # Build matrix of other tensors
            others = [c for c in range(rank) if c != r and c not in removed]
            if len(others) < 1:
                continue
            
            A = tensors[others]  # shape (len(others), n*m*p)
            b = tensors[r]       # shape (n*m*p,)
            
            # Solve: find alpha such that A.T @ alpha ≈ b
            try:
                ATA = A @ A.T  # (num_others, num_others)
                Ab = A @ b     # (num_others,)
                
                alpha, residuals, rank_lsq, sv = np.linalg.lstsq(ATA, Ab, rcond=1e-10)
                
                # Compute residual
                residual = b - A.T @ alpha
                if np.linalg.norm(residual) < 1e-8:
                    # T_r is in the span! Remove it and rescale others.
                    removed.add(r)
                    for idx, c in enumerate(others):
                        scale = 1.0 + alpha[idx]
                        if abs(scale) > 1e-12:
                            U[:, c] *= scale
                        else:
                            # scale is ~0, this term becomes zero; remove it too
                            removed.add(c)
                    found = True
                    break  # Restart scan after modification
            except np.linalg.LinAlgError:
                continue
        
        if not found:
            break
    
    if removed:
        keep = [c for c in range(rank) if c not in removed]
        if len(keep) > 0:
            U = U[:, keep]
            V = V[:, keep]
            W = W[:, keep]
            current_rank = len(keep)
        else:
            # All removed, should not happen for valid decomposition
            pass
    
    return U, V, W, current_rank


def _get_strassen_2x2x2():
    """
    Return U, V, W factors for 2x2x2 matrix multiplication (Strassen, rank 7).
    U: (4, 7) indexed as U[i*2+j, c] for A[i,j]
    V: (4, 7) indexed as V[j*2+k, c] for B[j,k]
    W: (4, 7) indexed as W[k*2+i, c] for C[k,i]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # U: A factors
    U[0, 0] = 1   # a in m0
    U[3, 0] = 1   # d in m0
    U[2, 1] = 1   # c in m1
    U[3, 1] = 1   # d in m1
    U[0, 2] = 1   # a in m2
    U[3, 3] = 1   # d in m3
    U[0, 4] = 1   # a in m4
    U[1, 4] = 1   # b in m4
    U[2, 5] = 1   # c in m5
    U[0, 5] = -1  # -a in m5
    U[1, 6] = 1   # b in m6
    U[3, 6] = -1  # -d in m6
    
    # V: B factors
    V[0, 0] = 1   # e in m0
    V[3, 0] = 1   # h in m0
    V[0, 1] = 1   # e in m1
    V[1, 2] = 1   # f in m2
    V[3, 2] = -1  # -h in m2
    V[2, 3] = 1   # g in m3
    V[0, 3] = -1  # -e in m3
    V[3, 4] = 1   # h in m4
    V[0, 5] = 1   # e in m5
    V[1, 5] = 1   # f in m5
    V[2, 6] = 1   # g in m6
    V[3, 6] = 1   # h in m6
    
    # W: C factors (W[k*2+i, c])
    W[0, 0] = 1   # m0 in C[0,0]
    W[0, 3] = 1   # m3 in C[0,0]
    W[0, 4] = -1  # -m4 in C[0,0]
    W[0, 6] = 1   # m6 in C[0,0]
    W[2, 2] = 1   # m2 in C[0,1]
    W[2, 4] = 1   # m4 in C[0,1]
    W[1, 1] = 1   # m1 in C[1,0]
    W[1, 3] = 1   # m3 in C[1,0]
    W[3, 0] = 1   # m0 in C[1,1]
    W[3, 1] = -1  # -m1 in C[1,1]
    W[3, 2] = 1   # m2 in C[1,1]
    W[3, 5] = 1   # m5 in C[1,1]
    
    return U, V, W


def _build_strassen_3x5x6(n, m, p):
    """
    Build tensor decomposition for 3x5 @ 5x6 using Strassen block structure.
    
    Per k-block:
      Top 2 rows (i=0,1):
        j in {0,1}: Strassen 2x2x2 (rank 7)
        j in {2,3}: Strassen 2x2x2 (rank 7)
        j = 4: schoolbook 2x1x2 (rank 4)
      Bottom row (i=2):
        j in {0,1}: schoolbook 1x2x2 (rank 4)
        j in {2,3}: schoolbook 1x2x2 (rank 4)
        j = 4: schoolbook 1x1x2 (rank 2)
    Total per k-block: 7+7+4+4+4+2 = 28
    3 k-blocks: 84
    """
    sU, sV, sW = _get_strassen_2x2x2()
    
    rank = 84
    U = np.zeros((n * m, rank))  # 15 x 84
    V = np.zeros((m * p, rank))  # 30 x 84
    W = np.zeros((p * n, rank))  # 18 x 84
    
    col = 0
    
    for k_block in range(3):
        k0 = k_block * 2  # start k index
        k1 = k0 + 1       # end k index
        
        # Top 2 rows (i=0,1), j in {0,1}: Strassen
        for c in range(7):
            U[0*5 + 0, col] = sU[0, c]   # i=0, j=0
            U[0*5 + 1, col] = sU[1, c]   # i=0, j=1
            U[1*5 + 0, col] = sU[2, c]   # i=1, j=0
            U[1*5 + 1, col] = sU[3, c]   # i=1, j=1
            V[0*6 + k0, col] = sV[0, c]   # j=0, k=k0
            V[0*6 + k1, col] = sV[1, c]   # j=0, k=k1
            V[1*6 + k0, col] = sV[2, c]   # j=1, k=k0
            V[1*6 + k1, col] = sV[3, c]   # j=1, k=k1
            W[k0*3 + 0, col] = sW[0, c]   # k=k0, i=0
            W[k0*3 + 1, col] = sW[1, c]   # k=k0, i=1
            W[k1*3 + 0, col] = sW[2, c]   # k=k1, i=0
            W[k1*3 + 1, col] = sW[3, c]   # k=k1, i=1
            col += 1
        
        # Top 2 rows (i=0,1), j in {2,3}: Strassen
        for c in range(7):
            U[0*5 + 2, col] = sU[0, c]
            U[0*5 + 3, col] = sU[1, c]
            U[1*5 + 2, col] = sU[2, c]
            U[1*5 + 3, col] = sU[3, c]
            V[2*6 + k0, col] = sV[0, c]
            V[2*6 + k1, col] = sV[1, c]
            V[3*6 + k0, col] = sV[2, c]
            V[3*6 + k1, col] = sV[3, c]
            W[k0*3 + 0, col] = sW[0, c]
            W[k0*3 + 1, col] = sW[1, c]
            W[k1*3 + 0, col] = sW[2, c]
            W[k1*3 + 1, col] = sW[3, c]
            col += 1
        
        # Top 2 rows (i=0,1), j=4: schoolbook 2x1x2
        for i in range(2):
            for k in range(2):
                U[i*5 + 4, col] = 1
                V[4*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + i, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {0,1}: schoolbook 1x2x2
        for j in range(2):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        
        # Bottom row (i=2), j in {2,3}: schoolbook 1x2x2
        for j in range(2, 4):
            for k in range(2):
                U[2*5 + j, col] = 1
                V[j*6 + k0 + k, col] = 1
                W[(k0 + k)*3 + 2, col] = 1
                col += 1
        
        # Bottom row (i=2), j=4: schoolbook 1x1x2
        for k in range(2):
            U[2*5 + 4, col] = 1
            V[4*6 + k0 + k, col] = 1
            W[(k0 + k)*3 + 2, col] = 1
            col += 1
    
    if col != rank:
        if col > rank:
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
        else:
            rank = col
            U = U[:, :rank]
            V = V[:, :rank]
            W = W[:, :rank]
    
    return U, V, W, rank


def _build_slice_method(n, m, p):
    """
    Slice method: decompose 3x5 @ 5x6 by treating each intermediate index j
    as a separate slice. For each j, we compute A[:,j] * B[j,:] which is a
    3x1 @ 1x6 = 3x6 outer product (rank 18 schoolbook, but we can use
    Strassen on 2x2 sub-blocks of the 3x6 result).
    
    Actually, the slice for index j contributes:
      C[:, k] += A[:, j] * B[j, k] for all k
    
    So for each j, the contribution is the outer product of A[:,j] (3x1)
    and B[j,:] (1x6), which has rank 1 in the tensor sense (it's a rank-1
    tensor T[i,j,k] = U[i,j]*V[j,k]*W[k,i]).
    
    Wait - the slice method works differently. Let me reconsider.
    
    The tensor T[i,j,k] = 1 for all i,j,k (this is the matrix multiplication tensor).
    
    Slice method: fix j. The slice T[:,j,:] is a 3x6 matrix of all ones.
    We need to express this as a sum of rank-1 terms in the i-k plane.
    A 3x6 all-ones matrix has rank 1, so T[:,j,:] = u_j * v_j^T where
    u_j = [1,1,1]^T and v_j = [1,1,1,1,1,1]^T.
    
    But in the tensor decomposition, each rank-1 term is:
      T[i,j,k] += U[i,j] * V[j,k] * W[k,i]
    
    For a fixed j, the contribution to T[:,j,:] from term c is:
      T[:,j,:] += U[:,j] * V[j,:] * W[:,i] (element-wise with W indexed by k,i)
    
    This is more complex. Let me use the standard approach:
    
    For each j (0..4), we need the 3x6 slice to be all ones.
    We decompose each 3x6 all-ones matrix using a combination of:
    - Strassen 2x2x2 for the top-left 2x2, top-right 2x2, bottom-left 1x2, bottom-right 1x2
    - Actually for 3x6, we can split k into two groups of 3, and use Strassen-like
      decomposition for 3x3 (but 3x3x3 has rank 23, not helpful here since
      we only need rank-1 slices).
    
    Actually, the simplest approach: each slice j contributes exactly one rank-1
    term to the tensor: U[i,j]=1 for all i, V[j,k]=1 for all k, W[k,i]=1 for all k,i.
    But W[k,i] must be the same across all j terms! So we can't just have 5 terms
    with different W factors.
    
    The correct slice method:
    - For each j, the slice T[:,j,:] = ones(3,6) must be represented.
    - A rank-1 term c contributes U[:,j,c] * V[j,:,c] * W[:,i,c] to T[:,j,:].
    - We need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,k.
    
    This is equivalent to: for each j, the 3x6 matrix M_j[i,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1.
    
    The simplest decomposition: use p=6 rank-1 terms per j, giving 5*6=30 total.
    But we can do better by sharing W factors.
    
    Better: split k into two groups of 3. For each j, use a 3x3x3-like structure.
    Actually, let's just use the direct approach with sharing:
    
    For k in 0..5, i in 0..2: W[k,i,c] needs to be structured.
    
    Let me use the approach where for each j, we have 6 terms (one per k),
    with U[i,j,c] = 1, V[j,k,c] = 1 (only for matching k), W[k,i,c] = 1.
    Total: 5*6 = 30 terms. Then try to merge.
    
    Alternatively, use Strassen on 2x2 blocks within each slice:
    For each j, split the 3x6 into:
      - Top 2 rows, k in {0,1}: 2x2 all-ones -> Strassen rank 7? No, that's for
        multiplication, not for representing a fixed matrix.
    
    For representing a fixed 3x6 all-ones matrix as sum of rank-1 tensors:
    Each term c contributes a 3x6 matrix: outer(U[:,j,c], V[j,:,c]) * W[:,i,c]
    where W[:,i,c] is a 6x3 matrix (k x i).
    
    This is getting complex. Let me just use the straightforward 30-term construction
    and then aggressively reduce.
    """
    # Start with schoolbook-like: 5 j's * 6 k's = 30 terms
    # Each term c corresponds to (j, k) pair
    rank = m * p  # 30
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for j in range(m):
        for k in range(p):
            U[:, j] = 0  # ensure clean
            for i in range(n):
                U[i * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            for i in range(n):
                W[k * n + i, col] = 1.0
            col += 1
    
    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p):
    """Fast verification using einsum."""
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)


def _verify(U, V, W, n, m, p):
    """
    Verify that the tensor decomposition is exact.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-END
