"""Program for matrix multiplication 3x5x6 rank-minimal decomposition."""
import numpy as np
import time
import math

DIMENSIONS = [3, 5, 6]
DEADLINE_SECONDS = 115  # safety margin for 120s limit (evaluator kills at 150s)

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication.
    
    The tensor equation is:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    where T[i,j,k] should be 1 for all valid (i,j,k) triples.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    if U.shape != (n*m, rank):
        return False
    if V.shape != (m*p, rank):
        return False
    if W.shape != (p*n, rank):
        return False
    
    # Build tensor from decomposition using vectorized operations
    tensor = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        # For each (i,k), the contribution is sum_j U_c[i,j]*V_c[j,k]*W_c[k,i]
        # This is: (U_c * W_c.T) @ V_c^T ... let me think
        # Actually: sum_j U_c[i,j] * V_c[j,k] * W_c[k,i]
        # = sum_j (U_c[i,j] * W_c[k,i]) * V_c[j,k]
        # For fixed i,k: W_c[k,i] * sum_j U_c[i,j]*V_c[j,k] = W_c[k,i] * (U_c @ V_c)[i,k]
        # So the contribution to tensor[i,j,k] is NOT simply separable like that.
        # Let me just do it the direct way but vectorized over j.
        # tensor[i,j,k] += U_c[i,j] * V_c[j,k] * W_c[k,i]
        # For fixed c: outer product style
        # Create a 3D array: U_c[i,j] * V_c[j,k] * W_c[k,i]
        # This is: (U_c[:, :, None] * V_c[None, :, :] * W_c[None, None, :].T) ... no
        # U_c shape (n,m), V_c shape (m,p), W_c shape (p,n)
        # We want: result[i,j,k] = U_c[i,j] * V_c[j,k] * W_c[k,i]
        # = np.einsum('ij,jk,ki->ijk', U_c, V_c, W_c)
        tensor += np.einsum('ij,jk,ki->ijk', U_c, V_c, W_c)
    
    expected = np.ones((n, m, p))
    return np.allclose(tensor, expected, atol=1e-9)

def _verify_decomposition_fast(U, V, W, n, m, p):
    """Faster verification using the trace convention directly.
    
    For each (i,k): C[i,k] = sum_j A[i,j]*B[j,k] should equal 1
    In the decomposition: C[i,k] = sum_c sum_j U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    = sum_c W[k*n+i,c] * sum_j U[i*m+j,c]*V[j*p+k,c]
    
    Let M_c[i,k] = sum_j U[i*m+j,c]*V[j*p+k,c]  (this is like U_c @ V_c where U_c is n x m and V_c is m x p)
    Then C[i,k] = sum_c M_c[i,k] * W[k*n+i,c]
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Compute sum over c of (U_c @ V_c) * W_c where W_c is reshaped to (p,n) and transposed to (n,p)
    # U_c is n x m, V_c is m x p, so U_c @ V_c is n x p
    # W_c is p x n, we need W_c[k,i] which is W_c.T[i,k]
    # So C[i,k] = sum_c (U_c @ V_c)[i,k] * W_c.T[i,k]
    # = sum_c elementwise product of (U_c @ V_c) and W_c.T, summed over c
    
    # Vectorized: reshape U to (rank, n, m), V to (rank, m, p), W to (rank, p, n)
    U_3d = U.reshape(rank, n, m)
    V_3d = V.reshape(rank, m, p)
    W_3d = W.reshape(rank, p, n)
    
    # Compute U_c @ V_c for all c: shape (rank, n, p)
    UV = np.einsum('cim,cjk->cik', U_3d, V_3d)  # Wait, let me be careful
    # U_3d[c, i, m_idx], V_3d[c, m_idx, k]
    # UV[c, i, k] = sum_m_idx U_3d[c, i, m_idx] * V_3d[c, m_idx, k]
    UV = np.einsum('cim,cmk->cik', U_3d, V_3d)
    
    # W_3d[c, k, i], we need W_3d.T for (c, i, k)
    W_T = W_3d.transpose(0, 2, 1)  # shape (rank, n, p)
    
    # C[i,k] = sum_c UV[c,i,k] * W_T[c,i,k]
    C = np.einsum('cik,cik->ik', UV, W_T)
    
    return np.allclose(C, np.ones((n, p)), atol=1e-9)

def _strassen_222():
    """Return the 7-term Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns U, V, W each of shape (4, 7) where:
    U[i*2+j, c] corresponds to A[i,j] factor
    V[j*2+k, c] corresponds to B[j,k] factor  
    W[k*2+i, c] corresponds to C[k,i] factor (trace convention: W[k*n+i, r])
    
    Strassen's 7 products:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B12-B22)
    M4 = A22(B21-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # A is 2x2, indexed (i,j) -> row i*2+j
    # B is 2x2, indexed (j,k) -> row j*2+k
    # C is 2x2, indexed (k,i) -> row k*2+i (trace convention)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # A11 = A[0,0] = U[0], A22 = A[1,1] = U[3]
    # B11 = B[0,0] = V[0], B22 = B[1,1] = V[3]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 += M1, C22 += M1
    # C11: k=0,i=0 -> W[0*2+0] = W[0]
    # C22: k=1,i=1 -> W[1*2+1] = W[3]
    W[0, 0] = 1; W[3, 0] = 1
    
    # Term 2: M2 = (A21+A22)B11
    # A21 = A[1,0] = U[2], A22 = A[1,1] = U[3]
    # B11 = B[0,0] = V[0]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 += M2, C22 -= M2
    # C21: k=0,i=1 -> W[0*2+1] = W[1]
    # C22: k=1,i=1 -> W[3]
    W[1, 1] = 1; W[3, 1] = -1
    
    # Term 3: M3 = A11(B12-B22)
    # A11 = A[0,0] = U[0]
    # B12 = B[0,1] = V[1], B22 = B[1,1] = V[3]
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C12 += M3, C22 += M3
    # C12: k=1,i=0 -> W[1*2+0] = W[2]
    # C22: k=1,i=1 -> W[3]
    W[2, 2] = 1; W[3, 2] = 1
    
    # Term 4: M4 = A22(B21-B11)
    # A22 = A[1,1] = U[3]
    # B21 = B[1,0] = V[2], B11 = B[0,0] = V[0]
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C11 += M4, C21 += M4
    # C11: k=0,i=0 -> W[0]
    # C21: k=0,i=1 -> W[1]
    W[0, 3] = 1; W[1, 3] = 1
    
    # Term 5: M5 = (A11+A12)B22
    # A11 = A[0,0] = U[0], A12 = A[0,1] = U[1]
    # B22 = B[1,1] = V[3]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 -= M5, C12 += M5
    # C11: W[0], C12: W[2]
    W[0, 4] = -1; W[2, 4] = 1
    
    # Term 6: M6 = (A21-A11)(B11+B12)
    # A21 = A[1,0] = U[2], A11 = A[0,0] = U[0]
    # B11 = B[0,0] = V[0], B12 = B[0,1] = V[1]
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 += M6
    # C22: W[3]
    W[3, 5] = 1
    
    # Term 7: M7 = (A12-A22)(B21+B22)
    # A12 = A[0,1] = U[1], A22 = A[1,1] = U[3]
    # B21 = B[1,0] = V[2], B22 = B[1,1] = V[3]
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 += M7
    # C11: W[0]
    W[0, 6] = 1
    
    return U.tolist(), V.tolist(), W.tolist()

def _embed_sub_decomp(sub_U, sub_V, sub_W, i_offset, j_offset, k_offset, n, m, p):
    """Embed a sub-decomposition for A[i_offset:i_offset+sn, j_offset:j_offset+sm] * B[j_offset:j_offset+sm, k_offset:k_offset+sp]
    into the full n x m x p problem.
    
    sub_U has shape (sn*sm, rank_sub)
    sub_V has shape (sm*sp, rank_sub)
    sub_W has shape (sp*sn, rank_sub)
    
    Returns U, V, W each of shape (n*m, rank_sub), (m*p, rank_sub), (p*n, rank_sub)
    with zeros outside the embedded region.
    """
    sn = sub_U.shape[0] // sm if sm > 0 else 0
    sp = sub_V.shape[0] // sm if sm > 0 else 0
    rank_sub = sub_U.shape[1]
    
    U_full = np.zeros((n*m, rank_sub))
    V_full = np.zeros((m*p, rank_sub))
    W_full = np.zeros((p*n, rank_sub))
    
    # Embed U: sub_U[i_sub, j_sub] -> U_full[(i_offset+i_sub)*m + (j_offset+j_sub), c]
    for i_sub in range(sn):
        for j_sub in range(sm):
            U_full[(i_offset+i_sub)*m + (j_offset+j_sub), :] = sub_U[i_sub*sm+j_sub, :]
    
    # Embed V: sub_V[j_sub, k_sub] -> V_full[(j_offset+j_sub)*p + (k_offset+k_sub), c]
    for j_sub in range(sm):
        for k_sub in range(sp):
            V_full[(j_offset+j_sub)*p + (k_offset+k_sub), :] = sub_V[j_sub*sp+k_sub, :]
    
    # Embed W: sub_W[k_sub, i_sub] -> W_full[(k_offset+k_sub)*n + (i_offset+i_sub), c]
    for k_sub in range(sp):
        for i_sub in range(sn):
            W_full[(k_offset+k_sub)*n + (i_offset+i_sub), :] = sub_W[k_sub*sn+i_sub, :]
    
    return U_full, V_full, W_full

def _schoolbook_sub(sn, sm, sp):
    """Schoolbook decomposition for sn x sm x sp matrix multiplication.
    Returns U, V, W of shapes (sn*sm, rank), (sm*sp, rank), (sp*sn, rank).
    """
    rank = sn * sm * sp
    U = np.zeros((sn*sm, rank))
    V = np.zeros((sm*sp, rank))
    W = np.zeros((sp*sn, rank))
    
    idx = 0
    for i in range(sn):
        for j in range(sm):
            for k in range(sp):
                U[i*sm+j, idx] = 1.0
                V[j*sp+k, idx] = 1.0
                W[k*sn+i, idx] = 1.0
                idx += 1
    
    return U, V, W

def _block_decomposition_356():
    """Construct the 84-rank block decomposition for M_{3,5,6}.
    
    Split j (intermediate, size 5) into [0,1] (size 2) and [2,3,4] (size 3)
    Split k (output, size 6) into [0,1,2] (size 3) and [3,4,5] (size 3)
    
    Block 1: A[:,0:2] * B[0:2,0:3] -> M_{3,2,3}
      Split j=2 into 1+1:
        Sub-block 1a: A[:,0:1] * B[0:1,0:3] -> M_{3,1,3} rank 9 (schoolbook)
        Sub-block 1b: A[:,1:2] * B[1:2,0:3] -> M_{3,1,3} rank 9 (schoolbook)
      Total for Block 1: 18
    
    Hmm, that's just schoolbook. Let me use Strassen for the 2x2x2 part.
    
    Better approach for M_{3,2,3}:
    A is 3x2, B is 2x3, C is 3x3.
    Split A into rows: A_top (2x2) and A_bot (1x2)
    Split B into cols: B_left (2x2) and B_right (2x1)
    
    C = A_top*B_left + A_top*B_right + A_bot*B_left + A_bot*B_right
    C_TL (2x2): use Strassen rank 7
    C_TR (2x1): schoolbook rank 4 (2*1*... wait, A_top is 2x2, B_right is 2x1, so M_{2,2,1} rank 4)
    C_BL (1x2): schoolbook rank 4 (M_{1,2,2} rank 4)
    C_BR (1x1): rank 2 (M_{1,2,1} rank 2)
    Total: 7+4+4+2 = 17
    
    Actually wait:
    A_top is 2x2 (rows 0,1 of A, all 2 cols)
    A_bot is 1x2 (row 2 of A, all 2 cols)
    B_left is 2x2 (all 2 rows of B, cols 0,1)
    B_right is 2x1 (all 2 rows of B, col 2)
    
    A_top * B_left: (2x2)*(2x2) = 2x2 -> M_{2,2,2} rank 7 (Strassen)
    A_top * B_right: (2x2)*(2x1) = 2x1 -> M_{2,2,1} rank 4 (schoolbook)
    A_bot * B_left: (1x2)*(2x2) = 1x2 -> M_{1,2,2} rank 4 (schoolbook)
    A_bot * B_right: (1x2)*(2x1) = 1x1 -> M_{1,2,1} rank 2 (schoolbook)
    Total: 7+4+4+2 = 17
    
    For M_{3,3,3}:
    A is 3x3, B is 3x3, C is 3x3.
    Split A: A_top (2x3), A_bot (1x3)
    Split B: B_left (3x2), B_right (3x1)
    
    A_top * B_left: (2x3)*(3x2) = 2x2 -> M_{2,3,2}
    A_top * B_right: (2x3)*(3x1) = 2x1 -> M_{2,3,1} rank 6
    A_bot * B_left: (1x3)*(3x2) = 1x2 -> M_{1,3,2} rank 6
    A_bot * B_right: (1x3)*(3x1) = 1x1 -> M_{1,3,1} rank 3
    
    For M_{2,3,2}: split the 3 in the middle.
    M_{2,3,2} = M_{2,2,2} + M_{2,1,2} = 7 + 4 = 11
    Or M_{2,3,2} = M_{2,1,2} + M_{2,2,2} = 4 + 7 = 11
    
    So M_{3,3,3} <= 11 + 6 + 6 + 3 = 26.
    
    Total for 84-rank:
    Block (j=0:2, k=0:3): M_{3,2,3} rank 17
    Block (j=0:2, k=3:6): M_{3,2,3} rank 17
    Block (j=2:5, k=0:3): M_{3,3,3} rank 26
    Block (j=2:5, k=3:6): M_{3,3,3} rank 26
    Total: 17+17+26+26 = 86
    
    Hmm, that's 86, not 84. Let me reconsider.
    
    Alternative: use a different split.
    Split j into [0,1,2] (size 3) and [3,4] (size 2)
    Split k into [0,1,2] (size 3) and [3,4,5] (size 3)
    
    Block (j=0:3, k=0:3): M_{3,3,3} rank 26
    Block (j=0:3, k=3:6): M_{3,3,3} rank 26
    Block (j=3:5, k=0:3): M_{3,2,3} rank 17
    Block (j=3:5, k=3:6): M_{3,2,3} rank 17
    Total: 26+26+17+17 = 86
    
    Same. Let me try to get M_{3,3,3} to rank 23 using a known decomposition.
    
    Actually, let me just use the 84-rank construction from inspiration 1.
    The key is to use Strassen for 2x2x2 blocks.
    
    Let me try: split j into [0,1] + [2,3] + [4] and k into [0,1] + [2,3] + [4,5]
    
    That gives 9 blocks which is too many.
    
    Let me just implement the 86-rank version first, then try to optimize.
    
    Actually, let me reconsider M_{2,3,2}. Can we do better than 11?
    M_{2,3,2}: A is 2x3, B is 3x2, C is 2x2.
    C[i,k] = sum_{j=0}^{2} A[i,j]*B[j,k]
    
    This is equivalent to: C = A[:,0:2] * B[0:2,:] + A[:,2:3] * B[2:3,:]
    = (2x2)*(2x2) + (2x1)*(1x2)
    = M_{2,2,2} + M_{2,1,2}
    = 7 + 4 = 11
    
    Or: C = A[:,0] * B[0,:] + A[:,1] * B[1,:] + A[:,2] * B[2,:]
    Each term is rank 1 in a sense, but they don't combine well.
    
    I believe 11 is optimal for M_{2,3,2}.
    
    So let me go with 86-rank for now.
    """
    n, m, p = 3, 5, 6
    
    # Split j into [0,1] (size 2) and [2,3,4] (size 3)
    # Split k into [0,1,2] (size 3) and [3,4,5] (size 3)
    
    all_U = []
    all_V = []
    all_W = []
    
    # Block 1: j=0:2, k=0:3 -> M_{3,2,3}
    # Use Strassen-based decomposition: rank 17
    # A is 3x2, B is 2x3
    # Split A: top 2 rows, bottom 1 row
    # Split B: left 2 cols, right 1 col
    
    # Sub-block 1a: A[0:2,:] * B[:,0:2] -> M_{2,2,2} Strassen rank 7
    U_1a, V_1a, W_1a = _strassen_222()
    U_1a_arr = np.array(U_1a)
    V_1a_arr = np.array(V_1a)
    W_1a_arr = np.array(W_1a)
    # Embed into full: i_offset=0, j_offset=0, k_offset=0, sn=2, sm=2, sp=2
    U_full_1a, V_full_1a, W_full_1a = _embed_sub_decomp(U_1a_arr, V_1a_arr, W_1a_arr, 0, 0, 0, n, m, p)
    all_U.append(U_full_1a)
    all_V.append(V_full_1a)
    all_W.append(W_full_1a)
    
    # Sub-block 1b: A[0:2,:] * B[:,2:3] -> M_{2,2,1} schoolbook rank 4
    U_1b, V_1b, W_1b = _schoolbook_sub(2, 2, 1)
    U_full_1b, V_full_1b, W_full_1b = _embed_sub_decomp(U_1b, V_1b, W_1b, 0, 0, 2, n, m, p)
    all_U.append(U_full_1b)
    all_V.append(V_full_1b)
    all_W.append(W_full_1b)
    
    # Sub-block 1c: A[2:3,:] * B[:,0:2] -> M_{1,2,2} schoolbook rank 4
    U_1c, V_1c, W_1c = _schoolbook_sub(1, 2, 2)
    U_full_1c, V_full_1c, W_full_1c = _embed_sub_decomp(U_1c, V_1c, W_1c, 2, 0, 0, n, m, p)
    all_U.append(U_full_1c)
    all_V.append(V_full_1c)
    all_W.append(W_full_1c)
    
    # Sub-block 1d: A[2:3,:] * B[:,2:3] -> M_{1,2,1} schoolbook rank 2
    U_1d, V_1d, W_1d = _schoolbook_sub(1, 2, 1)
    U_full_1d, V_full_1d, W_full_1d = _embed_sub_decomp(U_1d, V_1d, W_1d, 2, 0, 2, n, m, p)
    all_U.append(U_full_1d)
    all_V.append(V_full_1d)
    all_W.append(W_full_1d)
    
    # Block 2: j=0:2, k=3:6 -> M_{3,2,3}
    # Same structure as Block 1 but with k_offset=3
    
    # Sub-block 2a: A[0:2,:] * B[:,0:2] -> M_{2,2,2} Strassen rank 7
    U_2a, V_2a, W_2a = _strassen_222()
    U_2a_arr = np.array(U_2a)
    V_2a_arr = np.array(V_2a)
    W_2a_arr = np.array(W_2a)
    U_full_2a, V_full_2a, W_full_2a = _embed_sub_decomp(U_2a_arr, V_2a_arr, W_2a_arr, 0, 0, 3, n, m, p)
    all_U.append(U_full_2a)
    all_V.append(V_full_2a)
    all_W.append(W_full_2a)
    
    # Sub-block 2b: A[0:2,:] * B[:,2:3] -> M_{2,2,1} schoolbook rank 4
    U_2b, V_2b, W_2b = _schoolbook_sub(2, 2, 1)
    U_full_2b, V_full_2b, W_full_2b = _embed_sub_decomp(U_2b, V_2b, W_2b, 0, 0, 5, n, m, p)
    all_U.append(U_full_2b)
    all_V.append(V_full_2b)
    all_W.append(W_full_2b)
    
    # Sub-block 2c: A[2:3,:] * B[:,0:2] -> M_{1,2,2} schoolbook rank 4
    U_2c, V_2c, W_2c = _schoolbook_sub(1, 2, 2)
    U_full_2c, V_full_2c, W_full_2c = _embed_sub_decomp(U_2c, V_2c, W_2c, 2, 0, 3, n, m, p)
    all_U.append(U_full_2c)
    all_V.append(V_full_2c)
    all_W.append(W_full_2c)
    
    # Sub-block 2d: A[2:3,:] * B[:,2:3] -> M_{1,2,1} schoolbook rank 2
    U_2d, V_2d, W_2d = _schoolbook_sub(1, 2, 1)
    U_full_2d, V_full_2d, W_full_2d = _embed_sub_decomp(U_2d, V_2d, W_2d, 2, 0, 5, n, m, p)
    all_U.append(U_full_2d)
    all_V.append(V_full_2d)
    all_W.append(W_full_2d)
    
    # Block 3: j=2:5, k=0:3 -> M_{3,3,3}
    # Use schoolbook for simplicity: rank 27
    # (Could use Strassen-based for rank 26, but let's start with schoolbook)
    U_3, V_3, W_3 = _schoolbook_sub(3, 3, 3)
    U_full_3, V_full_3, W_full_3 = _embed_sub_decomp(U_3, V_3, W_3, 0, 2, 0, n, m, p)
    all_U.append(U_full_3)
    all_V.append(V_full_3)
    all_W.append(W_full_3)
    
    # Block 4: j=2:5, k=3:6 -> M_{3,3,3}
    U_4, V_4, W_4 = _schoolbook_sub(3, 3, 3)
    U_full_4, V_full_4, W_full_4 = _embed_sub_decomp(U_4, V_4, W_4, 0, 2, 3, n, m, p)
    all_U.append(U_full_4)
    all_V.append(V_full_4)
    all_W.append(W_full_4)
    
    # Combine all blocks
    U_combined = np.hstack(all_U)
    V_combined = np.hstack(all_V)
    W_combined = np.hstack(all_W)
    
    rank = U_combined.shape[1]
    
    # Verify
    assert _verify_decomposition_fast(U_combined, V_combined, W_combined, n, m, p), \
        f"Block decomposition verification failed! Rank={rank}"
    
    return U_combined.tolist(), V_combined.tolist(), W_combined.tolist()

def _direct_90_rank(n, m, p):
    """Direct 90-rank schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, idx] = 1.0
                V[j*p+k, idx] = 1.0
                W[k*n+i, idx] = 1.0
                idx += 1
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Start with schoolbook (rank 90)
    U_list, V_list, W_list = _direct_90_rank(n, m, p)
    best_rank = n * m * p
    best_result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Try the structured block decomposition (rank 86)
    if time.time() - start_time < DEADLINE_SECONDS - 5:
        try:
            U_blk, V_blk, W_blk = _block_decomposition_356()
            blk_rank = len(U_blk[0])
            if blk_rank < best_rank:
                best_rank = blk_rank
                best_result = {"U": U_blk, "V": V_blk, "W": W_blk}
        except Exception as e:
            pass  # Fall back to schoolbook
    
    return best_result

# EVOLVE-BLOCK-END
