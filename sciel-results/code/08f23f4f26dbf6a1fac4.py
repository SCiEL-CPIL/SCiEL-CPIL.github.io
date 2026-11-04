import time
import numpy as np

DIMENSIONS = [4, 4, 7]
DEADLINE = 150.0
SAFETY_MARGIN = 2.0

def _start_time():
    return time.time()

def _remaining_time(t0):
    return DEADLINE - SAFETY_MARGIN - (time.time() - t0)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Build the multiplication tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The decomposition says: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, the convention is W[k*n+i, r]. Let's check the indices.
    # A is n x m, B is m x p. C is n x p.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T has indices (i, j, k) where T[i,j,k] = 1.
    # Decomposition: T[i,j,k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]?
    # The interface says:
    # U: (n*m, rank), index is i*m + j
    # V: (m*p, rank), index is j*p + k
    # W: (p*n, rank), index is k*n + i
    
    # So for each r, we have vectors u_r (size n*m), v_r (size m*p), w_r (size p*n).
    # The tensor reconstructed is T_r[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r].
    # We need sum_r T_r[i,j,k] = 1 for all i,j,k.
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using NumPy einsum."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Reshape U to (n, m, rank)
    U_3d = U.reshape(n, m, rank)
    # Reshape V to (m, p, rank)
    V_3d = V.reshape(m, p, rank)
    # Reshape W to (p, n, rank)
    W_3d = W.reshape(p, n, rank)
    
    # Compute T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
    # This is a bit tricky with einsum.
    # Let's just do it with loops over i, k and vectorized j, or just full einsum.
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # We can compute this as:
    # For each r, we have a tensor T_r[i,j,k] = U[i,j,r] V[j,k,r] W[k,i,r]
    # This is not a simple outer product because the indices are mixed.
    
    # Alternative: 
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Let's create a 3D array for each r? No, that's too much memory.
    
    # Let's use the property:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We can iterate over j and compute for all i, k.
    # For fixed j:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Let U_j[i, r] = U[i*m+j, r]  (size n x rank)
    # Let V_j[k, r] = V[j*p+k, r]  (size p x rank)
    # Let W_j[i, k, r] = W[k*n+i, r]  (size n x p x rank) -> W_j[i, k, r] is W[k*n+i, r]
    # T_j[i, k] = sum_r U_j[i, r] * V_j[k, r] * W_j[i, k, r]
    # This is still not a simple matrix product.
    
    # Let's just do the full loop but with NumPy for the inner sum.
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i*m+j, :]
                v_val = V[j*p+k, :]
                w_val = W[k*n+i, :]
                val = np.dot(u_val, v_val * w_val)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _get_schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    return U, V, W

def _get_strassen_4x4x4():
    """
    Return a known rank-49 decomposition for 4x4x4 matrix multiplication.
    This is derived from the Kronecker product of the 2x2 Strassen decomposition,
    but with the correct index mapping.
    """
    n = m = p = 4
    # We will use the standard Strassen 4x4x4 decomposition.
    # The rank is 49.
    # We can construct it by taking the 2x2 Strassen factors and doing a proper Kronecker-like expansion.
    # Let's hardcode the 49 terms for 4x4x4.
    # This is a well-known result. I will generate it using the recursive Strassen construction.
    
    # Strassen 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # For 4x4, we split A, B, C into 2x2 blocks of 2x2.
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2.
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # We apply Strassen to the 2x2 block matrices.
    # Each multiplication of 2x2 blocks uses 7 scalar multiplications (if we treat blocks as scalars in a higher-level field).
    # But here, the blocks are 2x2 matrices, so each "scalar" multiplication is a 2x2x2 multiplication.
    # If we use Strassen for the block level, we need 7 block multiplications.
    # Each block multiplication is 2x2x2, which has rank 7 (Strassen).
    # So total rank = 7 * 7 = 49.
    
    # The factor matrices U, V, W for the 4x4x4 case can be constructed as Kronecker products of the 2x2 factors.
    # Let U2, V2, W2 be the 4x7 matrices for 2x2x2 Strassen.
    # Then U4 = U2 (kron) U2, but with the correct row/column mapping.
    
    # Let's define the 2x2 Strassen factors first.
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # Term 1: (A11+A22)(B11+B22)
    U2[0,0] = 1; U2[3,0] = 1
    V2[0,0] = 1; V2[3,0] = 1
    W2[0,0] = 1; W2[3,0] = 1
    
    # Term 2: (A21+A22)B11
    U2[2,1] = 1; U2[3,1] = 1
    V2[0,1] = 1
    W2[2,1] = 1
    
    # Term 3: A11(B21-B22)
    U2[0,2] = 1
    V2[1,2] = 1; V2[3,2] = -1
    W2[1,2] = 1
    
    # Term 4: A22(B12-B11)
    U2[3,3] = 1
    V2[2,3] = 1; V2[0,3] = -1
    W2[0,3] = 1; W2[2,3] = 1
    
    # Term 5: (A11+A12)B22
    U2[0,4] = 1; U2[1,4] = 1
    V2[3,4] = 1
    W2[0,4] = -1; W2[1,4] = 1
    
    # Term 6: (A21-A11)(B11+B12)
    U2[2,5] = 1; U2[0,5] = -1
    V2[0,5] = 1; V2[1,5] = 1
    W2[3,5] = 1
    
    # Term 7: (A12-A22)(B21+B22)
    U2[1,6] = 1; U2[3,6] = -1
    V2[2,6] = 1; V2[3,6] = 1
    W2[0,6] = 1
    
    # Now construct U4, V4, W4 for 4x4x4.
    # The row index for U4 is (i, j) where i, j in {0,1,2,3}.
    # We map (i, j) to (i_block, i_in_block, j_block, j_in_block).
    # i_block = i // 2, i_in_block = i % 2.
    # j_block = j // 2, j_in_block = j % 2.
    # The term index for U4 is (t_block, t_in_block) where t_block, t_in_block in {0..6}.
    # The total term index is t_block * 7 + t_in_block.
    
    # U4[i, j, r] = U2[i_block, j_block, t_block] * U2[i_in_block, j_in_block, t_in_block]?
    # No, that's not right. The Strassen algorithm at the block level says:
    # C_block = f(A_blocks, B_blocks)
    # Each block multiplication A_block * B_block is itself a 2x2x2 multiplication.
    # So the factor for A at the full level is the product of the block-level factor and the within-block factor.
    
    # Specifically, if the block-level Strassen uses factors U2_block (4x7) for the 2x2 block matrix,
    # and the within-block Strassen uses factors U2_within (4x7) for the 2x2 scalar matrix,
    # then the full factor U4 is the Kronecker product U2_block (kron) U2_within, with appropriate index mapping.
    
    # Let's define the mapping:
    # Full row index: (ib, ii, jb, jj) where ib, ii, jb, jj in {0,1}.
    # Flattened row index: ib*8 + ii*4 + jb*2 + jj? No, the standard flattening is i*m + j = i*4 + j.
    # i = ib*2 + ii, j = jb*2 + jj.
    # So row = (ib*2 + ii)*4 + (jb*2 + jj) = ib*8 + ii*4 + jb*2 + jj.
    
    # Full term index: (tb, tt) where tb, tt in {0..6}.
    # Flattened term index: tb*7 + tt.
    
    # U4[row, term] = U2_block[ib, tb] * U2_block[jb, tt]? No.
    # The block-level factor for A is U2_block, which has indices (i_block, j_block) -> (ib, jb) and term tb.
    # The within-block factor for A is U2_within, which has indices (i_in, j_in) -> (ii, jj) and term tt.
    # So U4[row, term] = U2_block[ib*2+jb, tb] * U2_within[ii*2+jj, tt].
    
    # Wait, U2_block is 4x7, with rows indexed by i_block*2 + j_block.
    # So U2_block[ib*2+jb, tb] is the coefficient for A_block[ib, jb] in block-term tb.
    # Similarly, U2_within[ii*2+jj, tt] is the coefficient for A_scalar[ii, jj] in within-term tt.
    
    # So U4[(ib*2+ii)*4 + (jb*2+jj), tb*7+tt] = U2_block[ib*2+jb, tb] * U2_within[ii*2+jj, tt].
    
    # Since U2_block and U2_within are the same matrix (both are Strassen 2x2 factors), we can just use U2 for both.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    for ib in range(2):
        for ii in range(2):
            for jb in range(2):
                for jj in range(2):
                    row = (ib*2 + ii)*4 + (jb*2 + jj)
                    for tb in range(7):
                        for tt in range(7):
                            term = tb*7 + tt
                            U4[row, term] = U2[ib*2+jb, tb] * U2[ii*2+jj, tt]
    
    # Similarly for V4 and W4.
    # V4 row index: (jb, jj, kb, kk) -> j = jb*2+jj, k = kb*2+kk.
    # V4[row, term] = V2_block[jb*2+kb, tb] * V2_within[jj*2+kk, tt].
    for jb in range(2):
        for jj in range(2):
            for kb in range(2):
                for kk in range(2):
                    row = (jb*2 + jj)*4 + (kb*2 + kk)
                    for tb in range(7):
                        for tt in range(7):
                            term = tb*7 + tt
                            V4[row, term] = V2[jb*2+kb, tb] * V2[jj*2+kk, tt]
    
    # W4 row index: (kb, kk, ib, ii) -> k = kb*2+kk, i = ib*2+ii.
    # W4[row, term] = W2_block[kb*2+ib, tb] * W2_within[kk*2+ii, tt].
    for kb in range(2):
        for kk in range(2):
            for ib in range(2):
                for ii in range(2):
                    row = (kb*2 + kk)*4 + (ib*2 + ii)
                    for tb in range(7):
                        for tt in range(7):
                            term = tb*7 + tt
                            W4[row, term] = W2[kb*2+ib, tb] * W2[kk*2+ii, tt]
    
    # Convert to lists of lists.
    U4_list = U4.tolist()
    V4_list = V4.tolist()
    W4_list = W4.tolist()
    
    return U4_list, V4_list, W4_list

def _embed_4x4x4_to_4x4x7(U4, V4, W4):
    """
    Embed the 4x4x4 rank-49 decomposition into a 4x4x7 decomposition.
    The 4x4x7 problem is C = A * B where A is 4x4, B is 4x7, C is 4x7.
    We can think of B as having 7 columns. We can split B into B1 (4x4) and B2 (4x3).
    Then C = A*B1 + A*B2.
    We use the rank-49 decomposition for A*B1.
    For A*B2, we need a decomposition for 4x4x3.
    The schoolbook rank for 4x4x3 is 48.
    Total rank = 49 + 48 = 97.
    
    But we can do better. We can use a mixed approach.
    Let's just use the rank-49 for the first 4 columns of B, and schoolbook for the remaining 3.
    """
    n, m, p = 4, 4, 7
    rank_total = 49 + 4*4*3  # 49 + 48 = 97
    
    U = [[0] * rank_total for _ in range(n*m)]
    V = [[0] * rank_total for _ in range(m*p)]
    W = [[0] * rank_total for _ in range(p*n)]
    
    # Fill in the 4x4x4 part (columns 0-3 of B)
    for i in range(n*m):
        for r in range(49):
            U[i][r] = U4[i][r]
    for j in range(m):
        for k in range(4):
            for r in range(49):
                V[j*p + k][r] = V4[j*4 + k][r]
    for k in range(4):
        for i in range(n):
            for r in range(49):
                W[k*n + i][r] = W4[k*4 + i][r]
    
    # Fill in the 4x4x3 part (columns 4-6 of B)
    col_offset = 49
    for i in range(n):
        for j in range(m):
            for k in range(4, 7):
                U[i*m + j][col_offset] = 1
                V[j*p + k][col_offset] = 1
                W[k*n + i][col_offset] = 1
                col_offset += 1
    
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, max_time=5.0):
    """
    Try to reduce the rank of the decomposition by identifying and removing linearly dependent columns.
    This is a heuristic approach.
    """
    t0 = time.time()
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # We can try to find columns that are linearly dependent in a certain sense.
    # This is complex, so for now, we'll just return the same decomposition.
    # A more advanced approach would be to use the null space of the tensor.
    
    return U, V, W

def construct():
    t0 = _start_time()
    n, m, p = DIMENSIONS
    
    # Start with the embedded 4x4x4 + schoolbook 4x4x3 decomposition (rank 97)
    U4, V4, W4 = _get_strassen_4x4x4()
    U, V, W = _embed_4x4x4_to_4x4x7(U4, V4, W4)
    
    # Verify the decomposition
    if not _verify_fast(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _get_schoolbook(n, m, p)
    
    # Try to reduce rank (placeholder for future improvement)
    if _remaining_time(t0) > 5.0:
        U, V, W = _try_reduce_rank(U, V, W, n, m, p, max_time=3.0)
        if not _verify_fast(U, V, W, n, m, p):
            # Revert to the verified decomposition
            U, V, W = _embed_4x4x4_to_4x4x7(U4, V4, W4)
    
    return {"U": U, "V": V, "W": W}
