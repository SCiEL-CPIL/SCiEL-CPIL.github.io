"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """
    Returns U (4,7), V (4,7), W (4,7) for the rank-7 Strassen decomposition
    of 2x2 matrix multiplication.
    
    Indexing: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    The decomposition satisfies: sum_c U[i,c]*V[j,c]*W[k,c] = delta(i,j,k)
    where delta is 1 if (i,j,k) is a valid matmul index triple, else 0.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = A00 * (B01 - B11)
    U[0, 0] = 1
    V[1, 0] = 1
    V[3, 0] = -1
    W[1, 0] = 1
    W[3, 0] = 1
    
    # M2 = (A00 + A10) * B11
    U[0, 1] = 1
    U[2, 1] = 1
    V[3, 1] = 1
    W[0, 1] = -1
    W[2, 1] = 1
    
    # M3 = (A01 - A11) * B00
    U[1, 2] = 1
    U[3, 2] = -1
    V[0, 2] = 1
    W[0, 2] = 1
    W[3, 2] = -1
    
    # M4 = A11 * (B10 - B00)
    U[3, 3] = 1
    V[2, 3] = 1
    V[0, 3] = -1
    W[0, 3] = 1
    W[2, 3] = 1
    
    # M5 = (A00 + A01) * (B00 + B11)
    U[0, 4] = 1
    U[1, 4] = 1
    V[0, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1
    W[1, 4] = 1
    
    # M6 = (A10 + A11) * (B00 + B01)
    U[2, 5] = 1
    U[3, 5] = 1
    V[0, 5] = 1
    V[1, 5] = 1
    W[2, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A00 - A10) * (B10 + B11)
    U[0, 6] = 1
    U[2, 6] = -1
    V[2, 6] = 1
    V[3, 6] = 1
    W[1, 6] = 1
    W[3, 6] = -1
    
    return U, V, W


def _verify_strassen_core():
    """Verify the Strassen 2x2x2 core is correct."""
    U, V, W = _strassen_2x2x2()
    # Contract: sum_c U[i,c] * V[j,c] * W[k,c] should give the matmul tensor
    # Tensor T[i,j,k] = 1 if k == j (for 2x2: C[i,k] = sum_j A[i,j] B[j,k])
    # Actually T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears, i.e., always for valid indices.
    # The matmul tensor has T[i,j,k] = 1 for all i,j,k in {0,1} (it's the identity in the middle).
    # Wait no. The matrix multiplication tensor T_{ijk} where C_k = sum_i,j A_i T_{ijk} B_j.
    # T_{ijk} = 1 if i==k (row of A matches row of C), 0 otherwise.
    # So T[i,j,k] = delta_{ik}.
    T = np.einsum('ic,jc,kc->ijk', U, V, W)
    expected = np.zeros((2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected[i, j, k] = 1.0 if i == k else 0.0
    return np.allclose(T, expected, atol=1e-10)


def _build_rank56():
    """
    Build the rank-56 decomposition for 2x4 * 4x8 -> 2x8.
    
    Strategy: The 2x4x8 matmul tensor can be seen as a 2x2 block matmul where:
    - A is split into 2 blocks of 2x2 (left and right halves of columns)
    - B is split into 4 blocks of 2x2 (2 row-blocks x 2 col-blocks)
    - C is split into 4 blocks of 2x2 (2 col-blocks, 1 row-block of 2)
    
    C = A_left * B_left + A_right * B_right
    where B_left = [B00; B10] (4x4) and B_right = [B01; B11] (4x4).
    
    Each 2x2 * 4x4 product = 2x2 * (2x2 stacked 2x2) can be done with 2 Strassen blocks.
    Total: 2 (A blocks) * 2 (B row-blocks) * 2 (B col-blocks) * 7 = 56.
    
    More precisely, we tile 8 independent 2x2x2 Strassen problems.
    """
    n, m, p = 2, 4, 8  # A: n x m, B: m x p, C: n x p
    
    U_core, V_core, W_core = _strassen_2x2x2()
    rank = 56
    
    U = np.zeros((n * m, rank))  # (8, 56)
    V = np.zeros((m * p, rank))  # (32, 56)
    W = np.zeros((p * n, rank))  # (16, 56)
    
    col = 0
    # A is 2x4. Split columns into 2 groups of 2: A[:, 0:2] and A[:, 2:4]
    # B is 4x8. Split into 4 blocks of 2x2:
    #   B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    #   B[2:4, 0:2], B[2:4, 2:4], B[2:4, 4:6], B[2:4, 6:8]
    # C is 2x8. Split into 4 blocks of 2x2:
    #   C[:, 0:2], C[:, 2:4], C[:, 4:6], C[:, 6:8]
    
    # The matmul: C[:, c_block*2 : c_block*2+2] = A[:, 0:2] * B[0:2, c_block*2 : c_block*2+2]
    #                                               + A[:, 2:4] * B[2:4, c_block*2 : c_block*2+2]
    
    for a_col_block in range(2):  # 0: A[:,0:2], 1: A[:,2:4]
        for b_row_block in range(2):  # 0: B[0:2,:], 1: B[2:4,:]
            for c_col_block in range(2):  # 0: C[:,0:4], 1: C[:,4:8]
                # Wait, I need to be more careful.
                # C[:, c] = sum_j A[:,j] * B[j,c]
                # C[:, 0:2] = A[:,0:2] * B[0:2, 0:2] + A[:,2:4] * B[2:4, 0:2]
                # C[:, 2:4] = A[:,0:2] * B[0:2, 2:4] + A[:,2:4] * B[2:4, 2:4]
                # C[:, 4:6] = A[:,0:2] * B[0:2, 4:6] + A[:,2:4] * B[2:4, 4:6]
                # C[:, 6:8] = A[:,0:2] * B[0:2, 6:8] + A[:,2:4] * B[2:4, 6:8]
                
                # So for each output column block (4 blocks of 2 cols),
                # we have 2 Strassen multiplications (one for each A column block).
                # Total: 4 * 2 * 7 = 56. Good.
                
                pass  # This loop structure is wrong, let me restructure.
    
    # Restructure:
    col = 0
    for c_col_block in range(4):  # 4 output column blocks of 2
        for a_col_block in range(2):  # 2 A column blocks
            # b_row_block is determined: a_col_block 0 -> B rows 0:2, a_col_block 1 -> B rows 2:4
            b_row_block = a_col_block
            
            # A block: rows 0:2, cols a_col_block*2 : a_col_block*2+2
            # B block: rows b_row_block*2 : b_row_block*2+2, cols c_col_block*2 : c_col_block*2+2
            # C block: rows 0:2, cols c_col_block*2 : c_col_block*2+2
            
            for s in range(7):
                # U: A indices
                for i in range(2):
                    for j in range(2):
                        val = U_core[i * 2 + j, s]
                        if val != 0:
                            a_row = i
                            a_col = a_col_block * 2 + j
                            U[a_row * m + a_col, col] = val
                
                # V: B indices
                for i in range(2):
                    for j in range(2):
                        val = V_core[i * 2 + j, s]
                        if val != 0:
                            b_row = b_row_block * 2 + i
                            b_col = c_col_block * 2 + j
                            V[b_row * p + b_col, col] = val
                
                # W: C indices, trace convention W[k*n + i, r]
                # k = column of C, i = row of C
                for i in range(2):
                    for j in range(2):
                        val = W_core[i * 2 + j, s]
                        if val != 0:
                            c_row = i
                            c_col = c_col_block * 2 + j
                            W[c_col * n + c_row, col] = val
                
                col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _build_schoolbook():
    """Build the rank-64 schoolbook decomposition as fallback."""
    n, m, p = 2, 4, 8
    rank = n * m * p  # 64
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    
    return U, V, W


def _verify_decomposition(U, V, W, n=2, m=4, p=8):
    """
    Vectorized verification: check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    for all valid (i,j,k) triples.
    """
    # Reshape for einsum
    U_r = U.reshape(n, m, -1)   # (2, 4, rank)
    V_r = V.reshape(m, p, -1)   # (4, 8, rank)
    W_r = W.reshape(p, n, -1)   # (8, 2, rank)
    
    # Contract over rank: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Using einsum: 'ijr,jkr,kir->ijk'
    T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    
    # Expected: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    expected = np.ones((n, m, p))
    
    return np.allclose(T, expected, atol=1e-9), T


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # First, verify the Strassen core
    core_ok = _verify_strassen_core()
    
    if not core_ok:
        # Core is wrong, use schoolbook
        U, V, W = _build_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Build rank-56 decomposition
    U, V, W = _build_rank56()
    
    # Verify with vectorized einsum
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        ok, T = _verify_decomposition(U, V, W, n, m, p)
        if ok:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        else:
            # Debug: find mismatches
            mismatches = np.argwhere(np.abs(T - 1.0) > 1e-9)
            # Fall back to schoolbook
            U, V, W = _build_schoolbook()
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Time pressure, return what we have (should be correct if core is verified)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
