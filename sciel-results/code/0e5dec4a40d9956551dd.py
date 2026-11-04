"""Rewritten program for 4x4x8 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 130  # safety margin before 150s kill


def _time_remaining():
    return _DEADLINE - (time.time() - _start_time)


def _verify(U, V, W, n, m, p, num_trials=3, seed=42):
    """Verify that the decomposition is exact for random inputs."""
    rank = len(U[0]) if U else 0
    rng = np.random.RandomState(seed)
    for _ in range(num_trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        C_expected = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            u_mat = np.array(U[:, r]).reshape(n, m)
            v_mat = np.array(V[:, r]).reshape(m, p)
            w_mat = np.array(W[:, r]).reshape(p, n)
            M_r = u_mat @ v_mat
            C_decomp += w_mat.T * M_r
        if not np.allclose(C_decomp, C_expected, atol=1e-9):
            return False
    return True


def _strassen_2x2():
    """
    Rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7).
    
    The 7 multiplications:
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
    # Index convention: row-major, so A11=0, A12=1, A21=2, A22=3
    # B11=0, B12=1, B21=2, B22=3
    # C11=0, C12=1, C21=2, C22=3
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1  # C11 += M1, C22 -= M1 ... wait
    
    # Let me redo this more carefully.
    # C[i,j] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For 2x2: U is 4x7 (indices 0-3 for A), V is 4x7 (indices 0-3 for B), W is 4x7 (indices 0-3 for C)
    # W index: k*n + i where n=2, so W[0*2+0]=W[0] is C[0,0]=C11, W[1*2+0]=W[2] is C[1,0]=C21
    # W[0*2+1]=W[1] is C[0,1]=C12, W[1*2+1]=W[3] is C[1,1]=C22
    
    # Actually let me re-read the convention: W[k*n+i, r]
    # So for C[i,k], the W index is k*n + i.
    # C11: i=0, k=0 -> W[0*2+0] = W[0]
    # C12: i=0, k=1 -> W[1*2+0] = W[2]
    # C21: i=1, k=0 -> W[0*2+1] = W[1]
    # C22: i=1, k=1 -> W[1*2+1] = W[3]
    
    # U[i*m+j, r]: A[i,j] with m=2
    # A11: i=0,j=0 -> U[0*2+0] = U[0]
    # A12: i=0,j=1 -> U[0*2+1] = U[1]
    # A21: i=1,j=0 -> U[1*2+0] = U[2]
    # A22: i=1,j=1 -> U[1*2+1] = U[3]
    
    # V[j*p+k, r]: B[j,k] with p=2
    # B11: j=0,k=0 -> V[0*2+0] = V[0]
    # B12: j=0,k=1 -> V[0*2+1] = V[1]
    # B21: j=1,k=0 -> V[1*2+0] = V[2]
    # B22: j=1,k=1 -> V[1*2+1] = V[3]
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    # C11 += M1, C22 += M1
    W[0, 0] += 1  # C11
    W[3, 0] += 1  # C22
    
    # Term 2: M2 = (A21+A22)B11
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    # C21 += M2, C22 -= M2
    W[1, 1] += 1  # C21
    W[3, 1] += -1  # C22
    
    # Term 3: M3 = A11(B12-B22)
    U[0, 2] += 1
    V[1, 2] += 1; V[3, 2] += -1
    # C12 += M3, C22 += M3
    W[2, 2] += 1  # C12
    W[3, 2] += 1  # C22
    
    # Term 4: M4 = A22(B21-B11)
    U[3, 3] += 1
    V[2, 3] += 1; V[0, 3] += -1
    # C11 += M4, C21 += M4
    W[0, 3] += 1  # C11
    W[1, 3] += 1  # C21
    
    # Term 5: M5 = (A11+A12)B22
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    # C11 -= M5, C12 += M5
    W[0, 4] += -1  # C11
    W[2, 4] += 1  # C12
    
    # Term 6: M6 = (A21-A11)(B11+B12)
    U[2, 5] += 1; U[0, 5] += -1
    V[0, 5] += 1; V[1, 5] += 1
    # C22 += M6
    W[3, 5] += 1  # C22
    
    # Term 7: M7 = (A12-A22)(B21+B22)
    U[1, 6] += 1; U[3, 6] += -1
    V[2, 6] += 1; V[3, 6] += 1
    # C11 += M7
    W[0, 6] += 1  # C11
    
    return U, V, W


def _embed_block(U_block, V_block, W_block, row_offset_U, row_offset_V, row_offset_W, n, m, p):
    """
    Embed a block decomposition into larger matrices.
    U_block: (n*m, rank), V_block: (m*p, rank), W_block: (p*n, rank)
    row_offset_U: offset in the full U matrix (n*m total)
    row_offset_V: offset in the full V matrix (m*p total)
    row_offset_W: offset in the full W matrix (p*n total)
    """
    rank = U_block.shape[1]
    U_full = np.zeros((n * m, rank))
    V_full = np.zeros((m * p, rank))
    W_full = np.zeros((p * n, rank))
    
    # Copy blocks
    U_full[row_offset_U:row_offset_U + n*m, :] = U_block
    V_full[row_offset_V:row_offset_V + m*p, :] = V_block
    W_full[row_offset_W:row_offset_W + p*n, :] = W_block
    
    return U_full, V_full, W_full


def _strassen_4x4():
    """
    Build rank-49 decomposition for 4x4x4 using Strassen recursion.
    Split 4x4 into 2x2 blocks, use 7 Strassen-2x2 decompositions.
    """
    n = m = p = 4
    
    U2, V2, W2 = _strassen_2x2()
    rank2 = U2.shape[1]  # 7
    
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    
    # 7 terms from top-level Strassen
    # For each term t (0..6), we have a 2x2x2 Strassen decomposition
    # The top-level structure:
    # A is 4x4, split into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # B is 4x4, split into 2x2 blocks: B11, B12, B21, B22 (each 2x2)
    # C is 4x4, split into 2x2 blocks: C11, C12, C21, C22 (each 2x2)
    
    # Top-level Strassen:
    # M1 = (A11+A22)(B11+B22) -> contributes to C11, C22
    # M2 = (A21+A22)B11 -> contributes to C21, C22
    # M3 = A11(B12-B22) -> contributes to C12, C22
    # M4 = A22(B21-B11) -> contributes to C11, C21
    # M5 = (A11+A12)B22 -> contributes to C11, C12
    # M6 = (A21-A11)(B11+B12) -> contributes to C22
    # M7 = (A12-A22)(B21+B22) -> contributes to C11
    
    # Each Mt is a 2x2 matrix multiplication, decomposed by Strassen-2x2 (rank 7)
    
    # Block positions in 4x4 (row-major):
    # A11: rows 0-1, cols 0-1 -> U indices 0-3
    # A12: rows 0-1, cols 2-3 -> U indices 4-7
    # A21: rows 2-3, cols 0-1 -> U indices 8-11
    # A22: rows 2-3, cols 2-3 -> U indices 12-15
    
    # B11: rows 0-1, cols 0-1 -> V indices 0-3
    # B12: rows 0-1, cols 2-3 -> V indices 4-7
    # B21: rows 2-3, cols 0-1 -> V indices 8-11
    # B22: rows 2-3, cols 2-3 -> V indices 12-15
    
    # C11: rows 0-1, cols 0-1 -> W indices 0-3 (W[k*4+i], k=0,1; i=0,1)
    # C12: rows 0-1, cols 2-3 -> W indices 8-11 (k=2,3; i=0,1)
    # C21: rows 2-3, cols 0-1 -> W indices 4-7 (k=0,1; i=2,3)
    # C22: rows 2-3, cols 2-3 -> W indices 12-15 (k=2,3; i=2,3)
    
    # For each top-level term t, the sub-multiplication uses U2, V2, W2
    # We need to map the 2x2 block indices to the 4x4 indices
    
    # Define the mapping for each top-level term
    # For term t, the A factor is a combination of 2x2 blocks
    # We'll handle each term separately
    
    # Term 1: M1 = (A11+A22)(B11+B22)
    # A factor: A11 + A22, B factor: B11 + B22
    # Result contributes to C11 and C22
    
    # For the sub-decomposition, U2[i,j,r] gives coefficients for A_sub[i,j]
    # A_sub corresponds to the 2x2 block being multiplied
    
    # This is getting complex. Let me use a different approach.
    # Instead of trying to compose Strassen recursively in a complex way,
    # let me just use the schoolbook for 4x4x4 (rank 64) and combine two for 4x4x8.
    # That gives rank 128, same as schoolbook for 4x4x8.
    
    # Actually, let me try a simpler composition.
    # For each of the 7 top-level terms, the sub-problem is 2x2x2.
    # The U factor for the sub-problem is embedded into the appropriate 2x2 block of A.
    # Similarly for V and W.
    
    # Let me define the block offsets:
    # For A: A11 at offset 0, A12 at offset 4, A21 at offset 8, A22 at offset 12
    # For B: B11 at offset 0, B12 at offset 4, B21 at offset 8, B22 at offset 12
    # For C: C11 at offset 0, C12 at offset 8, C21 at offset 4, C22 at offset 12
    
    # Top-level term definitions: (A_blocks, B_blocks, C_blocks, signs)
    # Each entry is a list of (block_idx, sign)
    
    terms = [
        # (A_blocks, B_blocks, C_blocks)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),       # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),               # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),               # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),               # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),               # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),               # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),               # M7
    ]
    
    block_offsets_U = [0, 4, 8, 12]  # A11, A12, A21, A22
    block_offsets_V = [0, 4, 8, 12]  # B11, B12, B21, B22
    block_offsets_W = [0, 8, 4, 12]  # C11, C12, C21, C22
    
    for t in range(7):
        A_blocks, B_blocks, C_blocks = terms[t]
        base_col = t * 7
        
        for (a_block, a_sign) in A_blocks:
            a_off = block_offsets_U[a_block]
            for i in range(4):
                for r in range(7):
                    U[a_off + i, base_col + r] += a_sign * U2[i, r]
        
        for (b_block, b_sign) in B_blocks:
            b_off = block_offsets_V[b_block]
            for j in range(4):
                for r in range(7):
                    V[b_off + j, base_col + r] += b_sign * V2[j, r]
        
        for (c_block, c_sign) in C_blocks:
            c_off = block_offsets_W[c_block]
            for k in range(4):
                for r in range(7):
                    W[c_off + k, base_col + r] += c_sign * W2[k, r]
    
    return U, V, W


def _build_4x4x8_strassen():
    """
    Build decomposition for 4x4x8 using two Strassen-4x4 decompositions.
    Rank = 2 * 49 = 98.
    """
    n, m, p = 4, 4, 8
    
    U4, V4, W4 = _strassen_4x4()
    rank4 = U4.shape[1]  # 49
    
    U = np.zeros((n * m, 2 * rank4))  # (16, 98)
    V = np.zeros((m * p, 2 * rank4))  # (32, 98)
    W = np.zeros((p * n, 2 * rank4))  # (32, 98)
    
    # First half: B[:, 0:4], C[:, 0:4]
    for r in range(rank4):
        U[:, r] = U4[:, r]
        V[:m*4, r] = V4[:, r]  # V4 is (16, 49), first 16 rows of V
        W[:p*4, r] = W4[:, r]  # W4 is (16, 49), first 16 rows of W... 
        # Wait, W is (p*n, rank) = (32, 98)
        # W4 is (p4*n, rank4) = (16, 49) where p4=4
        # For the first half, C[:, 0:4], so W indices 0-15
        W[:4*n, r] = W4[:, r]
    
    # Second half: B[:, 4:8], C[:, 4:8]
    for r in range(rank4):
        col = rank4 + r
        U[:, col] = U4[:, r]
        V[m*4:, col] = V4[:, r]  # V indices 16-31
        W[4*n:, col] = W4[:, r]  # W indices 16-31
    
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Try Strassen-based decomposition (rank 98)
    try:
        U, V, W = _build_4x4x8_strassen()
        if _verify(U, V, W, n, m, p, num_trials=3):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback to schoolbook (rank 128)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
