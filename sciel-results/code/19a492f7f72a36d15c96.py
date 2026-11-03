import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Build schoolbook decomposition (rank 90) as fallback
    U_s, V_s, W_s = schoolbook(n, m, p)
    best_U, best_V, best_W = U_s, V_s, W_s
    best_rank = n * m * p
    
    # Try block decomposition with Strassen on 2x2x2 sub-blocks
    # Split: n=1+2, m=2+2+1, p=2+2+2
    # Blocks:
    #   A[0:1, 0:2] @ B[0:2, 0:2] -> C[0:1, 0:2]  (1x2 @ 2x2) schoolbook rank 4
    #   A[0:1, 2:4] @ B[2:4, 0:2] -> C[0:1, 0:2]  (1x2 @ 2x2) schoolbook rank 4
    #   A[0:1, 4:5] @ B[4:5, 0:2] -> C[0:1, 0:2]  (1x1 @ 1x2) schoolbook rank 2
    #   A[0:1, 0:2] @ B[0:2, 2:4] -> C[0:1, 2:4]  (1x2 @ 2x2) schoolbook rank 4
    #   A[0:1, 2:4] @ B[2:4, 2:4] -> C[0:1, 2:4]  (1x2 @ 2x2) schoolbook rank 4
    #   A[0:1, 4:5] @ B[4:5, 2:4] -> C[0:1, 2:4]  (1x1 @ 1x2) schoolbook rank 2
    #   A[0:1, 0:2] @ B[0:2, 4:6] -> C[0:1, 4:6]  (1x2 @ 2x2) schoolbook rank 4
    #   A[0:1, 2:4] @ B[2:4, 4:6] -> C[0:1, 4:6]  (1x2 @ 2x2) schoolbook rank 4
    #   A[0:1, 4:5] @ B[4:5, 4:6] -> C[0:1, 4:6]  (1x1 @ 1x2) schoolbook rank 2
    #   A[1:3, 0:2] @ B[0:2, 0:2] -> C[1:3, 0:2]  (2x2 @ 2x2) Strassen rank 7
    #   A[1:3, 2:4] @ B[2:4, 0:2] -> C[1:3, 0:2]  (2x2 @ 2x2) Strassen rank 7
    #   A[1:3, 4:5] @ B[4:5, 0:2] -> C[1:3, 0:2]  (2x1 @ 1x2) schoolbook rank 4
    #   A[1:3, 0:2] @ B[0:2, 2:4] -> C[1:3, 2:4]  (2x2 @ 2x2) Strassen rank 7
    #   A[1:3, 2:4] @ B[2:4, 2:4] -> C[1:3, 2:4]  (2x2 @ 2x2) Strassen rank 7
    #   A[1:3, 4:5] @ B[4:5, 2:4] -> C[1:3, 2:4]  (2x1 @ 1x2) schoolbook rank 4
    #   A[1:3, 0:2] @ B[0:2, 4:6] -> C[1:3, 4:6]  (2x2 @ 2x2) Strassen rank 7
    #   A[1:3, 2:4] @ B[2:4, 4:6] -> C[1:3, 4:6]  (2x2 @ 2x2) Strassen rank 7
    #   A[1:3, 4:5] @ B[4:5, 4:6] -> C[1:3, 4:6]  (2x1 @ 1x2) schoolbook rank 4
    
    # Total Strassen rank: 4+4+2+4+4+2+4+4+2 + 7+7+4+7+7+4+7+7+4 = 42 + 50 = 92
    # Hmm, that's worse than 90. Let me reconsider.
    
    # Better split: n=3 (all), m=2+2+1, p=2+2+2
    # Row 0 of A is 1x5, rows 1-2 of A are 2x5
    # Actually let me just try a simpler approach:
    # Split m into 2+3, use Strassen on the 2x2 part by padding
    # Or just try all possible 2x2x2 Strassen placements
    
    # Alternative: try to directly construct using the known fact that
    # the minimum rank for 3x5x6 is likely around 60-80.
    
    # Let's try a different block structure:
    # n=3, m=5=2+3, p=6
    # A[0:3, 0:2] @ B[0:2, 0:6] : 3x2 @ 2x6
    # A[0:3, 2:5] @ B[2:5, 0:6] : 3x3 @ 3x6
    
    # For 3x2 @ 2x6: schoolbook rank = 36
    # For 3x3 @ 3x6: schoolbook rank = 54
    # Total: 90 (same)
    
    # Let me try yet another approach: use the verified finding that Strassen
    # on 2x2x2 blocks can reduce rank. Let's be more careful.
    
    # Actually, from the knowledge: "Using Strassen's rank-7 decomposition for 2x2x2 
    # sub-blocks reduces the total rank from 90 (schoolbook) to 84 by replacing 8 
    # schoolbook products per 2x2x2 block with 7 Strassen products, across 6 such 
    # blocks in the 3x5x6 tensor."
    
    # So 6 blocks of 2x2x2, each saving 1 rank: 90 - 6 = 84.
    # Let me figure out the right block decomposition.
    
    # n=3 can be split as 1+2, m=5 as 2+2+1, p=6 as 2+2+2
    # The 2x2x2 blocks are where we have 2 rows, 2 cols, 2 outputs
    # From the A[1:3,:] part (2 rows), we can form 2x2x2 blocks with m and p splits.
    # m splits: [0:2], [2:4], [4:5] -> only [0:2] and [2:4] are size 2
    # p splits: [0:2], [2:4], [4:6] -> all size 2
    # So 2x2x2 blocks: A[1:3, 0:2]@B[0:2, 0:2], A[1:3, 0:2]@B[0:2, 2:4], 
    #                  A[1:3, 0:2]@B[0:2, 4:6], A[1:3, 2:4]@B[2:4, 0:2],
    #                  A[1:3, 2:4]@B[2:4, 2:4], A[1:3, 2:4]@B[2:4, 4:6]
    # That's 6 blocks of 2x2x2!
    
    # Non-2x2x2 blocks:
    # A[0:1, 0:2]@B[0:2, 0:2] : 1x2@2x2, rank 4
    # A[0:1, 0:2]@B[0:2, 2:4] : 1x2@2x2, rank 4
    # A[0:1, 0:2]@B[0:2, 4:6] : 1x2@2x2, rank 4
    # A[0:1, 2:4]@B[2:4, 0:2] : 1x2@2x2, rank 4
    # A[0:1, 2:4]@B[2:4, 2:4] : 1x2@2x2, rank 4
    # A[0:1, 2:4]@B[2:4, 4:6] : 1x2@2x2, rank 4
    # A[0:1, 4:5]@B[4:5, 0:2] : 1x1@1x2, rank 2
    # A[0:1, 4:5]@B[4:5, 2:4] : 1x1@1x2, rank 2
    # A[0:1, 4:5]@B[4:5, 4:6] : 1x1@1x2, rank 2
    # A[1:3, 4:5]@B[4:5, 0:2] : 2x1@1x2, rank 4
    # A[1:3, 4:5]@B[4:5, 2:4] : 2x1@1x2, rank 4
    # A[1:3, 4:5]@B[4:5, 4:6] : 2x1@1x2, rank 4
    
    # Total rank = 6*7 + 6*4 + 3*2 + 3*4 = 42 + 24 + 6 + 12 = 84
    
    if time.time() < deadline:
        U_block, V_block, W_block, rank_block = block_strassen(n, m, p)
        if verify_decomp(n, m, p, U_block, V_block, W_block):
            best_U, best_V, best_W = U_block, V_block, W_block
            best_rank = rank_block
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


def schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def get_strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 @ 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 intermediate products for C = A @ B where A, B are 2x2:
    # m1 = (A00 + A11)(B01 - B11)
    # m2 = (A10 + A11) B00
    # m3 = A00 (B10 - B11)
    # m4 = A11 (B00 + B11)
    # m5 = (A00 + A01) B11
    # m6 = (A10 - A00)(B00 + B01)
    # m7 = (A01 - A11)(B10 + B11)
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # For the tensor decomposition, each rank-1 term corresponds to one m_r.
    # m_r = (linear form in A) * (linear form in B)
    # C_ik = sum_r W_ikr * (sum_j U_ijr A_ij) * (sum_j V_jkr B_jk)
    
    # Let me set up the factors:
    # U factors (4 entries per term, indexed by (i,j)):
    # m1: A00 + A11, A01 - A11 -> U[:,0] = [1, 1, 0, 1] wait, let me be careful.
    # A is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
    # U indexed by [i*2+j]:
    #   index 0 = (0,0), index 1 = (0,1), index 2 = (1,0), index 3 = (1,1)
    
    # m1 = (A00 + A11) * (B01 - B11)
    # U for m1: [1, 0, 0, 1] (A00 + A11)
    # V for m1: B01 - B11. B is 2x2: B[0,0], B[0,1], B[1,0], B[1,1]
    # V indexed by [j*2+k]:
    #   index 0 = (0,0), index 1 = (0,1), index 2 = (1,0), index 3 = (1,1)
    # B01 is index 1, B11 is index 3
    # V for m1: [0, 1, 0, -1]
    
    # m2 = (A10 + A11) * B00
    # U for m2: [0, 0, 1, 1]
    # V for m2: [1, 0, 0, 0]
    
    # m3 = A00 * (B10 - B11)
    # U for m3: [1, 0, 0, 0]
    # V for m3: [0, 0, 1, -1]
    
    # m4 = A11 * (B00 + B11)
    # U for m4: [0, 0, 0, 1]
    # V for m4: [1, 0, 0, 1]
    
    # m5 = (A00 + A01) * B11
    # U for m5: [1, 1, 0, 0]
    # V for m5: [0, 0, 0, 1]
    
    # m6 = (A10 - A00) * (B00 + B01)
    # U for m6: [-1, 0, 1, 0]
    # V for m6: [1, 1, 0, 0]
    
    # m7 = (A01 - A11) * (B10 + B11)
    # U for m7: [0, 1, 0, -1]
    # V for m7: [0, 0, 1, 1]
    
    U = np.array([
        [1, 0, 0, 0],  # m1
        [0, 0, 1, 1],  # m2
        [1, 0, 0, 0],  # m3
        [0, 0, 0, 1],  # m4
        [1, 1, 0, 0],  # m5
        [-1, 0, 1, 0], # m6
        [0, 1, 0, -1]  # m7
    ], dtype=float)
    
    V = np.array([
        [0, 1, 0, -1],  # m1
        [1, 0, 0, 0],   # m2
        [0, 0, 1, -1],  # m3
        [1, 0, 0, 1],   # m4
        [0, 0, 0, 1],   # m5
        [1, 1, 0, 0],   # m6
        [0, 0, 1, 1]    # m7
    ], dtype=float)
    
    # W factors: C_ik = sum_r W_ikr * ...
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    
    # W indexed by [k*2+i]:
    #   index 0 = (0,0), index 1 = (0,1), index 2 = (1,0), index 3 = (1,1)
    
    W = np.zeros((4, 7))
    # m1 contributes to C00 (+1) and C11 (+1)
    W[0, 0] = 1   # C00
    W[3, 0] = 1   # C11
    # m2 contributes to C10 (+1) and C11 (-1)
    W[2, 1] = 1   # C10
    W[3, 1] = -1  # C11
    # m3 contributes to C01 (+1) and C11 (+1)
    W[1, 2] = 1   # C01
    W[3, 2] = 1   # C11
    # m4 contributes to C00 (+1) and C10 (+1)
    W[0, 3] = 1   # C00
    W[2, 3] = 1   # C10
    # m5 contributes to C00 (-1) and C01 (+1)
    W[0, 4] = -1  # C00
    W[1, 4] = 1   # C01
    # m6 contributes to C11 (+1)
    W[3, 5] = 1   # C11
    # m7 contributes to C00 (+1)
    W[0, 6] = 1   # C00
    
    return U, V, W


def block_strassen(n, m, p):
    """
    Build the block decomposition with Strassen on 2x2x2 blocks.
    Split: n = 1+2, m = 2+2+1, p = 2+2+2
    """
    # n splits: [0:1], [1:3]
    # m splits: [0:2], [2:4], [4:5]
    # p splits: [0:2], [2:4], [4:6]
    
    total_rank = 84
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    
    # Helper to embed a sub-block decomposition
    def embed(U_sub, V_sub, W_sub, i0, j0, k0, ni, nj, nk):
        """Embed sub-block at offset (i0, j0, k0) with sizes (ni, nj, nk)."""
        nonlocal col
        for r in range(U_sub.shape[1]):
            # U: (ni*nj, 1) -> embed at (i0..i0+ni-1, j0..j0+nj-1)
            for a in range(ni):
                for b in range(nj):
                    U[(i0 + a) * m + (j0 + b), col] = U_sub[a * nj + b, r]
            # V: (nj*nk, 1) -> embed at (j0..j0+nj-1, k0..k0+nk-1)
            for b in range(nj):
                for c in range(nk):
                    V[(j0 + b) * p + (k0 + c), col] = V_sub[b * nk + c, r]
            # W: (nk*ni, 1) -> embed at (k0..k0+nk-1, i0..i0+ni-1)
            for c in range(nk):
                for a in range(ni):
                    W[(k0 + c) * n + (i0 + a), col] = W_sub[c * ni + a, r]
            col += 1
    
    # 6 Strassen 2x2x2 blocks: A[1:3, j:j+2] @ B[j:j+2, k:k+2]
    U_s, V_s, W_s = get_strassen_2x2x2()
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            embed(U_s, V_s, W_s, 1, j0, k0, 2, 2, 2)
    
    # 6 schoolbook 1x2x2 blocks: A[0:1, j:j+2] @ B[j:j+2, k:k+2]
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            U_122, V_122, W_122 = schoolbook(1, 2, 2)
            embed(U_122, V_122, W_122, 0, j0, k0, 1, 2, 2)
    
    # 3 schoolbook 1x1x2 blocks: A[0:1, 4:5] @ B[4:5, k:k+2]
    for k0 in [0, 2, 4]:
        U_112, V_112, W_112 = schoolbook(1, 1, 2)
        embed(U_112, V_112, W_112, 0, 4, k0, 1, 1, 2)
    
    # 3 schoolbook 2x1x2 blocks: A[1:3, 4:5] @ B[4:5, k:k+2]
    for k0 in [0, 2, 4]:
        U_212, V_212, W_212 = schoolbook(2, 1, 2)
        embed(U_212, V_212, W_212, 1, 4, k0, 2, 1, 2)
    
    assert col == total_rank, f"Column count {col} != expected {total_rank}"
    
    return U, V, W, total_rank


def verify_decomp(n, m, p, U, V, W):
    """
    Verify if the factors U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    C_ik = sum_c W_ikc * (sum_j U_ijc A_ij) * (sum_j V_jkc B_jk)
    """
    rank = U.shape[1]
    np.random.seed(42)
    A = np.random.randint(-5, 5, (n, m))
    B = np.random.randint(-5, 5, (m, p))
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n).T  # (n, p)
        
        X = np.sum(A * u_c, axis=1)  # (n,)
        Y = np.sum(B * v_c, axis=0)  # (p,)
        C_computed += w_c * np.outer(X, Y)
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        return False
    return True

# EVOLVE-BLOCK-END
