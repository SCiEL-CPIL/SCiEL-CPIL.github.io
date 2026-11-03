"""Rewritten program for matrix multiplication 4x4 * 4x9 tensor rank search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]
_DEADLINE = time.time() + 140  # safety margin before 150s kill

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-10)

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_2x2x2():
    """Rank-7 Strassen decomposition for 2x2x2."""
    # Standard Strassen 7 terms
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[0, 2] = 1
    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[3, 3] = 1
    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[1, 6] = 1
    return U, V, W

def _embed_2x2_to_4x4(U2, V2, W2):
    """Embed a 2x2x2 decomposition into a 4x4x4 decomposition.
    We split 4x4 into four 2x2 blocks and use the 2x2 decomposition for each,
    but that gives rank 7*4=28 which is worse than schoolbook 64... no wait.
    
    Actually, to multiply 4x4 by 4x4, we can think of it as 4x4 blocks of 2x2.
    But the standard way: C = A*B where A is 4x4, B is 4x4.
    We can write A as [[A00, A01], [A10, A11]] where each is 2x2.
    Similarly B. Then C = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
                           [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    Each 2x2*2x2 product uses 7 terms, so 4 such products = 28 terms.
    But we can share: the Strassen trick applies at the block level too.
    For 4x4 using 2x2 blocks: we need 7 block-multiplications, each of which
    is 2x2*2x2 = 7 scalar multiplications. Total = 49.
    
    Let me build this properly."""
    n = 4
    # The 4x4x4 tensor can be decomposed using 7 "block" Strassen terms,
    # each of which is a 2x2x2 multiplication (7 terms).
    # Total rank = 49.
    
    # First, define the 2x2 Strassen decomposition
    U2, V2, W2 = _strassen_2x2x2()
    
    # Now, for 4x4, we have 4x4 = 2x2 blocks.
    # The block-level Strassen: 
    # Let A = [[A00, A01], [A10, A11]], B = [[B00, B10], [B01, B11]] (2x2 blocks)
    # C = A*B
    # Using Strassen at block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M3
    # C01 = M1 - M3 + M5 - M6  -- wait, let me be more careful
    # Actually the standard Strassen for 2x2:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M5 + M6
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M4
    
    # Each block product is a 2x2*2x2, using 7 scalar terms.
    # So we have 7 block-products, each with 7 scalar terms = 49 total.
    
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    # Block indices: for 4x4 matrix, row r, col c -> block (r//2, c//2), local (r%2, c%2)
    # U index: i*m+j where i is row of A, j is col of A
    # V index: j*p+k where j is row of B, k is col of B
    # W index: k*n+i where k is col of C, i is row of C
    
    # The 7 Strassen terms at block level:
    # term 0: (A00+A11)(B00+B11) -> contributes to C00, C11
    # term 1: (A10+A11)B00 -> contributes to C10, C11
    # term 2: A00(B10-B11) -> contributes to C00, C10
    # term 3: A11(B01-B00) -> contributes to C00, C11
    # term 4: (A00+A01)B11 -> contributes to C00, C01
    # term 5: (A10-A00)(B00+B01) -> contributes to C01, C10
    # term 6: (A01-A11)(B10+B11) -> contributes to C00, C01
    
    # For each block-level term, we use the 2x2 Strassen decomposition.
    # Block (bi, bj) in A corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # Block (bj, bk) in B corresponds to rows 2*bj..2*bj+1, cols 2*bk..2*bk+1
    # Block (bi, bk) in C corresponds to rows 2*bi..2*bi+1, cols 2*bk..2*bk+1
    
    # Define the 7 block-level Strassen combinations:
    # (a_blocks, b_blocks, c_blocks, signs)
    # where a_blocks is list of (block_row, block_col, sign) for A
    # b_blocks is list of (block_row, block_col, sign) for B
    # c_blocks is list of (block_row, block_col, sign) for C
    
    strassen_blocks = [
        # term 0: (A00+A11)(B00+B11) -> C00 += , C11 +=
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # term 1: (A10+A11)B00 -> C10 += , C11 -=
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
        # term 2: A00(B10-B11) -> C00 -= , C10 +=
        ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,0,-1), (1,0,1)]),
        # term 3: A11(B01-B00) -> C00 += , C11 +=
        ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (1,1,1)]),
        # term 4: (A00+A01)B11 -> C00 -= , C01 +=
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
        # term 5: (A10-A00)(B00+B01) -> C01 += , C10 -=
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(0,1,1), (1,0,-1)]),
        # term 6: (A01-A11)(B10+B11) -> C00 += , C01 -=
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (0,1,-1)]),
    ]
    
    col = 0
    for a_blocks, b_blocks, c_blocks in strassen_blocks:
        for a_br, a_bc, a_sign in a_blocks:
            for b_br, b_bc, b_sign in b_blocks:
                for c_br, c_bc, c_sign in c_blocks:
                    # For this combination, use the 2x2 Strassen decomposition
                    for r2 in range(7):
                        for lr in range(2):
                            for lc in range(2):
                                # A local index: (lr, lc) in block (a_br, a_bc)
                                i = 2*a_br + lr
                                j = 2*a_bc + lc
                                U[i*4+j, col] += a_sign * U2[lr*2+lc, r2]
                                
                                # B local index: (lr2, lc2) in block (b_br, b_bc)
                                # U2[r2] gives the A-local and B-local indices
                                # Actually, U2 is for A, V2 for B, W2 for C
                                # U2[lr*2+lc, r2] is the A factor
                                # V2[lr*2+lc, r2] is the B factor (same local indices)
                                # W2[lr*2+lc, r2] is the C factor
                                
                                j2 = 2*b_br + lr
                                k = 2*b_bc + lc
                                V[j2*4+k, col] += b_sign * V2[lr*2+lc, r2]
                                
                                i2 = 2*c_br + lr
                                k2 = 2*c_bc + lc
                                W[k2*4+i2, col] += c_sign * W2[lr*2+lc, r2]
                        col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_improve(n, m, p, time_budget=120.0):
    """Try to find a better decomposition using optimization."""
    start_time = time.time()
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try the 49-rank 4x4x4 construction embedded in 4x4x9
    # For 4x4x9, we can use the 49-rank for the 4x4x4 part and schoolbook for the remaining 4x4x5
    # 49 + 4*4*5 = 49 + 80 = 129, which is worse than 144 schoolbook.
    # Actually schoolbook for 4x4x9 is 4*4*9 = 144.
    # 49 + 80 = 129 < 144, so this is an improvement!
    
    if time.time() - start_time > time_budget - 20:
        return best, best_rank
    
    # Construct 4x4x9 using 49-rank 4x4x4 + schoolbook 4x4x5
    # U shape: (4*4, rank), V shape: (4*9, rank), W shape: (9*4, rank)
    
    # First, get the 4x4x4 rank-49 decomposition
    decomp_4x4x4 = _embed_2x2_to_4x4(*_strassen_2x2x2())
    U4 = np.array(decomp_4x4x4['U'])
    V4 = np.array(decomp_4x4x4['V'])
    W4 = np.array(decomp_4x4x4['W'])
    rank4 = U4.shape[1]
    
    # Schoolbook for 4x4x5 (the remaining 5 columns)
    rank5 = 4*4*5
    U5 = np.zeros((16, rank5))
    V5 = np.zeros((4*9, rank5))
    W5 = np.zeros((9*4, rank5))
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                U5[i*4+j, col] = 1
                V5[j*9+k, col] = 1
                W5[k*4+i, col] = 1
                col += 1
    
    total_rank = rank4 + rank5
    U = np.hstack([U4, U5])
    V = np.hstack([V4, V5])
    W = np.hstack([W4, W5])
    
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        best_rank = total_rank
        best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to further optimize using gradient-based methods
    if time.time() - start_time > time_budget - 30:
        return best, best_rank
    
    # Try a smaller rank search using random initialization and gradient descent
    # This is risky and may not converge, so we'll try briefly
    
    # For now, return what we have
    return best, best_rank

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with schoolbook
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try the improved construction
    if time.time() - start_time < _DEADLINE - 30:
        improved, improved_rank = _try_improve(n, m, p, time_budget=_DEADLINE - start_time - 20)
        if improved_rank < best_rank:
            best_result = improved
            best_rank = improved_rank
    
    return best_result
# EVOLVE-BLOCK-END
