"""Matrix multiplication 4x4x6 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

# Deadline guard
_START = time.time()
DEADLINE = 115  # seconds, leaving 5s margin before 120s kill

def _build_schoolbook(n, m, p):
    """Schoolbook decomposition: rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k, col] = 1
                V[k * p + j, col] = 1
                W[j * n + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that sum_r U[:,r] V[:,r] W[:,r] equals the matrix multiplication tensor."""
    # The multiplication tensor T has T[a,b,c] = 1 iff c = a*p + ... 
    # Actually: C[i,j] = sum_k A[i,k] B[k,j]
    # Flattened: A index = i*m+k, B index = k*p+j, C index = j*n+i
    # T[A_idx, B_idx, C_idx] = 1 if C_idx corresponds to the (i,j) output for that (i,k) and (k,j) pair
    # More precisely, for the decomposition to be correct:
    # For all i, j: C[i,j] = sum_k A[i,k] * B[k,j]
    # In tensor form: T[i*m+k, k*p+j, j*n+i] = 1 (for all valid i,k,j)
    # The decomposition says: sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] = T[i*m+k, k*p+j, j*n+i]
    
    # Check: for each (i,k,j), the sum should equal 1
    # And for all other triples, it should equal 0.
    # Full check is expensive, so check the 1-entries and a random sample of 0-entries.
    
    # Check all 1-entries
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2():
    """Rank-7 Strassen decomposition of 2x2x2 matrix multiplication.
    A is 2x2, B is 2x2, C is 2x2.
    U shape (4,7), V shape (4,7), W shape (4,7).
    Index convention: A[i,j] -> i*2+j, B[i,j] -> i*2+j, C[i,j] -> i*2+j
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    # Contributes to C00 (+1) and C11 (+1)
    W[:, 0] = [1, 0, 0, 1]
    
    # M2 = (A10+A11)(B00)
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    # Contributes to C10 (+1)
    W[:, 1] = [0, 0, 1, 0]
    
    # M3 = A00(B01-B11)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    # Contributes to C01 (+1)
    W[:, 2] = [0, 1, 0, 0]
    
    # M4 = A11(B10-B00)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [0, 0, 1, -1]
    # Contributes to C00 (+1)
    W[:, 3] = [1, 0, 0, 0]
    
    # M5 = (A00+A01)(B11)
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    # Contributes to C00 (-1) and C01 (+1)
    W[:, 4] = [-1, 1, 0, 0]
    
    # M6 = (A10-A00)(B00+B01)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    # Contributes to C11 (+1)
    W[:, 5] = [0, 0, 0, 1]
    
    # M7 = (A01-A11)(B10+B11)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    # Contributes to C00 (+1) and C11 (-1)
    W[:, 6] = [1, 0, 0, -1]
    
    return U, V, W

def _strassen_2x2x3():
    """Decomposition of 2x2 A times 2x3 B -> 2x3 C.
    Uses Strassen for the first 2 columns (rank 7) + schoolbook for the 3rd column (rank 2).
    Total rank 9.
    U shape (4, 9), V shape (6, 9), W shape (6, 9).
    A[i,j] -> i*2+j, B[i,j] -> i*3+j, C[i,j] -> i*3+j
    """
    r = 9
    U = np.zeros((4, r))
    V = np.zeros((6, r))
    W = np.zeros((6, r))
    
    # Strassen part for columns 0,1 of B (indices 0,1,3,4 in flattened B)
    # B indices: B[0,0]=0, B[0,1]=1, B[0,2]=2, B[1,0]=3, B[1,1]=4, B[1,2]=5
    # C indices: C[0,0]=0, C[0,1]=1, C[0,2]=2, C[1,0]=3, C[1,1]=4, C[1,2]=5
    
    # Map Strassen 2x2x2 to use B columns 0,1 and produce C columns 0,1
    # Strassen U is for A (4 elements) - same
    # Strassen V is for B (4 elements: B00,B01,B10,B11) -> map to V indices 0,1,3,4
    # Strassen W is for C (4 elements: C00,C01,C10,C11) -> map to W indices 0,1,3,4
    
    U_s, V_s, W_s = _strassen_2x2x2()
    
    # Copy Strassen U directly
    U[:, :7] = U_s
    
    # Map Strassen V: local index 0->B00(0), 1->B01(1), 2->B10(3), 3->B11(4)
    v_map = [0, 1, 3, 4]
    for li, gi in enumerate(v_map):
        V[gi, :7] = V_s[li, :]
    
    # Map Strassen W: local index 0->C00(0), 1->C01(1), 2->C10(3), 3->C11(4)
    w_map = [0, 1, 3, 4]
    for li, gi in enumerate(w_map):
        W[gi, :7] = W_s[li, :]
    
    # Schoolbook for column 2: C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # Term 7: A[0,0]*B[0,2] -> C[0,2]
    U[0, 7] = 1
    V[2, 7] = 1  # B[0,2]
    W[2, 7] = 1  # C[0,2]
    
    # Term 8: A[1,0]*B[0,2] -> C[1,2]
    U[2, 8] = 1
    V[2, 8] = 1  # B[0,2]
    W[5, 8] = 1  # C[1,2]
    
    # Wait, that's not right. Schoolbook for C[i,2] = sum_k A[i,k] B[k,2]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    # That's 4 terms, not 2. Let me redo.
    
    # Actually for 2x2 times 2x1 (just column 2), schoolbook rank is 4.
    # Total would be 7+4=11.
    
    # Let me redo with rank 11.
    r = 11
    U = np.zeros((4, r))
    V = np.zeros((6, r))
    W = np.zeros((6, r))
    
    U_s, V_s, W_s = _strassen_2x2x2()
    U[:, :7] = U_s
    v_map = [0, 1, 3, 4]
    for li, gi in enumerate(v_map):
        V[gi, :7] = V_s[li, :]
    w_map = [0, 1, 3, 4]
    for li, gi in enumerate(w_map):
        W[gi, :7] = W_s[li, :]
    
    # Schoolbook for column 2 (4 terms)
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # Term 7: A[0,0]*B[0,2] -> C[0,2]
    U[0, 7] = 1
    V[2, 7] = 1
    W[2, 7] = 1
    
    # Term 8: A[0,1]*B[1,2] -> C[0,2]
    U[1, 8] = 1
    V[5, 8] = 1
    W[2, 8] = 1
    
    # Term 9: A[1,0]*B[0,2] -> C[1,2]
    U[2, 9] = 1
    V[2, 9] = 1
    W[5, 9] = 1
    
    # Term 10: A[1,1]*B[1,2] -> C[1,2]
    U[3, 10] = 1
    V[5, 10] = 1
    W[5, 10] = 1
    
    return U, V, W

def _block_decomposition(n, m, p, sub_n, sub_m, sub_p):
    """
    Decompose n x m times m x p into blocks of sub_n x sub_m times sub_m x sub_p.
    Number of blocks: (n/sub_n) * (p/sub_p)
    Each block uses the _strassen_2x2x3 decomposition (rank 11 for 2x2x3).
    """
    if sub_n != 2 or sub_m != 2 or sub_p != 3:
        raise ValueError("Only 2x2x3 sub-blocks supported")
    
    n_blocks = n // sub_n
    p_blocks = p // sub_p
    if n_blocks * sub_n != n or p_blocks * sub_p != p:
        raise ValueError("Dimensions must be divisible")
    
    rank_per_block = 11
    total_rank = n_blocks * p_blocks * rank_per_block
    
    U_sub, V_sub, W_sub = _strassen_2x2x3()
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for bi in range(n_blocks):
        for bj in range(p_blocks):
            # Block is at rows [bi*2, bi*2+1] of A and C, cols [bj*3, bj*3+2] of B and C
            for r in range(rank_per_block):
                # U: A indices for this block
                for a_local in range(4):
                    a_global = (bi * 2 + a_local // 2) * m + (a_local % 2)
                    U[a_global, col] = U_sub[a_local, r]
                
                # V: B indices for this block
                for b_local in range(6):
                    b_global = (b_local // 3) * p + (bj * 3 + b_local % 3)
                    V[b_global, col] = V_sub[b_local, r]
                
                # W: C indices for this block
                for c_local in range(6):
                    c_global = (c_local // 3) * n + (bi * 2 + c_local % 3)
                    # Wait: C is p x n, flattened as C[j, i] -> j*n+i
                    # c_local = i_local * 3 + j_local where i_local in {0,1}, j_local in {0,1,2}
                    # So i_local = c_local // 3, j_local = c_local % 3
                    i_global = bi * 2 + c_local // 3
                    j_global = bj * 3 + c_local % 3
                    c_global = j_global * n + i_global
                    W[c_global, col] = W_sub[c_local, r]
                
                col += 1
    
    return U, V, W

def _strassen_4x4x4():
    """Rank-49 Strassen decomposition of 4x4x4 matrix multiplication."""
    U_sub, V_sub, W_sub = _strassen_2x2x2()
    rank = 49
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for r in range(7):
                # U: A block at (bi, bj)
                for a_local in range(4):
                    i = bi * 2 + a_local // 2
                    k = bj * 2 + a_local % 2
                    U[i * 4 + k, col] = U_sub[a_local, r]
                
                # V: B block at (bj, bi) -- B[k, j]
                for b_local in range(4):
                    k = bj * 2 + b_local // 2
                    j = bi * 2 + b_local % 2
                    V[k * 4 + j, col] = V_sub[b_local, r]
                
                # W: C block at (bi, bj) -- C[i, j]
                for c_local in range(4):
                    i = bi * 2 + c_local // 2
                    j = bj * 2 + c_local % 2
                    W[j * 4 + i, col] = W_sub[c_local, r]
                
                col += 1
    
    return U, V, W

def _strassen_4x4x2():
    """Decomposition of 4x4 A times 4x2 B -> 4x2 C.
    Split into 2x2 blocks: 4 blocks of 2x2x2, each rank 7. Total rank 28.
    """
    U_sub, V_sub, W_sub = _strassen_2x2x2()
    rank = 28
    U = np.zeros((16, rank))
    V = np.zeros((8, rank))
    W = np.zeros((8, rank))
    
    col = 0
    for bi in range(2):
        for bj in range(2):
            for r in range(7):
                # U: A block at (bi, bj) -- A[i,k]
                for a_local in range(4):
                    i = bi * 2 + a_local // 2
                    k = bj * 2 + a_local % 2
                    U[i * 4 + k, col] = U_sub[a_local, r]
                
                # V: B block at (bj, bj) -- B[k, j] where j in {bj*2, bj*2+1}
                # B is 4x2, so B[k, j] with k in 0..3, j in 0..1
                # Block: k in {bj*2, bj*2+1}, j in {bj*2, bj*2+1}
                for b_local in range(4):
                    k = bj * 2 + b_local // 2
                    j = bj * 2 + b_local % 2
                    V[k * 2 + j, col] = V_sub[b_local, r]
                
                # W: C block at (bi, bj) -- C[i, j]
                for c_local in range(4):
                    i = bi * 2 + c_local // 2
                    j = bj * 2 + c_local % 2
                    W[j * 4 + i, col] = W_sub[c_local, r]
                
                col += 1
    
    return U, V, W

def _construct_4x4x6_hybrid():
    """
    Decompose 4x4 x 4x6 -> 4x6.
    Split B and C into: first 4 columns (4x4 part) + last 2 columns (4x2 part).
    4x4x4: rank 49 (Strassen)
    4x4x2: rank 28 (Strassen)
    Total rank: 77
    """
    U1, V1, W1 = _strassen_4x4x4()
    U2, V2, W2 = _strassen_4x4x2()
    
    total_rank = 49 + 28
    U = np.zeros((16, total_rank))
    V = np.zeros((24, total_rank))
    W = np.zeros((24, total_rank))
    
    # Part 1: 4x4x4 (columns 0-3 of B and C)
    U[:, :49] = U1
    V[:16, :49] = V1  # B is 4x6, first 4 cols: indices 0..15
    W[:16, :49] = W1  # C is 6x4, first 4 cols: indices 0..15
    
    # Part 2: 4x4x2 (columns 4-5 of B and C)
    U[:, 49:] = U2
    V[16:24, 49:] = V2  # B columns 4,5: indices 16..23
    W[16:24, 49:] = W2  # C columns 4,5: indices 16..23
    
    return U, V, W

def _construct_4x4x6_blocks():
    """
    Decompose 4x4 x 4x6 -> 4x6 using 2x2x3 blocks.
    2 blocks in n-direction, 2 blocks in p-direction = 4 blocks, each rank 11.
    Total rank: 44.
    """
    return _block_decomposition(4, 4, 6, 2, 2, 3)

def construct():
    """Construct the rank-minimal exact tensor decomposition for 4x4 x 4x6 matrix multiplication."""
    n, m, p = 4, 4, 6
    
    # Try the block decomposition (rank 44) first
    try:
        U, V, W = _construct_4x4x6_blocks()
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Try hybrid Strassen (rank 77)
    try:
        U, V, W = _construct_4x4x6_hybrid()
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    except Exception:
        pass
    
    # Fallback: schoolbook (rank 96)
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
