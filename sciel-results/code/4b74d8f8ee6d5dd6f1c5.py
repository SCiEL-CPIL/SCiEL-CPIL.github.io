"""Matrix multiplication 4x4x6 rank-minimal decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 6]

def _schoolbook():
    """Return the trivial schoolbook decomposition (rank 96)."""
    n, m, p = DIMENSIONS
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
    """
    Return the Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    U has shape (4, 7), V has shape (4, 7), W has shape (4, 7).
    Indexing: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    # Standard Strassen with the following 7 products:
    # M1 = (a11 + a21) * (b11 + b12)
    # M2 = (a11 + a21) * (b21 + b22)
    # M3 = (a12 + a22) * (b11 + b21)  -- wait, let me use the standard formulation
    #
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = [[c11, c12], [c21, c22]]
    # c11 = ae + bg, c12 = af + bh, c21 = ce + dg, c22 = cf + dh
    #
    # Strassen's 7 multiplications:
    # M1 = (a + d)(e + h)
    # M2 = (c + d)(e + f)
    # M3 = (a - b)(g + h)
    # M4 = (a + d)(f - h)
    # M5 = (a + b)(e - g)
    # M6 = (c - a)(f + h)
    # M7 = (b - d)(g + e)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M3 + M6 - M2
    #
    # Let me verify C11 = ae + bg:
    # M1 + M4 - M5 + M7
    # = (a+d)(e+h) + (a+d)(f-h) - (a+b)(e-g) + (b-d)(g+e)
    # = (ae+ah+de+dh) + (af-ah+df-dh) - (ae-ag-be+bg) + (bg+be-dg-de)
    # = ae + ah + de + dh + af - ah + df - dh - ae + ag + be - bg + bg + be - dg - de
    # = ae + af + ag + 2be - dg + df - bg + bg  (wait, let me redo)
    # 
    # Let me just hardcode a known-correct Strassen decomposition by checking
    # each term carefully.
    
    # I'll use this well-known formulation:
    # p1 = (a11 + a22)(b11 + b22)
    # p2 = (a21 + a22)(b11 + b12)
    # p3 = (a11 - a22)(b21 - b22)
    # p4 = a11*b11
    # p5 = a22*b22
    # p6 = (a12 - a22)(b21 + b22)
    # p7 = (a11 + a12)(b11 - b21)
    #
    # C11 = p1 + p3 - p5 + p7
    # C12 = p4 + p5 - p1 + p6
    # C21 = p4 + p2 - p3 - p7
    # C22 = p1 + p3 - p4 + p5
    
    # Let me verify C11 = a11*b11 + a12*b21:
    # p1 + p3 - p5 + p7
    # = (a11+a22)(b11+b22) + (a11-a22)(b21-b22) - a22*b22 + (a11+a12)(b11-b21)
    # = a11b11 + a11b22 + a22b11 + a22b22 + a11b21 - a11b22 - a22b21 + a22b22 - a22b22 + a11b11 - a11b21 + a12b11 - a12b21
    # = 2a11b11 + a22b11 - a22b21 + a22b22 + a12b11 - a12b21
    # That's not right either. Let me just use a verified implementation.
    
    # Actually, let me just use the schoolbook for 2x2x2 (rank 4) to be safe.
    # This gives total rank 12 * 4 = 48, which is still better than 96.
    
    U = np.zeros((4, 4))
    V = np.zeros((4, 4))
    W = np.zeros((4, 4))
    # Schoolbook: C[i,k] = sum_j A[i,j] B[j,k]
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # For each (i,j,k) triple, one factor
    col = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i * 2 + j, col] = 1
                V[j * 2 + k, col] = 1
                W[k * 2 + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    # Check rank consistency
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] == W[k*n+i, r]
    # Wait, that's not right. The tensor decomposition says:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] should be 1 if this is a valid multiplication tensor entry.
    # Actually, the matrix multiplication tensor M[i,j,k] = 1 always (it's the structure constant).
    # The decomposition satisfies: M[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _block_decomposition_2x2x2():
    """
    Decompose 4x4x6 as 12 independent 2x2x2 subproblems.
    4x4x6 = (2x2) x (2x2) x (2x3), so 2*2*2*3/4... wait.
    Actually, 4x4x6 can be partitioned into blocks:
    A is 4x4, B is 4x6, C = AB is 4x6.
    If we partition A into 2x2 blocks of size 2x2, B into 2x3 blocks of size 2x2,
    we get C in 2x3 blocks of size 2x2.
    Each C_block[i,j] = sum_k A_block[i,k] * B_block[k,j]
    That's 2*2*3 = 12 subproblems, but each involves summing over 2 block columns,
    so each C_block[i,j] = A_block[i,0]*B_block[0,j] + A_block[i,1]*B_block[1,j]
    Each of these is a 2x2 * 2x2 = 2x2 multiplication.
    So total we have 2*2*3*2 = 24 individual 2x2x2 multiplications.
    Using rank-4 schoolbook for each: 24 * 4 = 96 (same as before).
    Using Strassen rank-7: 24 * 7 = 168 (worse!).
    
    Wait, that's not right. Let me reconsider.
    Actually, the block structure gives us 2*3 = 6 block columns in B, and 2 block rows in A.
    C_block[i,j] = A_block[i,0] @ B_block[0,j] + A_block[i,1] @ B_block[1,j]
    So we need 2 * 2 * 3 = 12 matrix products of 2x2 matrices.
    If we use rank-4 for each: 12 * 4 = 48.
    If we use Strassen rank-7: 12 * 7 = 84.
    
    So schoolbook for the sub-blocks gives rank 48, which is better than 96!
    """
    n, m, p = DIMENSIONS
    # We have a 2x2 block grid for A, a 2x3 block grid for B
    # Each block is 2x2
    # Number of 2x2x2 subproblems: 2 (A rows) * 2 (A cols = B rows) * 3 (B cols) = 12
    # But each C_block[i,j] sums over 2 block columns, so 2 * 2 * 3 * 2 = 24? No.
    # A is 4x4, partitioned into 2x2 blocks of size 2x2: 2 rows, 2 cols
    # B is 4x6, partitioned into 2x3 blocks of size 2x2: 2 rows, 3 cols
    # C = AB is 4x6, partitioned into 2x3 blocks of size 2x2: 2 rows, 3 cols
    # C_block[i,j] = sum_{k=0}^{1} A_block[i,k] @ B_block[k,j]
    # So for each (i,j) pair, we have 2 matrix multiplications (k=0 and k=1)
    # Total: 2 * 3 * 2 = 12 matrix multiplications of 2x2 x 2x2 -> 2x2
    # Each uses rank 4 (schoolbook): total rank = 12 * 4 = 48
    
    rank = 48
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for bi in range(2):  # A block row
        for bk in range(2):  # A block col / B block row
            for bj in range(3):  # B block col
                for i in range(2):  # within A block row
                    for j in range(2):  # within A block col / B block row
                        for k in range(2):  # within B block col
                            # Global indices
                            gi = bi * 2 + i
                            gj = bk * 2 + j
                            gk = bj * 2 + k
                            U[gi * m + gj, col] = 1
                            V[gj * p + gk, col] = 1
                            W[gk * n + gi, col] = 1
                            col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    """Construct the matrix multiplication tensor decomposition for 4x4x6."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # First, try the block decomposition with schoolbook sub-blocks (rank 48)
    result = _block_decomposition_2x2x2()
    
    # Verify
    n, m, p = DIMENSIONS
    if _verify(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback to schoolbook (rank 96)
    return _schoolbook()

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
