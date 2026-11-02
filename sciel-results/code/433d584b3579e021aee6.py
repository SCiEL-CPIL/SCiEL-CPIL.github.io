"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W form a valid decomposition."""
    if not U or not V or not W:
        return False
    rank = len(U[0])
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = 0
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(val - 1) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns U, V, W as lists of lists where:
    U has shape (4, 7) for A indices (i*2+k)
    V has shape (4, 7) for B indices (k*2+j)  
    W has shape (4, 7) for C indices (j*2+i)
    
    Strassen's 7 multiplications:
    M1 = (A11+A22)*(B11+B22)
    M2 = (A21+A22)*B11
    M3 = A11*(B12-B22)
    M4 = A22*(B21-B11)
    M5 = (A11+A12)*B22
    M6 = (A21-A11)*(B11+B12)
    M7 = (A12-A22)*(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # U[i*2+k, c] for A[i,k]
    # V[k*2+j, c] for B[k,j]
    # W[j*2+i, c] for C[i,j]
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)*(B11+B22)
    # A11 is U[0,0], A22 is U[3,0] -> U[0,0]=1, U[3,0]=1
    # B11 is V[0,0], B22 is V[3,0] -> V[0,0]=1, V[3,0]=1
    U[0][0] = 1  # A11
    U[3][0] = 1  # A22
    V[0][0] = 1  # B11
    V[3][0] = 1  # B22
    # C11 gets +M1, C22 gets +M1
    W[0][0] = 1  # C11 (j=0,i=0)
    W[3][0] = 1  # C22 (j=1,i=1)
    
    # M2 = (A21+A22)*B11
    U[2][1] = 1  # A21 (i=1,k=0)
    U[3][1] = 1  # A22 (i=1,k=1)
    V[0][1] = 1  # B11
    # C21 gets +M2
    W[2][1] = 1  # C21 (j=0,i=1)
    
    # M3 = A11*(B12-B22)
    U[0][2] = 1  # A11
    V[1][2] = 1  # B12
    V[3][2] = -1 # B22
    # C12 gets +M3
    W[1][2] = 1  # C12 (j=1,i=0)
    
    # M4 = A22*(B21-B11)
    U[3][3] = 1  # A22
    V[2][3] = 1  # B21
    V[0][3] = -1 # B11
    # C11 gets +M4, C21 gets +M4
    W[0][3] = 1  # C11
    W[2][3] = 1  # C21
    
    # M5 = (A11+A12)*B22
    U[0][4] = 1  # A11
    U[1][4] = 1  # A12 (i=0,k=1)
    V[3][4] = 1  # B22
    # C11 gets -M5, C12 gets +M5
    W[0][4] = -1 # C11
    W[1][4] = 1  # C12
    
    # M6 = (A21-A11)*(B11+B12)
    U[2][5] = 1  # A21
    U[0][5] = -1 # A11
    V[0][5] = 1  # B11
    V[1][5] = 1  # B12
    # C22 gets +M6
    W[3][5] = 1  # C22
    
    # M7 = (A12-A22)*(B21+B22)
    U[1][6] = 1  # A12
    U[3][6] = -1 # A22
    V[2][6] = 1  # B21
    V[3][6] = 1  # B22
    # C11 gets +M7, C22 gets +M7
    W[0][6] = 1  # C11
    W[3][6] = 1  # C22
    
    return U, V, W

def _strassen_4x4x4():
    """Return rank-49 decomposition for 4x4x4 using Strassen recursively.
    
    Split 4x4 into four 2x2 blocks. Apply Strassen to get 7 sub-problems
    of size 2x2x2, each with rank 7, total rank 49.
    
    A = [[A11, A12], [A21, A22]] where each is 2x2
    B = [[B11, B12], [B21, B22]] where each is 2x2
    
    M1 = (A11+A22)*(B11+B22) -> 2x2x2, rank 7
    M2 = (A21+A22)*B11 -> 2x2x2, rank 7
    M3 = A11*(B12-B22) -> 2x2x2, rank 7
    M4 = A22*(B21-B11) -> 2x2x2, rank 7
    M5 = (A11+A12)*B22 -> 2x2x2, rank 7
    M6 = (A21-A11)*(B11+B12) -> 2x2x2, rank 7
    M7 = (A12-A22)*(B21+B22) -> 2x2x2, rank 7
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n = m = p = 4
    total_rank = 49
    
    U = [[0]*total_rank for _ in range(n*m)]  # 16 x 49
    V = [[0]*total_rank for _ in range(m*p)]  # 16 x 49
    W = [[0]*total_rank for _ in range(p*n)]  # 16 x 49
    
    # Get base Strassen 2x2x2
    U2, V2, W2 = _strassen_2x2x2()
    
    # Block indices for 4x4 split into 2x2 blocks:
    # A11: rows 0-1, cols 0-1 -> A indices 0,1,2,3
    # A12: rows 0-1, cols 2-3 -> A indices 0,1,4,5
    # A21: rows 2-3, cols 0-1 -> A indices 8,9,10,11
    # A22: rows 2-3, cols 2-3 -> A indices 8,9,12,13
    
    # B11: rows 0-1, cols 0-1 -> B indices 0,1,4,5
    # B12: rows 0-1, cols 2-3 -> B indices 0,1,6,7
    # B21: rows 2-3, cols 0-1 -> B indices 8,9,12,13
    # B22: rows 2-3, cols 2-3 -> B indices 8,9,14,15
    
    # C11: rows 0-1, cols 0-1 -> C indices 0,4,1,5 (j*n+i: j=0,i=0->0; j=0,i=1->1; j=1,i=0->4; j=1,i=1->5)
    # Wait, C index is j*n+i. For 4x4:
    # C[i,j] has index j*4+i
    # C11 block: i in {0,1}, j in {0,1} -> indices 0,1,4,5
    # C12 block: i in {0,1}, j in {2,3} -> indices 8,9,12,13
    # C21 block: i in {2,3}, j in {0,1} -> indices 2,3,6,7
    # C22 block: i in {2,3}, j in {2,3} -> indices 10,11,14,15
    
    def add_block(base_U, base_V, base_W, a_offsets, b_offsets, c_offsets, sign=1, start_col=0):
        """Add a scaled Strassen block to the decomposition."""
        for i in range(4):
            for k in range(4):
                u_idx = a_offsets[i*2+k]
                for c in range(7):
                    U[u_idx][start_col + c] += sign * base_U[i][c]
        for k in range(4):
            for j in range(4):
                v_idx = b_offsets[k*2+j]
                for c in range(7):
                    V[v_idx][start_col + c] += sign * base_V[k][c]
        for j in range(4):
            for i in range(4):
                w_idx = c_offsets[j*2+i]
                for c in range(7):
                    W[w_idx][start_col + c] += sign * base_W[j][c]
    
    # A block offsets (for 2x2 blocks within 4x4):
    # A11: (i,k) in {(0,0),(0,1),(1,0),(1,1)} -> flat: 0,1,2,3
    # A12: (i,k) in {(0,2),(0,3),(1,2),(1,3)} -> flat: 0*4+2=2? No wait.
    # A[i,k] flat index = i*m+k = i*4+k
    # A11: i in {0,1}, k in {0,1} -> 0,1,4,5
    # A12: i in {0,1}, k in {2,3} -> 2,3,6,7
    # A21: i in {2,3}, k in {0,1} -> 8,9,12,13
    # A22: i in {2,3}, k in {2,3} -> 10,11,14,15
    
    A11_off = [0, 1, 4, 5]
    A12_off = [2, 3, 6, 7]
    A21_off = [8, 9, 12, 13]
    A22_off = [10, 11, 14, 15]
    
    # B[k,j] flat index = k*p+j = k*4+j
    # B11: k in {0,1}, j in {0,1} -> 0,1,4,5
    # B12: k in {0,1}, j in {2,3} -> 2,3,6,7
    # B21: k in {2,3}, j in {0,1} -> 8,9,12,13
    # B22: k in {2,3}, j in {2,3} -> 10,11,14,15
    
    B11_off = [0, 1, 4, 5]
    B12_off = [2, 3, 6, 7]
    B21_off = [8, 9, 12, 13]
    B22_off = [10, 11, 14, 15]
    
    # C[i,j] flat index = j*n+i = j*4+i
    # C11: i in {0,1}, j in {0,1} -> j=0,i=0->0; j=0,i=1->1; j=1,i=0->4; j=1,i=1->5
    # C12: i in {0,1}, j in {2,3} -> j=2,i=0->8; j=2,i=1->9; j=3,i=0->12; j=3,i=1->13
    # C21: i in {2,3}, j in {0,1} -> j=0,i=2->2; j=0,i=3->3; j=1,i=2->6; j=1,i=3->7
    # C22: i in {2,3}, j in {2,3} -> j=2,i=2->10; j=2,i=3->11; j=3,i=2->14; j=3,i=3->15
    
    C11_off = [0, 1, 4, 5]
    C12_off = [8, 9, 12, 13]
    C21_off = [2, 3, 6, 7]
    C22_off = [10, 11, 14, 15]
    
    # M1 = (A11+A22)*(B11+B22)
    # This is a 2x2x2 problem where the "A" block is A11+A22 and "B" block is B11+B22
    # The U for this sub-problem gets contributions from both A11 and A22 positions
    # We need to map the 2x2 A indices to the 4x4 A indices
    
    # For M1: A_sub[i,k] = A11[i,k] + A22[i,k]
    # So U_sub[i*2+k, c] should add to both A11[i,k] and A22[i,k] positions
    # A11[i,k] is at flat index: i*4+k for i,k in {0,1}
    # A22[i,k] is at flat index: (i+2)*4+(k+2) for i,k in {0,1}
    
    def add_strassen_term(a_blocks, b_blocks, c_blocks, signs, start_col):
        """Add one Strassen term.
        a_blocks: list of (offsets, sign) for A contributions
        b_blocks: list of (offsets, sign) for B contributions  
        c_blocks: list of (offsets, sign) for C contributions
        """
        for c in range(7):
            for a_offs, a_sign in a_blocks:
                for idx in range(4):
                    U[a_offs[idx]][start_col + c] += a_sign * U2[idx][c]
            for b_offs, b_sign in b_blocks:
                for idx in range(4):
                    V[b_offs[idx]][start_col + c] += b_sign * V2[idx][c]
            for c_offs, c_sign in c_blocks:
                for idx in range(4):
                    W[c_offs[idx]][start_col + c] += c_sign * W2[idx][c]
    
    col = 0
    # M1 = (A11+A22)*(B11+B22)
    add_strassen_term(
        [(A11_off, 1), (A22_off, 1)],
        [(B11_off, 1), (B22_off, 1)],
        [(C11_off, 1), (C22_off, 1)],  # C11 += M1, C22 += M1
        col
    )
    col += 7
    
    # M2 = (A21+A22)*B11
    add_strassen_term(
        [(A21_off, 1), (A22_off, 1)],
        [(B11_off, 1)],
        [(C21_off, 1)],  # C21 += M2
        col
    )
    col += 7
    
    # M3 = A11*(B12-B22)
    add_strassen_term(
        [(A11_off, 1)],
        [(B12_off, 1), (B22_off, -1)],
        [(C12_off, 1)],  # C12 += M3
        col
    )
    col += 7
    
    # M4 = A22*(B21-B11)
    add_strassen_term(
        [(A22_off, 1)],
        [(B21_off, 1), (B11_off, -1)],
        [(C11_off, 1), (C21_off, 1)],  # C11 += M4, C21 += M4
        col
    )
    col += 7
    
    # M5 = (A11+A12)*B22
    add_strassen_term(
        [(A11_off, 1), (A12_off, 1)],
        [(B22_off, 1)],
        [(C11_off, -1), (C12_off, 1)],  # C11 -= M5, C12 += M5
        col
    )
    col += 7
    
    # M6 = (A21-A11)*(B11+B12)
    add_strassen_term(
        [(A21_off, 1), (A11_off, -1)],
        [(B11_off, 1), (B12_off, 1)],
        [(C22_off, 1)],  # C22 += M6
        col
    )
    col += 7
    
    # M7 = (A12-A22)*(B21+B22)
    add_strassen_term(
        [(A12_off, 1), (A22_off, -1)],
        [(B21_off, 1), (B22_off, 1)],
        [(C11_off, 1), (C22_off, 1)],  # C11 += M7, C22 += M7
        col
    )
    col += 7
    
    return {"U": U, "V": V, "W": W}

def _combine_decompositions(decs, n, m, p, col_offsets):
    """Combine multiple decompositions by concatenating columns.
    
    decs: list of dicts with 'U', 'V', 'W'
    col_offsets: list of starting column indices for each decomposition
    """
    total_rank = sum(len(d['U'][0]) for d in decs)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for d, offset in zip(decs, col_offsets):
        r = len(d['U'][0])
        for i in range(n * m):
            for c in range(r):
                U[i][offset + c] = d['U'][i][c]
        for k in range(m * p):
            for c in range(r):
                V[k][offset + c] = d['V'][k][c]
        for j in range(p * n):
            for c in range(r):
                W[j][offset + c] = d['W'][j][c]
    
    return {"U": U, "V": V, "W": W}

def _schoolbook_partial(n, m, p, j_start, j_end):
    """Schoolbook decomposition for columns j_start to j_end-1 of B."""
    num_cols = j_end - j_start
    rank = n * m * num_cols
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(j_start, j_end):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Strategy: Split B into B1 (4x4) and B2 (4x5)
    # Use Strassen for M_{4,4,4} (rank 49) and schoolbook for M_{4,4,5} (rank 80)
    # Total rank: 49 + 80 = 129
    # But wait, we can do better by splitting differently.
    
    # Actually, let's try: B = [B1 | B2] where B1 is 4x4, B2 is 4x5
    # C = [C1 | C2] where C1 = A*B1, C2 = A*B2
    # rank(C1) <= rank(M_{4,4,4}) = 49
    # rank(C2) <= rank(M_{4,4,5}) = 4*4*5 = 80 (schoolbook)
    # Total: 129
    
    # Alternative: B = [B1 | B2 | B3] where each is 4x3
    # rank(M_{4,4,3}) = 48 (schoolbook)
    # Total: 3*48 = 144 (same as full schoolbook)
    
    # Let's go with the 4+5 split: rank 49 + 80 = 129
    
    # Build M_{4,4,4} with Strassen
    dec1 = _strassen_4x4x4()
    r1 = len(dec1['U'][0])  # should be 49
    
    # Build M_{4,4,5} with schoolbook for columns 4-8
    dec2 = _schoolbook_partial(n, m, p, 4, 9)
    r2 = len(dec2['U'][0])  # should be 80
    
    # Combine
    result = _combine_decompositions([dec1, dec2], n, m, p, [0, r1])
    
    # Verify
    if _verify(result['U'], result['V'], result['W'], n, m, p):
        return result
    else:
        # Fallback to schoolbook
        return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
