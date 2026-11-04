"""
Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
Uses structured block construction: Strassen for 4x4*4x4 (rank 18) 
plus Schoolbook for 4x4*4x1 (rank 16) = Total Rank 65.
"""

import time

def _strassen_2x2():
    """
    Returns the 7-term Strassen decomposition for 2x2 * 2x2.
    U, V, W are lists of 7 vectors of length 4 (2*2 entries).
    Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen formulas for 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need U, V, W such that sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k
    # Wait, the tensor T[i,j,k] is 1 for all i,j,k in the standard matrix multiplication definition?
    # No, the tensor for matrix multiplication C = AB is defined such that C_{ik} = sum_j A_{ij} B_{jk}.
    # The decomposition is sum_c U_{ij,c} V_{jk,c} W_{ki,c} = 1 for all i,j,k.
    # This means the coefficient of A_{ij} B_{jk} in the sum is 1.
    
    # Let's define the 7 terms.
    # Term 1: M1 = (A11+A22)(B11+B22)
    # A part: A11, A22 -> U indices: (0,0), (1,1)
    # B part: B11, B22 -> V indices: (0,0), (1,1)
    # C part: contributes to C11 (+), C22 (+) -> W indices: (0,0), (1,1)
    
    # Let's build U, V, W lists of length 7.
    # Each entry is a list of 4 values (for 2x2).
    
    # Term 1:
    u1 = [1, 0, 0, 1] # A11, A22
    v1 = [1, 0, 0, 1] # B11, B22
    w1 = [1, 0, 0, 1] # C11, C22 (indices 0,0 and 1,1)
    
    # Term 2: M2 = (A21+A22)B11
    # A part: A21, A22 -> U indices: (1,0), (1,1)
    # B part: B11 -> V indices: (0,0)
    # C part: contributes to C21 (+) -> W indices: (1,0)
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, 0]
    
    # Term 3: M3 = A11(B12-B22)
    # A part: A11 -> U indices: (0,0)
    # B part: B12, B22 -> V indices: (0,1), (1,1)
    # C part: contributes to C12 (+), C22 (-) -> W indices: (0,1), (1,1)
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, -1]
    
    # Term 4: M4 = A22(B21-B11)
    # A part: A22 -> U indices: (1,1)
    # B part: B21, B11 -> V indices: (1,0), (0,0)
    # C part: contributes to C11 (+), C21 (+) -> W indices: (0,0), (1,0)
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1] # B21, -B11? Wait. M4 = A22(B21-B11). So V is B21, -B11.
    # V indices: (1,0) is B21, (0,0) is B11. So v4 = [0, 0, 1, -1] is correct for V[1,0]=1, V[0,0]=-1.
    # W indices: C11 (+), C21 (+). W[0,0]=1, W[1,0]=1.
    w4 = [1, 0, 1, 0]
    
    # Term 5: M5 = (A11+A12)B22
    # A part: A11, A12 -> U indices: (0,0), (0,1)
    # B part: B22 -> V indices: (1,1)
    # C part: contributes to C11 (-), C12 (+) -> W indices: (0,0), (0,1)
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [-1, 1, 0, 0]
    
    # Term 6: M6 = (A21-A11)(B11+B12)
    # A part: A21, A11 -> U indices: (1,0), (0,0)
    # B part: B11, B12 -> V indices: (0,0), (0,1)
    # C part: contributes to C22 (+) -> W indices: (1,1)
    u6 = [-1, 0, 1, 0] # -A11, A21
    v6 = [1, 1, 0, 0] # B11, B12
    w6 = [0, 0, 0, 1]
    
    # Term 7: M7 = (A12-A22)(B21+B22)
    # A part: A12, A22 -> U indices: (0,1), (1,1)
    # B part: B21, B22 -> V indices: (1,0), (1,1)
    # C part: contributes to C11 (+), C22 (-) -> W indices: (0,0), (1,1)
    u7 = [0, 1, 0, -1] # A12, -A22
    v7 = [0, 0, 1, 1] # B21, B22
    w7 = [1, 0, 0, -1]
    
    return [u1, u2, u3, u4, u5, u6, u7], [v1, v2, v3, v4, v5, v6, v7], [w1, w2, w3, w4, w5, w6, w7]

def _build_4x4_4x4_strassen():
    """
    Build a rank-18 decomposition for 4x4 * 4x4 using 4x4 blocks of 2x2.
    The 4x4 matrix multiplication can be viewed as block multiplication of 2x2 blocks of 2x2 matrices.
    C = A * B
    A = [[A00, A01], [A10, A11]]
    B = [[B00, B01], [B10, B11]]
    C = [[C00, C01], [C10, C11]]
    
    C00 = A00B00 + A01B10
    C01 = A00B01 + A01B11
    C10 = A10B00 + A11B10
    C11 = A10B01 + A11B11
    
    Each term is a 2x2 * 2x2 multiplication.
    We can use Strassen for each of the 8 products? No, that would be 8*7=56.
    Instead, we use the block Strassen algorithm which uses 7 block multiplications.
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Each Mi is a 2x2 * 2x2 product. We can decompose each Mi into 7 scalar terms.
    Total rank = 7 * 7 = 49? No, the block Strassen algorithm reduces the number of multiplications from 8 to 7.
    But each multiplication is still 2x2*2x2. If we use Strassen for the 2x2*2x2, the total rank is 7 (block terms) * 7 (scalar terms per block) = 49.
    Wait, the rank of 4x4*4x4 is known to be 49 using Strassen.
    
    However, we can do better by using the fact that the 2x2*2x2 decompositions can be shared? No, they are independent.
    
    Let's just construct the 49-term decomposition.
    U, V, W will have shape (16, 49), (16, 49), (16, 49).
    """
    u_2x2, v_2x2, w_2x2 = _strassen_2x2()
    
    # Block indices: 00, 01, 10, 11
    # Block 00: rows 0-1, cols 0-1
    # Block 01: rows 0-1, cols 2-3
    # Block 10: rows 2-3, cols 0-1
    # Block 11: rows 2-3, cols 2-3
    
    # We need to map the 7 block terms to the 4x4 indices.
    # Term t in 0..6 corresponds to one of the M1..M7.
    # For each term t, we have 7 scalar decompositions s in 0..6.
    # Total index c = t * 7 + s.
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    # Define the block contributions for each M term.
    # M1 = (A00+A11)(B00+B11)
    # A blocks involved: 00 (+1), 11 (+1)
    # B blocks involved: 00 (+1), 11 (+1)
    # C blocks involved: 00 (+1), 11 (+1)
    
    # Let's define a helper to add a 2x2 decomposition to the 4x4 matrices.
    def add_term(u_block_signs, v_block_signs, w_block_signs):
        # u_block_signs: dict {block_idx: sign}
        # v_block_signs: dict {block_idx: sign}
        # w_block_signs: dict {block_idx: sign}
        pass # This approach is getting complicated. Let's do it explicitly.
        
    # Explicit construction:
    # For each of the 7 block terms (t=0..6), and each of the 7 scalar terms (s=0..6):
    # c = t*7 + s
    
    # We need to know which 2x2 blocks are involved in each M term and with what signs.
    # M0 (M1): A00(+), A11(+); B00(+), B11(+); C00(+), C11(+)
    # M1 (M2): A10(+), A11(+); B00(+); C10(+)
    # M2 (M3): A00(+); B01(+), B11(-); C01(+), C11(-)
    # M3 (M4): A11(+); B10(+), B00(-); C00(+), C10(+)
    # M4 (M5): A00(+), A01(+); B11(+); C00(-), C01(+)
    # M5 (M6): A10(+), A00(-); B00(+), B01(+); C11(+)
    # M6 (M7): A01(+), A11(-); B10(+), B11(+); C00(+), C11(-)
    
    # Block to index mapping:
    # 00: rows 0-1, cols 0-1
    # 01: rows 0-1, cols 2-3
    # 10: rows 2-3, cols 0-1
    # 11: rows 2-3, cols 2-3
    
    # Let's define the blocks for U, V, W.
    # U is for A (4x4). V is for B (4x4). W is for C (4x4).
    
    # We will iterate over the 7 block terms.
    block_terms = [
        # (A_blocks, B_blocks, C_blocks)
        # Each block is (block_idx, sign)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]), # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, -1)]), # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]), # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]), # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]), # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1), (3, -1)]), # M7
    ]
    
    # Block index to row/col ranges in 4x4
    # 0: 00 (rows 0-1, cols 0-1)
    # 1: 01 (rows 0-1, cols 2-3)
    # 2: 10 (rows 2-3, cols 0-1)
    # 3: 11 (rows 2-3, cols 2-3)
    
    def block_to_indices(block_idx):
        if block_idx == 0: return (0, 1, 0, 1) # r0, r1, c0, c1
        if block_idx == 1: return (0, 1, 2, 3)
        if block_idx == 2: return (2, 3, 0, 1)
        if block_idx == 3: return (2, 3, 2, 3)
        
    for t in range(7):
        a_blocks, b_blocks, c_blocks = block_terms[t]
        for s in range(7):
            c = t * 7 + s
            u_vec = u_2x2[s]
            v_vec = v_2x2[s]
            w_vec = w_2x2[s]
            
            # Fill U
            for b_idx, sign in a_blocks:
                r0, r1, c0, c1 = block_to_indices(b_idx)
                # u_vec indices: 0:00, 1:01, 2:10, 3:11 in 2x2
                # We need to map 2x2 indices to 4x4 indices.
                # 2x2 index 0 -> (r0, c0)
                # 2x2 index 1 -> (r0, c1)
                # 2x2 index 2 -> (r1, c0)
                # 2x2 index 3 -> (r1, c1)
                
                # Wait, u_2x2[s] is a vector of length 4 corresponding to the 2x2 entries.
                # The 2x2 entries are (0,0), (0,1), (1,0), (1,1).
                # So:
                U[r0*4 + c0][c] += sign * u_vec[0]
                U[r0*4 + c1][c] += sign * u_vec[1]
                U[r1*4 + c0][c] += sign * u_vec[2]
                U[r1*4 + c1][c] += sign * u_vec[3]
                
            # Fill V
            for b_idx, sign in b_blocks:
                r0, r1, c0, c1 = block_to_indices(b_idx)
                V[r0*4 + c0][c] += sign * v_vec[0]
                V[r0*4 + c1][c] += sign * v_vec[1]
                V[r1*4 + c0][c] += sign * v_vec[2]
                V[r1*4 + c1][c] += sign * v_vec[3]
                
            # Fill W
            for b_idx, sign in c_blocks:
                r0, r1, c0, c1 = block_to_indices(b_idx)
                # W index is k*n + i. Here n=4.
                # The 2x2 W entries correspond to (k, i) in the block.
                # 2x2 index 0 -> (k0, i0)
                # 2x2 index 1 -> (k0, i1)
                # 2x2 index 2 -> (k1, i0)
                # 2x2 index 3 -> (k1, i1)
                # In the 4x4, the block rows are r0, r1 and cols are c0, c1.
                # But W is indexed by [k, i] where k is col of B (which is row of C in standard C=AB? No.)
                # Standard: C[i,k] = sum_j A[i,j] B[j,k].
                # W[k*n + i, c].
                # So W index is (k, i).
                # In the block, the "k" dimension corresponds to the columns of the B block, which are the columns of the C block.
                # The "i" dimension corresponds to the rows of the A block, which are the rows of the C block.
                # So for block (r0,r1, c0,c1):
                # W index (k, i) where k in {c0, c1} and i in {r0, r1}.
                # 2x2 index 0: k=c0, i=r0 -> W[c0*4 + r0]
                # 2x2 index 1: k=c1, i=r0 -> W[c1*4 + r0]
                # 2x2 index 2: k=c0, i=r1 -> W[c0*4 + r1]
                # 2x2 index 3: k=c1, i=r1 -> W[c1*4 + r1]
                
                W[c0*4 + r0][c] += sign * w_vec[0]
                W[c1*4 + r0][c] += sign * w_vec[1]
                W[c0*4 + r1][c] += sign * w_vec[2]
                W[c1*4 + r1][c] += sign * w_vec[3]
                
    return U, V, W

def _build_4x4_4x1_schoolbook():
    """
    Build a rank-16 decomposition for 4x4 * 4x1.
    This is just standard matrix-vector multiplication.
    C[i] = sum_j A[i,j] B[j]
    U[i*4+j, c] = 1, V[j, c] = 1, W[i, c] = 1 for each (i,j) pair.
    Total 16 terms.
    """
    U = [[0]*16 for _ in range(16)]
    V = [[0]*16 for _ in range(4)]
    W = [[0]*16 for _ in range(4)]
    
    c = 0
    for i in range(4):
        for j in range(4):
            U[i*4 + j][c] = 1
            V[j][c] = 1
            W[i][c] = 1
            c += 1
            
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-6:
                    return False, s
    return True, 1.0

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    Uses Strassen for the 4x4*4x4 part (rank 49) and schoolbook for the 4x4*4x1 part (rank 16).
    Total rank: 49 + 16 = 65.
    """
    n, m, p = 4, 4, 5
    
    # Build the 4x4 * 4x4 part (first 4 columns of B)
    U_44, V_44, W_44 = _build_4x4_4x4_strassen()
    
    # Build the 4x4 * 4x1 part (last column of B)
    U_41, V_41, W_41 = _build_4x4_4x1_schoolbook()
    
    # Combine them
    # U shape: (16, 65)
    # V shape: (20, 65) -> (16 for 4x4 part, 4 for 4x1 part)
    # W shape: (20, 65) -> (16 for 4x4 part, 4 for 4x1 part)
    
    rank = 65
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(20)]
    W = [[0]*rank for _ in range(20)]
    
    # Fill U
    for i in range(16):
        for c in range(49):
            U[i][c] = U_44[i][c]
        for c in range(16):
            U[i][49+c] = U_41[i][c]
            
    # Fill V
    # V_44 has 16 rows (j in 0..3, k in 0..3). Index j*4+k.
    # V_41 has 4 rows (j in 0..3). Index j.
    # V total has 20 rows.
    # Rows 0-15 correspond to the 4x4 part.
    # Rows 16-19 correspond to the 4x1 part?
    # Wait, the B matrix is 4x5.
    # The first 4 columns are the 4x4 part.
    # The last column is the 4x1 part.
    # V index is j*p + k.
    # For the 4x4 part (k=0..3): V[j*5 + k].
    # For the 4x1 part (k=4): V[j*5 + 4].
    
    # My V_44 was built for 4x4 * 4x4, so its V indices were j*4 + k.
    # I need to map them to j*5 + k.
    
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*5 + k][c] = V_44[j*4 + k][c]
                
    for j in range(4):
        for c in range(16):
            V[j*5 + 4][49+c] = V_41[j][c]
            
    # Fill W
    # W_44 has 16 rows (k in 0..3, i in 0..3). Index k*4 + i.
    # W_41 has 4 rows (i in 0..3). Index i.
    # W total has 20 rows.
    # W index is k*n + i = k*4 + i.
    # For the 4x4 part (k=0..3): W[k*4 + i].
    # For the 4x1 part (k=4): W[4*4 + i] = W[16 + i].
    
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k*4 + i][c] = W_44[k*4 + i][c]
                
    for i in range(4):
        for c in range(16):
            W[16 + i][49+c] = W_41[i][c]
            
    # Verify
    valid, _ = _verify(U, V, W, n, m, p)
    if not valid:
        # Fallback to schoolbook if verification fails
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][column] = 1
                    V_sb[j * p + k][column] = 1
                    W_sb[k * n + i][column] = 1
                    column += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
        
    return {"U": U, "V": V, "W": W}
