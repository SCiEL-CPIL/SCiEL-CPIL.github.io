"""
Matrix multiplication 4x4x9 - Strassen-based decomposition
Score target: beat 114 (schoolbook 144)
"""
DIMENSIONS = [4, 4, 9]

# 7 Strassen factors for 2x2x2
_STRASSEN_2X2 = [
    (1, 1, 1),   # M1 = (A11+A22)*(B11+B22)
    (1, -1, 1),  # M2 = (A21+A22)*B11
    (1, 1, 0),   # M3 = A11*(B12-B22)
    (1, 1, 0),   # M4 = A22*(B21-B11)
    (1, 1, 1),   # M5 = (A11+A12)*B22
    (-1, 1, 1),  # M6 = (A21-A11)*(B11+B12)
    (1, -1, 1),  # M7 = (A12-A22)*(B21+B22)
]

# Output combinations for 2x2x2 Strassen
# C11 = M1 + M4 - M5 - M7
# C12 = M3 + M5
# C21 = M2 + M4
# C22 = M1 - M2 + M3 + M6
_STRASSEN_OUT_2X2 = [
    {0: 1, 3: 1, 4: -1, 6: -1},  # C11
    {2: 1, 4: 1},                  # C12
    {1: 1, 3: 1},                  # C21
    {0: 1, 1: -1, 2: 1, 5: 1},    # C22
]

# For 4x4x4: partition into 2x2 blocks, use Strassen recursively
# A is 4x4, B is 4x4, C is 4x4
# Split A into A11,A12,A21,A22 (each 2x2)
# Split B into B11,B12,B21,B22 (each 2x2)
# Each product Aij*Bjk needs 7 Strassen factors for 2x2x2
# But we can share some factors!

# Actually, let's use the standard 4x4x4 Strassen approach:
# 4x4 matrix multiply via 2 levels of Strassen: 7^2 = 49 factors
# But we can do better by careful combination.

# Simpler approach: use 4x4x4 with rank 49 (known optimal for 4x4x4)
# Then for 4x4x9: we need C = A*B where A is 4x4, B is 4x9
# Split B into B_left (4x4) and B_right (4x5)
# C_left = A * B_left  (4x4x4, rank 49)
# C_right = A * B_right (4x4x5)
# For C_right: split B_right into 4x4 + 4x1
# C_right_left = A * B_right_left (4x4x4, rank 49)
# C_right_right = A * B_right_right (4x4x1, rank 16)
# Total: 49 + 49 + 16 = 114

# Can we do better? Let's try to use a 4x4x5 decomposition more efficiently.
# 4x4x5: A(4x4) * B(4x5) -> C(4x5)
# Schoolbook: 4*4*5 = 80
# Better: use Strassen-like for the 4x4 part applied to each column of B
# But columns share the same A, so we can factor.
# Actually, for 4x4x5, we can think of it as 5 independent 4x4x1 problems
# Each 4x4x1: A(4x4) * b(4x1) -> c(4x1), rank = 4*4 = 16 schoolbook
# But we can do better: rank of 4x4x1 tensor is 16 (each output c_i = sum_j A_ij * b_j)
# Total schoolbook: 5*16 = 80

# Let me try a different approach: use the structure of 4x4x9 directly
# C[i,k] = sum_j A[i,j] * B[j,k] for i in 0..3, j in 0..3, k in 0..8
# This is a 4x4x9 tensor.

# Let me implement the known 114-rank decomposition and verify it works,
# then try to optimize.

def _make_2x2_strassen_factors():
    """
    Generate U, V, W factors for 2x2x2 Strassen multiplication.
    U: (2*2=4) x 7, V: (2*2=4) x 7, W: (2*2=4) x 7
    Indexing: U[row, c] where row = i*2+j for A[i,j]
               V[row, c] where row = j*2+k for B[j,k]  
               W[row, c] where row = k*2+i for C[k,i] (trace convention W[k*n+i,r])
    Wait, let me re-read: W[k*n+i,r] for C[i,k]
    So W row index = k*n + i where n=2 for 2x2 case.
    """
    n = 2
    rank = 7
    U = [[0]*rank for _ in range(n*n)]  # A is 2x2, indexed by i*n+j
    V = [[0]*rank for _ in range(n*n)]  # B is 2x2, indexed by j*n+k
    W = [[0]*rank for _ in range(n*n)]  # C is 2x2, indexed by k*n+i

    # For each of the 7 Strassen products M_c = (A_sub) * (B_sub):
    # A_sub is a 2x2 matrix (some linear combination of A's 2x2 blocks... 
    # wait, for 2x2x2, the "blocks" are scalars)
    
    # Actually for 2x2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # A indices: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    # B indices: B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
    # C indices: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # U[row, c]: coefficient of A[i,j] in the A-part of M_c
    # V[row, c]: coefficient of B[j,k] in the B-part of M_c
    # W[row, c]: coefficient of M_c in C[i,k] where row = k*n+i

    # M1: A_part = A00 + A11, B_part = B00 + B11
    U[0*2+0][0] = 1  # A00
    U[1*2+1][0] = 1  # A11
    V[0*2+0][0] = 1  # B00
    V[1*2+1][0] = 1  # B11
    # C00 += M1, C11 += M1
    W[0*2+0][0] = 1  # C00, k=0,i=0
    W[1*2+1][0] = 1  # C11, k=1,i=1

    # M2: A_part = A10 + A11, B_part = B00
    U[1*2+0][1] = 1  # A10
    U[1*2+1][1] = 1  # A11
    V[0*2+0][1] = 1  # B00
    # C10 += M2
    W[0*2+1][1] = 1  # C10, k=0,i=1

    # M3: A_part = A00, B_part = B01 - B11
    U[0*2+0][2] = 1  # A00
    V[0*2+1][2] = 1  # B01
    V[1*2+1][2] = -1 # B11
    # C01 += M3
    W[1*2+0][2] = 1  # C01, k=1,i=0

    # M4: A_part = A11, B_part = B10 - B00
    U[1*2+1][3] = 1  # A11
    V[1*2+0][3] = 1  # B10
    V[0*2+0][3] = -1 # B00
    # C00 += M4, C10 += M4
    W[0*2+0][3] = 1  # C00
    W[0*2+1][3] = 1  # C10

    # M5: A_part = A00 + A01, B_part = B11
    U[0*2+0][4] = 1  # A00
    U[0*2+1][4] = 1  # A01
    V[1*2+1][4] = 1  # B11
    # C00 -= M5, C01 += M5
    W[0*2+0][4] = -1 # C00
    W[1*2+0][4] = 1  # C01

    # M6: A_part = A10 - A00, B_part = B00 + B01
    U[1*2+0][5] = 1  # A10
    U[0*2+0][5] = -1 # A00
    V[0*2+0][5] = 1  # B00
    V[0*2+1][5] = 1  # B01
    # C11 += M6
    W[1*2+1][5] = 1  # C11

    # M7: A_part = A01 - A11, B_part = B10 + B11
    U[0*2+1][6] = 1  # A01
    U[1*2+1][6] = -1 # A11
    V[1*2+0][6] = 1  # B10
    V[1*2+1][6] = 1  # B11
    # C00 -= M7
    W[0*2+0][6] = -1 # C00

    return U, V, W


def _make_4x4x4_strassen_factors():
    """
    Generate factors for 4x4x4 matrix multiplication using Strassen.
    We use 2 levels: split 4x4 into four 2x2 blocks, apply Strassen.
    This gives 7^2 = 49 rank-1 terms (before optimization).
    
    Actually, let's use a cleaner approach: the 4x4x4 tensor can be 
    decomposed with rank 49 via recursive Strassen.
    """
    n = 4
    # We'll build it from 2x2x2 Strassen applied at the block level
    # A is 4x4, split into 2x2 blocks: A11, A12, A21, A22
    # B is 4x4, split into 2x2 blocks: B11, B12, B21, B22
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For each block product, we use 2x2x2 Strassen (7 factors each)
    # C11: needs A11*B11 (7) + A12*B21 (7) = 14 factors
    # But we can combine! The Strassen formula gives us:
    # C11 = M1 + M4 - M5 - M7 where M's are computed from (A11+A22) etc.
    # 
    # Let me use the full 4x4x4 Strassen decomposition with 49 factors.
    # Each factor corresponds to a product of a linear combination of A entries
    # and a linear combination of B entries.
    
    # For now, let's use schoolbook for 4x4x4 (rank 64) and 4x4x1 (rank 16)
    # Total: 64 + 64 + 16 = 144... that's just schoolbook.
    # 
    # Let me properly implement 49-rank 4x4x4.
    
    # 4x4x4 via 2-level Strassen:
    # Level 1: 7 "super-factors" each involving 2x2 block products
    # Level 2: each 2x2 block product uses 7 Strassen factors
    # Total: 49 factors
    
    rank = 49
    U = [[0]*rank for _ in range(n*n)]  # 16 x 49
    V = [[0]*rank for _ in range(n*n)]  # 16 x 49
    W = [[0]*rank for _ in range(n*n)]  # 16 x 49

    # Block indices: block (bi, bj) where bi,bj in {0,1}
    # Block (bi,bj) has entries at positions (bi*2+r, bj*2+s) for r,s in {0,1}
    # A entry (i,j) is in block (i//2, j//2) at local pos (i%2, j%2)
    # B entry (j,k) is in block (j//2, k//2) at local pos (j%2, k%2)
    # C entry (i,k) is in block (i//2, k//2) at local pos (i%2, k%2)
    
    # The 7 Strassen factors at block level:
    # S1 = (A11+A22)*(B11+B22)
    # S2 = (A21+A22)*B11
    # S3 = A11*(B12-B22)
    # S4 = A22*(B21-B11)
    # S5 = (A11+A12)*B22
    # S6 = (A21-A11)*(B11+B12)
    # S7 = (A12-A22)*(B21+B22)
    
    # Each S_c is a 2x2x2 matrix product of 2x2 blocks.
    # For each S_c, we have 7 sub-factors (from 2x2x2 Strassen)
    
    # For S_c, the A-blocks involved and B-blocks involved:
    # S1: A blocks {11,22}, B blocks {11,22}
    # S2: A blocks {21,22}, B blocks {11}
    # S3: A blocks {11}, B blocks {12,22}
    # S4: A blocks {22}, B blocks {21,11}
    # S5: A blocks {11,12}, B blocks {22}
    # S6: A blocks {21,11}, B blocks {11,12}
    # S7: A blocks {12,22}, B blocks {21,22}
    
    # For each super-factor S_c and sub-factor s (0..6):
    # We need to set U, V, W entries
    
    # Let me define for each S_c:
    # - Which A blocks and their signs
    # - Which B blocks and their signs
    
    strassen_super = [
        # (A_blocks: list of (block_idx, sign), B_blocks: list of (block_idx, sign))
        ([(0,1),(1,1)], [(0,1),(1,1)]),  # S1: (A11+A22)*(B11+B22)
        ([(1,0),(1,1)], [(0,1)]),          # S2: (A21+A22)*B11
        ([(0,0)], [(0,1),(1,1)]),          # S3: A11*(B12-B22)  -- wait
    ]
    
    # Let me be more careful. Block indices: 
    # A11 = block (0,0), A12 = block (0,1), A21 = block (1,0), A22 = block (1,1)
    # B11 = block (0,0), B12 = block (0,1), B21 = block (1,0), B22 = block (1,1)
    
    # S1 = (A11+A22)*(B11+B22): A blocks (0,0)+, (1,1)+; B blocks (0,0)+, (1,1)+
    # S2 = (A21+A22)*B11: A blocks (1,0)+, (1,1)+; B blocks (0,0)+
    # S3 = A11*(B12-B22): A blocks (0,0)+; B blocks (0,1)+, (1,1)-
    # S4 = A22*(B21-B11): A blocks (1,1)+; B blocks (1,0)+, (0,0)-
    # S5 = (A11+A12)*B22: A blocks (0,0)+, (0,1)+; B blocks (1,1)+
    # S6 = (A21-A11)*(B11+B12): A blocks (1,0)+, (0,0)-; B blocks (0,0)+, (0,1)+
    # S7 = (A12-A22)*(B21+B22): A blocks (0,1)+, (1,1)-; B blocks (1,0)+, (1,1)+
    
    super_defs = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),  # S1
        ([(1,0,1),(1,1,1)], [(0,0,1)]),            # S2
        ([(0,0,1)], [(0,1,1),(1,1,-1)]),           # S3
        ([(1,1,1)], [(1,0,1),(0,0,-1)]),           # S4
        ([(0,0,1),(0,1,1)], [(1,1,1)]),            # S5
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)]),   # S6
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)]),   # S7
    ]
    
    # 2x2x2 Strassen factors (U2, V2, W2) as computed above
    U2, V2, W2 = _make_2x2_strassen_factors()
    
    # For each super-factor c (0..6) and sub-factor s (0..6):
    # The A-part of sub-factor s in the 2x2x2 Strassen uses A_local[r,s]
    # where r is the local row (0 or 1) in the 2x2 block
    # But the A blocks involved in super-factor c may be multiple blocks.
    # Each A block has a sign. The local position within the block matters.
    
    # For sub-factor s, U2[row_local, s] gives the coefficient for A_local[row_local, col_local]
    # But in our 2x2x2, the A entries are A00, A01, A10, A11 (local 2x2)
    # U2 is indexed by row = i*2+j for A[i,j]
    
    # For a given super-factor c and sub-factor s:
    # For each A block (bi, bj) with sign sa in super_defs[c][0]:
    #   For each local position (r, s_local) in the 2x2 block:
    #     A_global position = (bi*2+r, bj*2+s_local)
    #     U2 index for this local position = r*2+s_local
    #     Contribution to U[(bi*2+r)*4 + (bj*2+s_local), c*7+s] += sa * U2[r*2+s_local, s]
    
    # Similarly for V and W.
    
    for c in range(7):
        a_blocks = super_defs[c][0]  # list of (bi, bj, sign)
        b_blocks = super_defs[c][1]
        
        for s in range(7):
            col = c * 7 + s
            
            # U: for each A block
            for (bi, bj, sa) in a_blocks:
                for r in range(2):
                    for sl in range(2):
                        a_global_i = bi * 2 + r
                        a_global_j = bj * 2 + sl
                        u2_val = U2[r*2+sl, s]
                        if u2_val != 0:
                            U[a_global_i * 4 + a_global_j][col] += sa * u2_val
            
            # V: for each B block
            # B entry (j,k): block index (j//2, k//2), local (j%2, k%2)
            # V2 is indexed by row = j_local*2 + k_local
            for (bi, bj, sb) in b_blocks:
                for jl in range(2):
                    for kl in range(2):
                        b_global_j = bi * 2 + jl
                        b_global_k = bj * 2 + kl
                        v2_val = V2[jl*2+kl, s]
                        if v2_val != 0:
                            V[b_global_j * 4 + b_global_k][col] += sb * v2_val
            
            # W: for each C block
            # C entry (i,k): block index (i//2, k//2), local (i%2, k%2)
            # W2 is indexed by row = k_local*2 + i_local
            # C = sum of super-factors with signs from Strassen output
            # C11 = S1 + S4 - S5 - S7
            # C12 = S3 + S5
            # C21 = S2 + S4
            # C22 = S1 - S2 + S3 + S6
            
            c_block_signs = {
                (0,0): [1, 0, 0, 1, -1, 0, -1],  # C11: S1+S4-S5-S7
                (0,1): [0, 0, 1, 0, 1, 0, 0],     # C12: S3+S5
                (1,0): [0, 1, 0, 1, 0, 0, 0],     # C21: S2+S4
                (1,1): [1, -1, 1, 0, 0, 1, 0],    # C22: S1-S2+S3+S6
            }
            
            for (ci, ck, sign) in [(0,0,1),(0,0,1),(0,1,1),(1,0,1),(1,1,1)]:
                pass
            
            # For each C block (ci, ck) with super-sign:
            for ci in range(2):
                for ck in range(2):
                    super_sign = c_block_signs[(ci,ck)][c]
                    if super_sign == 0:
                        continue
                    for il in range(2):
                        for kl in range(2):
                            c_global_i = ci * 2 + il
                            c_global_k = ck * 2 + kl
                            w2_val = W2[kl*2+il, s]
                            if w2_val != 0:
                                W[c_global_k * 4 + c_global_i][col] += super_sign * w2_val
    
    return U, V, W, rank


def _make_4x4x1_factors():
    """
    Schoolbook factors for 4x4x1: A(4x4) * b(4x1) -> c(4x1)
    Rank = 16
    U: 16x16, V: 4x16, W: 4x16
    C[i,0] = sum_j A[i,j] * B[j,0]
    """
    n, p = 4, 1
    rank = n * p  # 4*4 = 16... wait, m=4, so it's A(4x4)*B(4x1)
    # C[i,k] for i in 0..3, k=0 only
    # C[i,0] = sum_j A[i,j]*B[j,0]
    # U[i*4+j, c] where c = i*4+j... no, rank = 4*4 = 16
    # Each term: A[i,j] * B[j,0] contributes to C[i,0]
    rank = 16
    U = [[0]*rank for _ in range(16)]  # A is 4x4
    V = [[0]*rank for _ in range(4)]   # B is 4x1, m*p = 4*1 = 4
    W = [[0]*rank for _ in range(4)]   # C is 4x1, p*n = 1*4 = 4, W[k*n+i] = W[i]
    
    col = 0
    for i in range(4):
        for j in range(4):
            U[i*4+j][col] = 1
            V[j][col] = 1
            W[i][col] = 1
            col += 1
    
    return U, V, W, rank


def _concatenate(U_list, V_list, W_list):
    """Concatenate factor matrices horizontally (along rank dimension)."""
    total_rank = sum(len(u[0]) for u in U_list)
    n_rows = [len(u) for u in U_list]
    # All should have same number of rows per type
    U_out = [[0]*total_rank for _ in range(n_rows[0])]
    V_out = [[0]*total_rank for _ in range(len(V_list[0]))]
    W_out = [[0]*total_rank for _ in range(len(W_list[0]))]
    
    col_offset = 0
    for U_i, V_i, W_i in zip(U_list, V_list, W_list):
        r_i = len(U_i[0])
        for row in range(len(U_i)):
            for c in range(r_i):
                U_out[row][col_offset + c] = U_i[row][c]
        for row in range(len(V_i)):
            for c in range(r_i):
                V_out[row][col_offset + c] = V_i[row][c]
        for row in range(len(W_i)):
            for c in range(r_i):
                W_out[row][col_offset + c] = W_i[row][c]
        col_offset += r_i
    
    return U_out, V_out, W_out, total_rank


def construct():
    import time
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # Strategy: 4x4x9 = 4x4x4 (Strassen, rank 49) + 4x4x4 (Strassen, rank 49) + 4x4x1 (schoolbook, rank 16)
    # Total rank: 114
    
    U_444_1, V_444_1, W_444_1, r1 = _make_4x4x4_strassen_factors()
    U_444_2, V_444_2, W_444_2, r2 = _make_4x4x4_strassen_factors()
    U_441, V_441, W_441, r3 = _make_4x4x1_factors()
    
    # For 4x4x4: U is 16x49, V is 16x49, W is 16x49
    # For 4x4x1: U is 16x16, V is 4x16, W is 4x16
    # 
    # The full problem: A(4x4) * B(4x9) -> C(4x9)
    # B columns 0-3: first 4x4x4
    # B columns 4-7: second 4x4x4
    # B column 8: 4x4x1
    #
    # U for the full problem: 16 x (49+49+16)
    # V for the full problem: 36 x (49+49+16)  [m*p = 4*9 = 36]
    # W for the full problem: 36 x (49+49+16)  [p*n = 9*4 = 36]
    
    total_rank = r1 + r2 + r3  # 49 + 49 + 16 = 114
    
    U_full = [[0]*total_rank for _ in range(16)]
    V_full = [[0]*total_rank for _ in range(36)]
    W_full = [[0]*total_rank for _ in range(36)]
    
    # First 4x4x4: columns 0-3 of B, columns 0-3 of C
    col_off = 0
    for row in range(16):
        for c in range(r1):
            U_full[row][col_off + c] = U_444_1[row][c]
    for row in range(16):  # V rows 0-15 correspond to B[0:4, 0:4]
        for c in range(r1):
            V_full[row][col_off + c] = V_444_1[row][c]
    for row in range(16):  # W rows 0-15 correspond to C[0:4, 0:4] -> k*n+i for k=0..3, i=0..3
        for c in range(r1):
            W_full[row][col_off + c] = W_444_1[row][c]
    col_off += r1
    
    # Second 4x4x4: columns 4-7 of B, columns 4-7 of C
    for row in range(16):
        for c in range(r2):
            U_full[row][col_off + c] = U_444_2[row][c]
    for row in range(16):  # V rows 16-31 correspond to B[0:4, 4:8]
        for c in range(r2):
            V_full[16 + row][col_off + c] = V_444_2[row][c]
    for row in range(16):  # W rows 16-31 correspond to C[4:8, 0:4] -> k=4..7, i=0..3
        for c in range(r2):
            W_full[16 + row][col_off + c] = W_444_2[row][c]
    col_off += r2
    
    # 4x4x1: column 8 of B, column 8 of C
    for row in range(16):
        for c in range(r3):
            U_full[row][col_off + c] = U_441[row][c]
    for row in range(4):  # V row 32 corresponds to B[0:4, 8]
        for c in range(r3):
            V_full[32 + row][col_off + c] = V_441[row][c]
    for row in range(4):  # W row 32 corresponds to C[8, 0:4] -> k=8, i=0..3
        for c in range(r3):
            W_full[32 + row][col_off + c] = W_441[row][c]
    
    result = {"U": U_full, "V": V_full, "W": W_full}
    
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        pass  # We're still within budget since this is fast
    
    return result
