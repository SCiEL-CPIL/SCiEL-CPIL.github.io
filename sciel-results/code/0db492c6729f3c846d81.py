"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2_tensors():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    Shapes: (4, 7)
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # P1 = (a11+a22)*(b11+b22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (a21+a22)*b11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a11*(b12-b22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = a22*(b21-b11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a11+a12)*b22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (a21-a11)*(b11+b12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (a12-a22)*(b21+b22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_decomposition_4x4x4():
    """
    Builds 4x4x4 decomposition using 4x4 Strassen (Rank 49).
    This is composed of 7 Strassen 2x2x2 calls, each of which is 7 rank-1 terms.
    Total rank 49.
    """
    U2, V2, W2 = _strassen_2x2_tensors()
    rank = 49
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(16)]
    W = [[0]*rank for _ in range(16)]
    
    # Strassen 2x2 definitions for the 4x4x4 case
    # The 4x4x4 Strassen uses 7 sub-multiplications of 2x2x2.
    # Each sub-multiplication involves linear combinations of 2x2 blocks.
    
    # We define the 7 Strassen steps for the 4x4x4 matrix multiplication.
    # A, B, C are 4x4 matrices partitioned into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    
    # The 7 products P1..P7 are computed.
    # Each Pi is a 2x2 matrix product.
    # We need to map the 2x2x2 Strassen tensors to the global 4x4x4 indices.
    
    # Helper to add a 2x2x2 Strassen block to the global matrices
    def add_block(a_i, a_j, b_i, b_j, c_i, c_j, base_rank):
        # a_i, a_j: row/col offset of A block (0 or 1, representing 2x2 blocks)
        # b_i, b_j: row/col offset of B block
        # c_i, c_j: row/col offset of C block
        # The block is A[a_i*2:(a_i+1)*2, a_j*2:(a_j+1)*2] * B[b_i*2:(b_i+1)*2, b_j*2:(b_j+1)*2] -> C[c_i*2:(c_i+1)*2, c_j*2:(c_j+1)*2]
        
        for t in range(7):
            col = base_rank + t
            # U part
            for r in range(4):
                ur = U2[r][t]
                if ur != 0:
                    local_i = r // 2
                    local_j = r % 2
                    global_i = (a_i * 2) + local_i
                    global_j = (a_j * 2) + local_j
                    U[global_i * 4 + global_j][col] += ur
            
            # V part
            for r in range(4):
                vr = V2[r][t]
                if vr != 0:
                    local_j = r // 2
                    local_k = r % 2
                    global_j = (b_i * 2) + local_j
                    global_k = (b_j * 2) + local_k
                    V[global_j * 4 + global_k][col] += vr
            
            # W part
            for r in range(4):
                wr = W2[r][t]
                if wr != 0:
                    local_k = r // 2
                    local_i = r % 2
                    global_k = (c_j * 2) + local_k
                    global_i = (c_i * 2) + local_i
                    W[global_k * 4 + global_i][col] += wr

    # Strassen 2x2x2 definitions for the block-level 2x2 matrices
    # P1 = (A11+A22)(B11+B22)
    # We need to express A11+A22 as a linear combination of 2x2 blocks.
    # This requires 2 Strassen 2x2x2 calls? No.
    # The standard 4x4x4 Strassen uses 7 multiplications of 2x2 matrices.
    # Each multiplication is a 2x2x2 tensor.
    # So we call add_block 7 times?
    # No, the inputs to the 2x2x2 multiplications are sums of blocks.
    # E.g., P1 input A is A11+A22. This is not a single 2x2 block.
    # So we cannot just call add_block once.
    # We must expand the sums.
    
    # This makes the 4x4x4 construction complex.
    # Alternative: Use the known fact that Rank(4x4x4) = 49.
    # We can construct it by using 7 sets of 7 rank-1 tensors.
    # But the inputs are linear combinations.
    
    # Let's use a simpler verified decomposition for 4x4x4 if possible, 
    # or stick to the 101 rank which is sum of simpler blocks.
    
    # For 4x4x4, we can use 4x4x2 (28) + 4x4x2 (28)? No, 4x4x4 is not two 4x4x2.
    # 4x4x4 = 4x4x2 + 4x4x2?
    # C = A B. Split B into B1 (4x2) and B2 (4x2).
    # C1 = A B1 (4x2), C2 = A B2 (4x2).
    # Rank(4x4x2) = 28.
    # So Rank(4x4x4) <= 28 + 28 = 56.
    # But Strassen gives 49.
    # So 56 is not optimal.
    
    # I will implement the 101 rank solution using:
    # 1. 4x4x4 Strassen (49)
    # 2. 4x1x6 Schoolbook (24)
    # 3. 4x4x2 Optimized (28)
    
    # To avoid the complexity of 4x4x4 Strassen, I will use the 56 rank for 4x4x4?
    # No, 49 is better.
    # I will implement the 4x4x4 Strassen correctly.
    
    # The 7 Strassen products for 4x4x4 are:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B12-B22)
    # P4 = A22(B21-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    
    # Each Pi is a 2x2 matrix.
    # To compute Pi, we use the 2x2x2 Strassen decomposition.
    # This requires 7 scalar multiplications (in the tensor sense, rank 1).
    # So total 7 * 7 = 49.
    
    # We need to map the "A_part" and "B_part" of each Pi to the global indices.
    # E.g., for P1, A_part = A11 + A22.
    # In the U matrix, we add U2 coefficients for A11 and A22.
    
    def add_strassen_product(a_blocks, b_blocks, c_blocks, base_rank):
        # a_blocks: list of (i_offset, j_offset, coeff)
        # b_blocks: list of (j_offset, k_offset, coeff)
        # c_blocks: list of (k_offset, i_offset, coeff)
        
        for t in range(7):
            col = base_rank + t
            
            # U
            for r in range(4):
                ur = U2[r][t]
                if ur != 0:
                    local_i = r // 2
                    local_j = r % 2
                    for (ai, aj, ca) in a_blocks:
                        global_i = ai * 2 + local_i
                        global_j = aj * 2 + local_j
                        U[global_i * 4 + global_j][col] += ur * ca
                        
            # V
            for r in range(4):
                vr = V2[r][t]
                if vr != 0:
                    local_j = r // 2
                    local_k = r % 2
                    for (bj, bk, cb) in b_blocks:
                        global_j = bj * 2 + local_j
                        global_k = bk * 2 + local_k
                        V[global_j * 4 + global_k][col] += vr * cb
                        
            # W
            for r in range(4):
                wr = W2[r][t]
                if wr != 0:
                    local_k = r // 2
                    local_i = r % 2
                    for (ck, ci, cc) in c_blocks:
                        global_k = ck * 2 + local_k
                        global_i = ci * 2 + local_i
                        W[global_k * 4 + global_i][col] += wr * cc

    # Define the 7 products
    # P1
    add_strassen_product(
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)],
        [(0,0,1), (1,1,1)],
        0
    )
    # P2
    add_strassen_product(
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(1,0,1), (1,1,-1)],
        7
    )
    # P3
    add_strassen_product(
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        14
    )
    # P4
    add_strassen_product(
        [(1,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (1,0,1)],
        21
    )
    # P5
    add_strassen_product(
        [(0,0,1), (0,1,1)],
        [(1,1,1)],
        [(0,0,-1), (0,1,1)],
        28
    )
    # P6
    add_strassen_product(
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1)],
        [(1,1,1)],
        35
    )
    # P7
    add_strassen_product(
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        42
    )
    
    return U, V, W

def _build_decomposition_4x4x2():
    """
    Builds 4x4x2 decomposition (Rank 28).
    A (4x4) * B (4x2) -> C (4x2).
    We split A into 2x2 blocks (4 blocks).
    B into 2x2 blocks (2 blocks).
    C into 2x2 blocks (2 blocks).
    This is 2 (i) * 1 (l) * 2 (k) = 4 blocks of 2x2x2?
    No, B is 4x2. Splitting rows of B into 2+2 gives two 2x2 matrices.
    Splitting rows of A into 2+2 gives two 2x2 matrices.
    Splitting rows of C into 2+2 gives two 2x2 matrices.
    The multiplication is C = A B.
    C_top = A_top B_top + A_top B_bot? No.
    C_top = A_top B_top + A_bot B_top?
    A = [[A11, A12], [A21, A22]]
    B = [[B11, B12], [B21, B22]] (B is 4x2, so B11 is 2x2, B12 is 2x2, etc.)
    C = [[C11, C12], [C21, C22]]
    C11 = A11 B11 + A12 B21
    C12 = A11 B12 + A12 B22
    C21 = A21 B11 + A22 B21
    C22 = A21 B12 + A22 B22
    
    This requires 8 multiplications of 2x2 matrices.
    Rank 8 * 7 = 56.
    But we want 28.
    How to get 28?
    Maybe 4x4x2 can be done with 4 Strassen blocks?
    If we view it as 4 independent 2x2x2 multiplications?
    No, the sums couple them.
    
    Actually, Rank(4x4x2) is known to be 28?
    Let's check if we can use 2x2x2 Strassen on the whole thing?
    No.
    
    I will use the 56 rank for 4x4x2 if 28 is not easily constructible.
    Total rank: 49 (4x4x4) + 24 (4x1x6) + 56 (4x4x2) = 129.
    This is worse than 101.
    
    So I must find the 28 rank for 4x4x2.
    Or use a different partition.
    
    Let's use 4x4x4 (49) + 4x1x6 (24) + 4x4x2 (28).
    If I can't get 28, I'll use schoolbook for 4x4x2 (32).
    Total 49 + 24 + 32 = 105.
    This is better than 120, but worse than 101.
    
    I will implement the 105 solution first to ensure correctness, then try to optimize 4x4x2.
    """
    U2, V2, W2 = _strassen_2x2_tensors()
    rank = 28 # Assume 28 is possible, let's try to construct it.
    # If 28 is not possible, I will return schoolbook (32).
    
    # For now, I will use schoolbook for 4x4x2 to be safe.
    rank = 32
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(8)]
    W = [[0]*rank for _ in range(8)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U[i*4+j][col] = 1
                V[j*2+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
                
    return U, V, W

def _build_decomposition_4x1x6():
    """
    Builds 4x1x6 decomposition (Rank 24).
    A (4x1) * B (1x6) -> C (4x6).
    Schoolbook.
    """
    rank = 24
    U = [[0]*rank for _ in range(4)]
    V = [[0]*rank for _ in range(6)]
    W = [[0]*rank for _ in range(24)]
    
    col = 0
    for i in range(4):
        for k in range(6):
            U[i][col] = 1
            V[k][col] = 1
            W[k*4+i][col] = 1
            col += 1
            
    return U, V, W

def _combine_decompositions(decomps, n, m, p):
    total_rank = sum(len(d[0][0]) for d in decomps)
    U = [[0]*total_rank for _ in range(n*m)]
    V = [[0]*total_rank for _ in range(m*p)]
    W = [[0]*total_rank for _ in range(p*n)]
    
    current_rank = 0
    for U_sub, V_sub, W_sub, (i_s, i_e), (j_s, j_e), (k_s, k_e) in decomps:
        n_sub = i_e - i_s
        m_sub = j_e - j_s
        p_sub = k_e - k_s
        rank_sub = len(U_sub[0])
        
        for r_sub in range(n_sub * m_sub):
            i_loc = r_sub // m_sub
            j_loc = r_sub % m_sub
            i_glob = i_s + i_loc
            j_glob = j_s + j_loc
            r_glob = i_glob * m + j_glob
            for c_sub in range(rank_sub):
                U[r_glob][current_rank + c_sub] = U_sub[r_sub][c_sub]
        
        for r_sub in range(m_sub * p_sub):
            j_loc = r_sub // p_sub
            k_loc = r_sub % p_sub
            j_glob = j_s + j_loc
            k_glob = k_s + k_loc
            r_glob = j_glob * p + k_glob
            for c_sub in range(rank_sub):
                V[r_glob][current_rank + c_sub] = V_sub[r_sub][c_sub]
        
        for r_sub in range(p_sub * n_sub):
            k_loc = r_sub // n_sub
            i_loc = r_sub % n_sub
            k_glob = k_s + k_loc
            i_glob = i_s + i_loc
            r_glob = k_glob * n + i_glob
            for c_sub in range(rank_sub):
                W[r_glob][current_rank + c_sub] = W_sub[r_sub][c_sub]
        
        current_rank += rank_sub
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    U1, V1, W1 = _build_decomposition_4x4x4()
    U2, V2, W2 = _build_decomposition_4x1x6()
    U3, V3, W3 = _build_decomposition_4x4x2()
    
    U, V, W = _combine_decompositions(
        [(U1, V1, W1, (0, 4), (0, 4), (0, 4)),
         (U2, V2, W2, (0, 4), (4, 5), (0, 6)),
         (U3, V3, W3, (0, 4), (0, 4), (4, 6))],
        n, m, p
    )
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
