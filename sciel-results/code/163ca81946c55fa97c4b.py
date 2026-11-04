"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for multiplying a 2x4 matrix by a 4x8 matrix.
    We use a block decomposition approach.
    
    The multiplication is C = A * B where A is 2x4, B is 4x8, C is 2x8.
    
    We can decompose this into two independent 2x2x2 multiplications:
    - Left half: A[:, 0:2] * B[0:2, :] gives C[:, 0:2]  (wait, this isn't quite right)
    
    Actually, let me think more carefully. The standard way to decompose:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    We can group the j indices. If we split j into {0,1} and {2,3}:
    C[i,k] = sum_{j in {0,1}} A[i,j] B[j,k] + sum_{j in {2,3}} A[i,j] B[j,k]
    
    This gives us two independent 2x2x2 multiplications:
    1. A1 = A[:, 0:2] (2x2), B1 = B[0:2, :] (2x8), C1 = C[:, :] (2x8)
    2. A2 = A[:, 2:4] (2x2), B2 = B[2:4, :] (2x8), C2 = C[:, :] (2x8)
    
    Wait, that's not quite right either. Let me reconsider.
    
    Actually, for the first part: C1[i,k] = sum_{j=0}^{1} A[i,j] B[j,k]
    This is a 2x2x8 multiplication, not 2x2x2.
    
    Let me think differently. We can decompose the 4x8 part of B.
    
    Alternative approach: decompose into four 2x2x2 blocks.
    A is 2x4, B is 4x8.
    
    Split A into A[:, 0:2] and A[:, 2:4] (each 2x2)
    Split B into B[0:2, :] and B[2:4, :] (each 2x8)
    
    C = A[:, 0:2] * B[0:2, :] + A[:, 2:4] * B[2:4, :]
    
    Now each term is a 2x2x8 multiplication.
    
    For a 2x2x8 multiplication, we can further decompose the 8 columns of B into four 2-column blocks.
    B[0:2, 0:2], B[0:2, 2:4], B[0:2, 4:6], B[0:2, 6:8]
    
    So C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]
    etc.
    
    Each of these is a sum of two 2x2x2 multiplications.
    
    So we have 8 independent 2x2x2 multiplications (2 j-blocks × 4 k-blocks).
    Using Strassen's algorithm (rank 7) for each 2x2x2, we get total rank = 8 × 7 = 56.
    
    This is better than the schoolbook rank of 64.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # We'll decompose into 8 independent 2x2x2 multiplications
    # j-blocks: [0,1] and [2,3]
    # k-blocks: [0,1], [2,3], [4,5], [6,7]
    
    # Strassen decomposition for 2x2x2:
    # Local A (2x2), B (2x2), C (2x2)
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Define Strassen coefficients for U, V, W
    # For local indices (a, b, c) in {0,1}:
    # U contributes to A[a,b], V contributes to B[b,c], W contributes to C[c,a]
    
    # Strassen matrices (7 tensors):
    # Each tensor is (U_part, V_part, W_part) where each part is a vector in the local space
    
    strassen_tensors = []
    
    # M1 = (A00 + A11)(B00 + B11)
    # U: A00=1, A11=1 -> (1,0,0,1) in (A00,A01,A10,A11)
    # V: B00=1, B11=1 -> (1,0,0,1) in (B00,B01,B10,B11)
    # W: C00=1, C11=1 -> (1,0,0,1) in (C00,C01,C10,C11)
    strassen_tensors.append((
        [1, 0, 0, 1],  # U part for A
        [1, 0, 0, 1],  # V part for B
        [1, 0, 0, 1]   # W part for C
    ))
    
    # M2 = (A10 + A11)B00
    # U: A10=1, A11=1 -> (0,0,1,1)
    # V: B00=1 -> (1,0,0,0)
    # W: C10=1 -> (0,0,1,0)
    strassen_tensors.append((
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ))
    
    # M3 = A00(B01 - B11)
    # U: A00=1 -> (1,0,0,0)
    # V: B01=1, B11=-1 -> (0,1,0,-1)
    # W: C01=1 -> (0,1,0,0)
    strassen_tensors.append((
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 1, 0, 0]
    ))
    
    # M4 = A11(B10 - B00)
    # U: A11=1 -> (0,0,0,1)
    # V: B10=1, B00=-1 -> (-1,0,1,0)
    # W: C00=1, C10=1 -> (1,0,1,0)
    strassen_tensors.append((
        [0, 0, 0, 1],
        [-1, 0, 1, 0],
        [1, 0, 1, 0]
    ))
    
    # M5 = (A00 + A01)B11
    # U: A00=1, A01=1 -> (1,1,0,0)
    # V: B11=1 -> (0,0,0,1)
    # W: C01=1, C11=-1 -> (0,1,0,-1)
    strassen_tensors.append((
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, -1]
    ))
    
    # M6 = (A10 - A00)(B00 + B01)
    # U: A10=1, A00=-1 -> (-1,0,1,0)
    # V: B00=1, B01=1 -> (1,1,0,0)
    # W: C11=1 -> (0,0,0,1)
    strassen_tensors.append((
        [-1, 0, 1, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1]
    ))
    
    # M7 = (A01 - A11)(B10 + B11)
    # U: A01=1, A11=-1 -> (0,1,0,-1)
    # V: B10=1, B11=1 -> (0,0,1,1)
    # W: C00=1, C11=-1 -> (1,0,0,-1)
    strassen_tensors.append((
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 0, 0, -1]
    ))
    
    # Now construct the global U, V, W matrices
    # U shape: (n*m, rank) = (2*4, 56) = (8, 56)
    # V shape: (m*p, rank) = (4*8, 56) = (32, 56)
    # W shape: (p*n, rank) = (8*2, 56) = (16, 56)
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    col = 0
    # j-blocks: 0 -> j in [0,1], 1 -> j in [2,3]
    # k-blocks: 0 -> k in [0,1], 1 -> k in [2,3], 2 -> k in [4,5], 3 -> k in [6,7]
    
    for jb in range(2):  # j-block index
        for kb in range(4):  # k-block index
            for t in range(7):  # Strassen tensor index
                u_part, v_part, w_part = strassen_tensors[t]
                
                # Local A indices: i in {0,1}, a in {0,1} where a corresponds to j-offset within block
                # Global A index: i * m + (jb * 2 + a)
                # u_part is indexed as (a00, a01, a10, a11) where a = (i_local, j_local)
                # Actually, u_part[i_local * 2 + j_local]
                
                for i_local in range(2):
                    for a_local in range(2):
                        coeff = u_part[i_local * 2 + a_local]
                        if coeff != 0:
                            i_global = i_local  # i is already 0 or 1
                            j_global = jb * 2 + a_local
                            idx_U = i_global * m + j_global
                            U[idx_U][col] += coeff
                
                # Local B indices: b in {0,1} (j-offset), c in {0,1} (k-offset)
                # Global B index: (jb * 2 + b) * p + (kb * 2 + c)
                # v_part is indexed as (b0c0, b0c1, b1c0, b1c1) = (b_local * 2 + c_local)
                
                for b_local in range(2):
                    for c_local in range(2):
                        coeff = v_part[b_local * 2 + c_local]
                        if coeff != 0:
                            j_global = jb * 2 + b_local
                            k_global = kb * 2 + c_local
                            idx_V = j_global * p + k_global
                            V[idx_V][col] += coeff
                
                # Local C indices: c in {0,1} (k-offset), i in {0,1}
                # Global C index: (kb * 2 + c) * n + i
                # w_part is indexed as (c0i0, c0i1, c1i0, c1i1) = (c_local * 2 + i_local)
                
                for c_local in range(2):
                    for i_local in range(2):
                        coeff = w_part[c_local * 2 + i_local]
                        if coeff != 0:
                            k_global = kb * 2 + c_local
                            i_global = i_local
                            idx_W = k_global * n + i_global
                            W[idx_W][col] += coeff
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
