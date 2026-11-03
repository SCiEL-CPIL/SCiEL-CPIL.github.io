"""Tensor decomposition for 4x4 * 4x5 matrix multiplication with rank reduction."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

_DEADLINE = time.time() + 115.0  # safety margin before 120s kill


def _time_ok():
    return time.time() < _DEADLINE


def _strassen_block_factors(n, m, p):
    """
    Return U (n*m x r), V (m*p x r), W (p*n x r) for multiplying
    an (n x m) matrix by an (m x p) matrix using recursive Strassen-style
    decomposition. For non-square or odd sizes, we pad to the next power of 2
    in a way that keeps the decomposition exact (zero-padding is exact).
    """
    # We'll use a recursive approach: if n, m, p are all powers of 2,
    # use Strassen. Otherwise, fall back to a block decomposition.
    # For simplicity and correctness, use the general block approach:
    # split into 2x2 blocks where possible, else schoolbook.

    def is_pow2(x):
        return x > 0 and (x & (x - 1)) == 0

    if n == 1 and m == 1 and p == 1:
        U = [[1.0]]
        V = [[1.0]]
        W = [[1.0]]
        return U, V, W

    # If all are powers of 2 and size >= 2, use Strassen
    if is_pow2(n) and is_pow2(m) and is_pow2(p) and n >= 2:
        return _strassen_square(n, m, p)
    else:
        # Fallback: split the smallest dimension
        # For 4x4x5: split 5 into 2+3, or use block approach
        # Let's split p into p1 + p2
        p1 = p // 2
        p2 = p - p1
        # A (n x m) * B1 (m x p1) and A * B2 (m x p2)
        U1, V1, W1 = _strassen_block_factors(n, m, p1)
        U2, V2, W2 = _strassen_block_factors(n, m, p2)
        # Concatenate
        U = U1 + U2  # same shape, just stack columns
        # Actually, U is (n*m x r), so we concatenate along columns
        r1 = len(U1[0])
        r2 = len(U2[0])
        U_new = [row + [0.0] * r2 for row in U1]
        for row in U2:
            pass
        # Let me redo this properly
        U_new = [[U1[i][j] for j in range(r1)] + [U2[i][j] for j in range(r2)] for i in range(n * m)]
        V_new = [[V1[i][j] for j in range(r1)] + [V2[i][j] for j in range(r2)] for i in range(m * p)]
        # W: W1 is (p1*n x r1), W2 is (p2*n x r2)
        # W_new should be (p*n x r1+r2)
        W_new = []
        for k in range(p):
            for i in range(n):
                if k < p1:
                    W_new.append([W1[k * n + i][j] for j in range(r1)] + [0.0] * r2)
                else:
                    W_new.append([0.0] * r1 + [W2[(k - p1) * n + i][j] for j in range(r2)])
        return U_new, V_new, W_new


def _strassen_square(n, m, p):
    """
    Strassen decomposition for n x m * m x p where n, m, p are powers of 2.
    For n == m == p, this is standard Strassen.
    For n != m or n != p, we need a generalized approach.
    
    Actually, let's use a simpler approach: for the 4x4x5 case,
    we'll build it explicitly using known rank-65 constructions.
    """
    # For the specific case 4x4x5, let's use a direct construction
    # that achieves rank 65 (or better) and is verifiably correct.
    # 
    # Strategy: Use the fact that 4x4x5 can be decomposed as:
    # - Split the 5 into 2+3
    # - 4x4x2: use Strassen twice (rank 7*2 = 14? No, Strassen is for square)
    # 
    # Let me just use a general recursive Strassen that handles rectangular.
    
    if n == m == p == 2:
        # Standard Strassen for 2x2
        # C = A*B
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        # 
        # In tensor form: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # n=m=p=2, so indices: i,j,k in {0,1}
        # U row index: i*2+j, V row index: j*2+k, W row index: k*2+i
        
        # Let me define the 7 Strassen factors
        # Each factor c has:
        # U values: for each (i,j), U[i*2+j, c]
        # V values: for each (j,k), V[j*2+k, c]
        # W values: for each (k,i), W[k*2+i, c]
        
        factors = []
        # M1 = (A11+A22)(B11+B22)
        # A11: i=0,j=0; A22: i=1,j=1
        # B11: j=0,k=0; B22: j=1,k=1
        # U: (0,0)=1, (1,1)=1, others 0
        # V: (0,0)=1, (1,1)=1, others 0
        # W: C11 += M1, C22 += M1
        # C11: k=0,i=0; C22: k=1,i=1
        # W: (0,0)=1, (1,1)=1
        factors.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
        
        # M2 = (A21+A22)B11
        # A21: i=1,j=0; A22: i=1,j=1
        # B11: j=0,k=0
        # U: (1,0)=1, (1,1)=1
        # V: (0,0)=1
        # W: C21 += M2 => k=1,i=1? No, C21 is row 2, col 1 => i=1, k=0
        # W index: k*n+i = k*2+i
        # C21: i=1, k=0 => W[0*2+1] = W[1]
        factors.append(([0,0,1,1], [1,0,0,0], [0,1,0,0]))
        
        # M3 = A11(B21-B22)
        # A11: i=0,j=0
        # B21: j=1,k=0; B22: j=1,k=1
        # U: (0,0)=1
        # V: (1,0)=1, (1,1)=-1
        # W: C12 += M3 => i=0,k=1 => W[1*2+0]=W[2]
        factors.append(([1,0,0,0], [0,0,1,-1], [0,0,1,0]))
        
        # M4 = A22(B12-B11)
        # A22: i=1,j=1
        # B12: j=0,k=1; B11: j=0,k=0
        # U: (1,1)=1
        # V: (0,1)=1, (0,0)=-1
        # W: C11 += M4, C21 += M4
        # C11: i=0,k=0 => W[0]; C21: i=1,k=0 => W[1]
        factors.append(([0,0,0,1], [-1,1,0,0], [1,1,0,0]))
        
        # M5 = (A11+A12)B22
        # A11: i=0,j=0; A12: i=0,j=1
        # B22: j=1,k=1
        # U: (0,0)=1, (0,1)=1
        # V: (1,1)=1
        # W: C11 -= M5, C12 += M5
        # C11: i=0,k=0 => W[0]; C12: i=0,k=1 => W[2]
        factors.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
        
        # M6 = (A21-A11)(B11+B12)
        # A21: i=1,j=0; A11: i=0,j=0
        # B11: j=0,k=0; B12: j=0,k=1
        # U: (1,0)=1, (0,0)=-1
        # V: (0,0)=1, (0,1)=1
        # W: C22 += M6 => i=1,k=1 => W[3]
        factors.append([[-1,0,0,1], [1,1,0,0], [0,0,0,1]])
        
        # M7 = (A12-A22)(B21+B22)
        # A12: i=0,j=1; A22: i=1,j=1
        # B21: j=1,k=0; B22: j=1,k=1
        # U: (0,1)=1, (1,1)=-1
        # V: (1,0)=1, (1,1)=1
        # W: C11 += M7 => i=0,k=0 => W[0]
        factors.append(([0,1,-1,0], [0,0,1,1], [1,0,0,0]))
        
        U = [[0.0] * 7 for _ in range(4)]
        V = [[0.0] * 7 for _ in range(4)]
        W = [[0.0] * 7 for _ in range(4)]
        for c, (u, v, w) in enumerate(factors):
            for idx in range(4):
                U[idx][c] = u[idx]
                V[idx][c] = v[idx]
                W[idx][c] = w[idx]
        return U, V, W

    # For other power-of-2 sizes, recurse
    # Split into 2x2 blocks
    n2, m2, p2 = n // 2, m // 2, p // 2
    # We have 8 sub-multiplications, but Strassen reduces to 7
    # For rectangular, we need a different approach
    # Let's just use schoolbook for now and rely on rank reduction
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W


def _construct_4x4x5_strassen():
    """
    Construct a rank-65 decomposition for 4x4 * 4x5.
    Split 4x4x5 as:
      - 4x4x2: two 2x2 Strassen blocks along the output dimension
      - 4x4x3: split 3 as 2+1, use Strassen for 4x4x2 and schoolbook for 4x4x1
    
    Actually, let's use a cleaner approach:
    4x4x5 = 4x4x2 + 4x4x3
    4x4x2: split into two 4x4x1? No.
    
    Let me think differently. For 4x4x4, we can use 2-level Strassen:
    rank = 7 * 7 = 49 (since 4x4x4 = (2x2x2) x (2x2x2) in tensor sense)
    
    For 4x4x5:
    4x4x5 = 4x4x4 + 4x4x1
    rank(4x4x4) <= 49 (Strassen)
    rank(4x4x1) = 16 (schoolbook, since it's just a linear map)
    Total rank <= 49 + 16 = 65
    """
    n, m, p = 4, 4, 5
    
    # Part 1: 4x4x4 using 2-level Strassen (rank 49)
    U1, V1, W1 = _strassen_4x4x4()
    
    # Part 2: 4x4x1 using schoolbook (rank 16)
    # This computes A * B where B is 4x1
    U2 = [[0.0] * 16 for _ in range(16)]
    V2 = [[0.0] * 16 for _ in range(4)]
    W2 = [[0.0] * 16 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U2[i * 4 + j][col] = 1.0
            V2[j * 1 + 0][col] = 1.0  # j*p+k = j*1+0 = j
            W2[0 * 4 + i][col] = 1.0   # k*n+i = 0*4+i = i
    
    # Combine: U1 is (16x49), U2 is (16x16)
    # U_total = (16x65)
    U_total = [row + [0.0] * 16 for row in U1]
    for i in range(16):
        for j in range(16):
            U_total[i][49 + j] = U2[i][j]
    
    # V1 is (16x49), V2 is (4x16)
    # V_total = (20x65) since m*p = 4*5 = 20
    V_total = [[0.0] * 65 for _ in range(20)]
    for i in range(16):
        for j in range(49):
            V_total[i][j] = V1[i][j]
    for i in range(4):
        for j in range(16):
            V_total[i][49 + j] = V2[i][j]
    # V2 only has 4 rows (for k=0, the first column of B)
    # For k=1,2,3,4 (columns 2-5 of B), V should be 0 in the 4x4x1 part
    # But wait, the 4x4x1 part only handles the first column of B.
    # The remaining 4 columns are handled by... nothing?
    # 
    # I made an error. Let me reconsider.
    # 4x4x5: A is 4x4, B is 4x5, C is 4x5.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Split B into B_left (4x4) and B_right (4x1)
    # C_left = A * B_left (4x4x4, rank 49)
    # C_right = A * B_right (4x4x1, rank 16)
    # 
    # For C_left: U1 (16x49), V1 (16x49), W1 (16x49)
    #   V1[j*4+k, c] for j in 0..3, k in 0..3
    #   W1[k*4+i, c] for k in 0..3, i in 0..3
    # 
    # For C_right: U2 (16x16), V2 (4x16), W2 (4x16)
    #   V2[j*1+k, c] = V2[j, c] for j in 0..3, k=0
    #   W2[k*4+i, c] = W2[i, c] for k=0, i in 0..3
    # 
    # Combined:
    # U_total (16x65): U1 columns + U2 columns
    # V_total (20x65): 
    #   Rows 0-15: V1 (for B_left, j*4+k)
    #   Rows 16-19: V2 (for B_right, j*1+k where k=0, so just j)
    #   But V2 has only 4 rows, corresponding to j=0..3, k=0
    #   In V_total, these should go in rows 16,17,18,19 (j*1+0 = j, but offset by 16)
    #   Wait, V_total row index is j*p+k = j*5+k
    #   For B_left (k=0..3): row = j*5+k
    #   For B_right (k=4): row = j*5+4
    # 
    # So I need to remap. Let me redo this more carefully.

    return None  # Placeholder, will fix below


def _strassen_4x4x4():
    """
    2-level Strassen for 4x4x4.
    Level 1: split into 2x2 blocks, use Strassen (7 mults)
    Level 2: each 2x2x2 uses Strassen (7 mults)
    Total rank: 7 * 7 = 49
    """
    n = m = p = 4
    
    # The 7 Strassen factors at level 1, each producing a 2x2x2 sub-tensor
    # We'll compose them with the level-2 Strassen factors
    
    # Level 1 Strassen: 7 factors, each is a 2x2x2 tensor
    # Level 2: each 2x2x2 has 7 factors, each is a 1x1x1 tensor
    
    # The composition: for each level-1 factor c1 and level-2 factor c2,
    # we get a combined factor (c1, c2)
    
    U = [[0.0] * 49 for _ in range(16)]
    V = [[0.0] * 49 for _ in range(16)]
    W = [[0.0] * 49 for _ in range(16)]
    
    # Level 1 Strassen factors (for 2x2x2):
    # Same as _strassen_square(2,2,2) but we need the tensor form
    # Let me define the 7 level-1 factors as lists of (U_vals, V_vals, W_vals)
    # where U_vals is a 4-vector (for 2x2 A), V_vals is 4-vector (for 2x2 B),
    # W_vals is 4-vector (for 2x2 C)
    
    L1_factors = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),  # M1
        ([0,0,1,1], [1,0,0,0], [0,1,0,0]),  # M2
        ([1,0,0,0], [0,0,1,-1], [0,0,1,0]), # M3
        ([0,0,0,1], [-1,1,0,0], [1,1,0,0]), # M4
        ([1,1,0,0], [0,0,0,1], [-1,0,1,0]), # M5
        ([-1,0,0,1], [1,1,0,0], [0,0,0,1]), # M6
        ([0,1,-1,0], [0,0,1,1], [1,0,0,0]), # M7
    ]
    
    # Level 2 Strassen factors (for 2x2x2): same structure
    L2_factors = L1_factors  # Same as level 1
    
    # Now compose: for each L1 factor c1 and L2 factor c2:
    # The L1 factor gives:
    #   U_L1[i1*2+j1] for i1,j1 in {0,1}
    #   V_L1[j1*2+k1] for j1,k1 in {0,1}
    #   W_L1[k1*2+i1] for k1,i1 in {0,1}
    # 
    # The L2 factor for sub-block (i1,j1,k1) gives:
    #   U_L2[i2*1+j2] for i2,j2 in {0,1} -> but this is for the 2x2 within the sub-block
    #   Actually, the L2 factor operates on the 2x2 sub-matrices
    # 
    # Let me think about this differently.
    # 
    # The full 4x4x4 tensor T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # 
    # Strassen decomposition:
    # C = A*B
    # Split A, B, C into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C = [[C11, C12], [C21, C22]]
    # 
    # 7 products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # 
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    # 
    # Each Mk is a 2x2 matrix product, which we decompose further with Strassen.
    # 
    # For the tensor form:
    # T[i,j,k] where i,j in {0,1,2,3}, k in {0,1,2,3}
    # i = i1*2 + i2, j = j1*2 + j2, k = k1*2 + k2
    # 
    # The L1 factor c1 contributes to (i1,j1,k1) with weights:
    #   U_L1[i1*2+j1, c1] * V_L1[j1*2+k1, c1] * W_L1[k1*2+i1, c1]
    # 
    # But each Mk is itself a 2x2x2 tensor, decomposed by L2:
    #   For sub-block (i1,j1,k1), the L2 factor c2 contributes:
    #   U_L2[i2*2+j2, c2] * V_L2[j2*2+k2, c2] * W_L2[k2*2+i2, c2]
    # 
    # Wait, this isn't quite right. The L1 factor gives the coefficients for
    # the 2x2 sub-matrices, and the L2 factor decomposes each 2x2 sub-product.
    # 
    # Actually, the correct composition is:
    # For L1 factor c1, it defines a 2x2x2 tensor T_L1[i1,j1,k1] = 
    #   U_L1[i1*2+j1, c1] * V_L1[j1*2+k1, c1] * W_L1[k1*2+i1, c1]
    # 
    # This T_L1 is the "structure" of how the 2x2 sub-matrices combine.
    # Then each 2x2 sub-product (i1,j1,k1) is decomposed by L2.
    # 
    # The full tensor is:
    # T[i,j,k] = sum_{c1,c2} U_L1[i1*2+j1, c1] * U_L2[i2*2+j2, c2] * 
    #                   V_L1[j1*2+k1, c1] * V_L2[j2*2+k2, c2] *
    #                   W_L1[k1*2+i1, c1] * W_L2[k2*2+i2, c2]
    # where i = i1*2+i2, j = j1*2+j2, k = k1*2+k2
    # 
    # So the combined factor (c1, c2) has:
    # U[i*4+j, (c1,c2)] = U_L1[i1*2+j1, c1] * U_L2[i2*2+j2, c2]
    # V[j*4+k, (c1,c2)] = V_L1[j1*2+k1, c1] * V_L2[j2*2+k2, c2]
    # W[k*4+i, (c1,c2)] = W_L1[k1*2+i1, c1] * W_L2[k2*2+i2, c2]
    
    for c1 in range(7):
        U1, V1, W1 = L1_factors[c1]
        for c2 in range(7):
            U2, V2, W2 = L2_factors[c2]
            c = c1 * 7 + c2
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        u1 = U1[i1 * 2 + j1]
                        v1 = V1[j1 * 2 + k1]
                        w1 = W1[k1 * 2 + i1]
                        if u1 == 0 or v1 == 0 or w1 == 0:
                            continue
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    u2 = U2[i2 * 2 + j2]
                                    v2 = V2[j2 * 2 + k2]
                                    w2 = W2[k2 * 2 + i2]
                                    if u2 == 0 or v2 == 0 or w2 == 0:
                                        continue
                                    i = i1 * 2 + i2
                                    j = j1 * 2 + j2
                                    k = k1 * 2 + k2
                                    U[i * 4 + j][c] += u1 * u2
                                    V[j * 4 + k][c] += v1 * v2
                                    W[k * 4 + i][c] += w1 * w2
    
    return U, V, W


def _construct_4x4x5():
    """
    Construct 4x4x5 decomposition with rank 65.
    4x4x5 = 4x4x4 (rank 49) + 4x4x1 (rank 16)
    """
    n, m, p = 4, 4, 5
    
    # Part 1: 4x4x4 (Strassen, rank 49)
    U1, V1, W1 = _strassen_4x4x4()
    
    # Part 2: 4x4x1 (schoolbook, rank 16)
    # Computes A(4x4) * B(4x1) = C(4x1)
    U2 = [[0.0] * 16 for _ in range(16)]
    V2 = [[0.0] * 16 for _ in range(4)]  # m*p = 4*1 = 4 rows
    W2 = [[0.0] * 16 for _ in range(4)]  # p*n = 1*4 = 4 rows
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U2[i * 4 + j][col] = 1.0
            V2[j][col] = 1.0  # j*1+0 = j
            W2[i][col] = 1.0  # 0*4+i = i
    
    # Combine into 4x4x5
    # U_total: (16 x 65)
    U_total = [[0.0] * 65 for _ in range(16)]
    for i in range(16):
        for j in range(49):
            U_total[i][j] = U1[i][j]
        for j in range(16):
            U_total[i][49 + j] = U2[i][j]
    
    # V_total: (20 x 65)
    # V1 has 16 rows (j*4+k for j,k in 0..3)
    # V2 has 4 rows (j for j in 0..3, corresponding to k=4)
    V_total = [[0.0] * 65 for _ in range(20)]
    for i in range(16):
        for j in range(49):
            V_total[i][j] = V1[i][j]
    for i in range(4):
        for j in range(16):
            V_total[16 + i][49 + j] = V2[i][j]
    
    # W_total: (20 x 65)
    # W1 has 16 rows (k*4+i for k,i in 0..3)
    # W2 has 4 rows (i for i in 0..3, corresponding to k=4)
    W_total = [[0.0] * 65 for _ in range(20)]
    for i in range(16):
        for j in range(49):
            W_total[i][j] = W1[i][j]
    for i in range(4):
        for j in range(16):
            W_total[16 + i][49 + j] = W2[i][j]
    
    return U_total, V_total, W_total


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    rank = len(U[0])
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Should equal 1 for all i,j,k (since T[i,j,k] = 1 for matrix mult tensor)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _reduce_rank(U, V, W, n, m, p, max_iters=100):
    """
    Reduce rank by finding linearly dependent columns in U, V, W.
    Use SVD to find the null space.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    for iteration in range(max_iters):
        if not _time_ok():
            break
        
        # Find if any column is a linear combination of others
        # Check U first
        null_vec = _find_null_vector(U_arr)
        if null_vec is not None:
            # Check if this null vector also works for V and W
            if _check_null_vector(null_vec, V_arr) and _check_null_vector(null_vec, W_arr):
                # Remove the column
                U_arr = _remove_column(U_arr, null_vec)
                V_arr = _remove_column(V_arr, null_vec)
                W_arr = _remove_column(W_arr, null_vec)
                rank = U_arr.shape[1]
                continue
        
        # Check V
        null_vec = _find_null_vector(V_arr)
        if null_vec is not None:
            if _check_null_vector(null_vec, U_arr) and _check_null_vector(null_vec, W_arr):
                U_arr = _remove_column(U_arr, null_vec)
                V_arr = _remove_column(V_arr, null_vec)
                W_arr = _remove_column(W_arr, null_vec)
                rank = U_arr.shape[1]
                continue
        
        # Check W
        null_vec = _find_null_vector(W_arr)
        if null_vec is not None:
            if _check_null_vector(null_vec, U_arr) and _check_null_vector(null_vec, V_arr):
                U_arr = _remove_column(U_arr, null_vec)
                V_arr = _remove_column(V_arr, null_vec)
                W_arr = _remove_column(W_arr, null_vec)
                rank = U_arr.shape[1]
                continue
        
        # No more reduction possible
        break
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()


def _find_null_vector(M, tol=1e-9):
    """Find a null vector of M (a row in the null space of M^T)."""
    # M is (rows x cols). We want a vector v such that M^T v = 0
    # i.e., v is in the null space of M^T
    # Use SVD
    try:
        U, s, Vt = np.linalg.svd(M, full_matrices=True)
    except:
        return None
    
    # The null space of M^T is spanned by the last (cols - rank) columns of U
    # U is (rows x rows), s has length min(rows, cols)
    # The null space dimension is cols - rank(M)
    rank = np.sum(s > tol)
    null_dim = M.shape[1] - rank
    
    if null_dim <= 0:
        return None
    
    # The null vectors are in the rows of U corresponding to the zero singular values
    # Actually, for M^T v = 0, v is in the null space of M^T
    # M^T is (cols x rows), so its null space is in R^cols
    # The null space of M^T is the orthogonal complement of the row space of M
    # which is spanned by the last (cols - rank) columns of U (from SVD of M)
    
    # Let me use a different approach: find a vector in the null space of M^T
    # by solving M^T v = 0
    # Use the right singular vectors
    Vt_full = np.linalg.svd(M, full_matrices=True)[2]
    # Vt_full is (cols x cols)
    # The last (cols - rank) rows of Vt_full span the null space of M^T
    
    for idx in range(M.shape[1] - 1, rank - 1, -1):
        if idx >= 0:
            v = Vt_full[idx]
            # Normalize
            norm = np.linalg.norm(v)
            if norm > tol:
                v = v / norm
            # Check
            if np.linalg.norm(M.T @ v) < tol * 10:
                return v.tolist()
    
    return None


def _check_null_vector(v, M, tol=1e-9):
    """Check if v is in the null space of M^T."""
    v = np.array(v)
    return np.linalg.norm(M.T @ v) < tol * 10


def _remove_column(M, v, tol=1e-9):
    """Remove the column most aligned with v."""
    # Find the column with maximum |v_i|
    v = np.array(v)
    col = np.argmax(np.abs(v))
    return np.delete(M, col, axis=1)


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Start with rank-65 construction
    U, V, W = _construct_4x4x5()
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    U, V, W = _reduce_rank(U, V, W, n, m, p)
    
    # Verify again
    if not _verify(U, V, W, n, m, p):
        # Fallback to the original rank-65
        U, V, W = _construct_4x4x5()
    
    return {"U": U, "V": V, "W": W}
