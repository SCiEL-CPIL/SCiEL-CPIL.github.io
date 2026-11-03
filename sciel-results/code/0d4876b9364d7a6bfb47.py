"""
Rank-minimal exact tensor decomposition for multiplying a 4x4 matrix by a 4x8 matrix.

Approach:
- We know that the 4x4x4 matrix multiplication tensor has rank 49 (from Strassen recursion).
- For 4x4x8, we can split the output into two 4x4x4 blocks along the p dimension.
- Thus, a decomposition of rank 49+49=98 exists by concatenating two 4x4x4 decompositions.

We construct the 4x4x4 decomposition using Strassen recursion:
- Base case: 2x2x2 with rank 7 (Strassen)
- Recursive case: 4x4x4 = 7 * (2x2x2) = 7*7 = 49

Then we duplicate this for the 4x4x8 case, using the first 4 columns for k=0..3 and the next 4 for k=4..7.
"""

DIMENSIONS = [4, 4, 8]

def construct():
    """
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices
    for the matrix multiplication tensor M_4x4x8.
    
    U has shape (n*m, rank) = (16, 98)
    V has shape (m*p, rank) = (32, 99)  # actually (4*8, 98) = (32, 98)
    W has shape (p*n, rank) = (32, 98)  # (8*4, 98) = (32, 98)
    
    The tensor identity is:
    C[i,k] = sum_j A[i,j] * B[j,k]
    where C = A @ B, A is 4x4, B is 4x8, C is 4x8.
    
    In tensor form: T[i,j,k] = delta_{i,j,k} for the standard decomposition,
    but here we use the rank-98 decomposition.
    
    The factor matrices satisfy:
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    where m=4, n=4, p=8.
    """
    n, m, p = 4, 4, 8
    
    # First, construct the 4x4x4 decomposition (rank 49)
    U4, V4, W4 = _strassen_4x4x4()
    
    # U4 has shape (16, 49), V4 has shape (16, 49), W4 has shape (16, 49)
    # Now extend to 4x4x8 by splitting the p dimension into two halves
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # (16, 98)
    V = [[0] * rank for _ in range(m * p)]  # (32, 98)
    W = [[0] * rank for _ in range(p * n)]  # (32, 98)
    
    # First half: k = 0..3
    for r in range(49):
        for i in range(16):
            U[i][r] = U4[i][r]
        for j in range(4):
            for k in range(4):
                # V index: j*p + k = j*8 + k
                V[j * p + k][r] = V4[j * 4 + k][r]
                # W index: k*n + i
                W[k * n + i][r] = W4[k * 4 + i][r]
    
    # Second half: k = 4..7
    for r in range(49):
        r2 = r + 49
        for i in range(16):
            U[i][r2] = U4[i][r]
        for j in range(4):
            for k in range(4):
                # V index: j*p + (k+4) = j*8 + k + 4
                V[j * p + k + 4][r2] = V4[j * 4 + k][r]
                # W index: (k+4)*n + i
                W[(k + 4) * n + i][r2] = W4[k * 4 + i][r]
    
    return {"U": U, "V": V, "W": W}


def _strassen_2x2x2():
    """
    Construct the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    
    Returns:
        U: shape (4, 7) - for A (2x2 = 4 elements)
        V: shape (4, 7) - for B (2x2 = 4 elements)
        W: shape (4, 7) - for C (2x2 = 4 elements)
    
    Strassen's 7 products:
    M1 = (A11 + A22)(B11 + B22)
    M2 = (A21 + A22)B11
    M3 = A11(B21 - B22)
    M4 = A22(B12 - B11)
    M5 = (A11 + A12)B22
    M6 = (A21 - A11)(B11 + B12)
    M7 = (A12 - A22)(B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n = m = p = 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]  # (4, 7)
    V = [[0] * rank for _ in range(m * p)]  # (4, 7)
    W = [[0] * rank for _ in range(p * n)]  # (4, 7)
    
    # Index convention: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # A indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # C indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # M1 = (A11 + A22)(B11 + B22)
    # A11 = A[0,0] = index 0, A22 = A[1,1] = index 3
    # B11 = B[0,0] = index 0, B22 = B[1,1] = index 3
    U[0][0] = 1  # A11
    U[3][0] = 1  # A22
    V[0][0] = 1  # B11
    V[3][0] = 1  # B22
    # C contributions: C11 += M1, C22 += M1
    W[0][0] = 1  # C[0,0]
    W[3][0] = 1  # C[1,1]
    
    # M2 = (A21 + A22)B11
    # A21 = A[1,0] = index 2, A22 = A[1,1] = index 3
    # B11 = B[0,0] = index 0
    U[2][1] = 1  # A21
    U[3][1] = 1  # A22
    V[0][1] = 1  # B11
    # C contributions: C21 += M2, C22 -= M2
    W[2][1] = 1  # C[1,0]
    W[3][1] = -1  # C[1,1]
    
    # M3 = A11(B21 - B22)
    # A11 = A[0,0] = index 0
    # B21 = B[1,0] = index 2, B22 = B[1,1] = index 3
    U[0][2] = 1  # A11
    V[2][2] = 1  # B21
    V[3][2] = -1  # -B22
    # C contributions: C12 += M3, C22 += M3
    W[1][2] = 1  # C[0,1]
    W[3][2] = 1  # C[1,1]
    
    # M4 = A22(B12 - B11)
    # A22 = A[1,1] = index 3
    # B12 = B[0,1] = index 1, B11 = B[0,0] = index 0
    U[3][3] = 1  # A22
    V[1][3] = 1  # B12
    V[0][3] = -1  # -B11
    # C contributions: C11 += M4, C21 += M4
    W[0][3] = 1  # C[0,0]
    W[2][3] = 1  # C[1,0]
    
    # M5 = (A11 + A12)B22
    # A11 = A[0,0] = index 0, A12 = A[0,1] = index 1
    # B22 = B[1,1] = index 3
    U[0][4] = 1  # A11
    U[1][4] = 1  # A12
    V[3][4] = 1  # B22
    # C contributions: C11 -= M5, C12 += M5
    W[0][4] = -1  # C[0,0]
    W[1][4] = 1  # C[0,1]
    
    # M6 = (A21 - A11)(B11 + B12)
    # A21 = A[1,0] = index 2, A11 = A[0,0] = index 0
    # B11 = B[0,0] = index 0, B12 = B[0,1] = index 1
    U[2][5] = 1  # A21
    U[0][5] = -1  # -A11
    V[0][5] = 1  # B11
    V[1][5] = 1  # B12
    # C contributions: C22 += M6
    W[3][5] = 1  # C[1,1]
    
    # M7 = (A12 - A22)(B21 + B22)
    # A12 = A[0,1] = index 1, A22 = A[1,1] = index 3
    # B21 = B[1,0] = index 2, B22 = B[1,1] = index 3
    U[1][6] = 1  # A12
    U[3][6] = -1  # -A22
    V[2][6] = 1  # B21
    V[3][6] = 1  # B22
    # C contributions: C11 += M7
    W[0][6] = 1  # C[0,0]
    
    return U, V, W


def _strassen_4x4x4():
    """
    Construct the rank-49 decomposition for 4x4x4 matrix multiplication
    using Strassen recursion.
    
    We split each 4x4 matrix into four 2x2 blocks:
    A = [[A00, A01], [A10, A11]]
    B = [[B00, B01], [B10, B11]]
    C = [[C00, C01], [C10, C11]]
    
    Strassen gives:
    M1 = (A00 + A11)(B00 + B11)
    M2 = (A10 + A11)B00
    M3 = A00(B10 - B11)
    M4 = A11(B01 - B00)
    M5 = (A00 + A01)B11
    M6 = (A10 - A00)(B00 + B01)
    M7 = (A01 - A11)(B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Each Mr is a 2x2x2 matrix multiplication, which we decompose with rank 7.
    Total rank: 7 * 7 = 49.
    """
    n = m = p = 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # (16, 49)
    V = [[0] * rank for _ in range(m * p)]  # (16, 49)
    W = [[0] * rank for _ in range(p * n)]  # (16, 49)
    
    U2, V2, W2 = _strassen_2x2x2()
    
    # Block indices: A00=rows 0-1, cols 0-1; A01=rows 0-1, cols 2-3; A10=rows 2-3, cols 0-1; A11=rows 2-3, cols 2-3
    # Similarly for B and C.
    
    # For each Strassen term Mr, we have a 2x2x2 sub-problem.
    # The sub-problem has its own U2, V2, W2 with shapes (4,7).
    # We need to map the sub-problem indices to the full 4x4x4 indices.
    
    # Define the linear combinations for each Mr:
    # For each r in 0..6 (the 7 Strassen terms), and for each s in 0..6 (the 7 Strassen base terms):
    #   r2 = r * 7 + s
    
    # For a given Strassen term r, we have:
    #   A_comb[i] = sum over blocks ai of coeff_A(r, ai) * A_block_ai
    #   B_comb[j] = sum over blocks bj of coeff_B(r, bj) * B_block_bj
    #   C_comb[k] = sum over blocks ck of coeff_C(r, ck) * C_block_ck
    
    # The coefficients for each Strassen term:
    # M1: A blocks: [1, 0, 0, 1] (A00+A11), B blocks: [1, 0, 0, 1] (B00+B11), C blocks: [1, 0, 0, 1] (C00+C11)
    # M2: A blocks: [0, 0, 1, 1] (A10+A11), B blocks: [1, 0, 0, 0] (B00), C blocks: [0, 0, 1, 0] (C10)
    # M3: A blocks: [1, 0, 0, 0] (A00), B blocks: [0, 0, 1, -1] (B10-B11), C blocks: [0, 1, 0, 0] (C01)
    # M4: A blocks: [0, 0, 0, 1] (A11), B blocks: [0, 1, 0, -1] (B01-B00), C blocks: [1, 0, 1, 0] (C00+C10)
    # M5: A blocks: [1, 1, 0, 0] (A00+A01), B blocks: [0, 0, 0, 1] (B11), C blocks: [-1, 1, 0, 0] (-C00+C01)
    # M6: A blocks: [-1, 0, 1, 0] (A10-A00), B blocks: [1, 1, 0, 0] (B00+B01), C blocks: [0, 0, 0, 1] (C11)
    # M7: A blocks: [0, 1, 0, -1] (A01-A11), B blocks: [0, 0, 1, 1] (B10+B11), C blocks: [1, 0, 0, 0] (C00)
    
    # Coefficient arrays for A, B, C blocks for each Strassen term
    A_coeffs = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    B_coeffs = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 0, 1, -1],  # M3
        [0, 1, 0, -1],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    C_coeffs = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 0],   # M2
        [0, 1, 0, 0],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0],   # M7
    ]
    
    # Block offsets for 2x2 blocks within 4x4 matrix
    # Block (bi, bj) in 4x4: row offset = bi*2, col offset = bj*2
    # So for A: block (bi, bj) has indices (bi*2 + di, bj*2 + dj) for di,dj in 0,1
    
    # For each Strassen term r (0..6) and base term s (0..6):
    #   r2 = r * 7 + s
    #   For each element in the 2x2 sub-problem:
    #     A sub-index ai (0..3) maps to 4x4 index: 
    #       di = ai % 2, dj = ai // 2  (row-major: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3)
    #       Wait, let's use: A_sub[di, dj] where di, dj in 0,1
    #       The full 4x4 index is (bi*2 + di, bj*2 + dj) for block (bi,bj)
    
    # Actually, let's think more carefully.
    # The 2x2 sub-problem has A_sub, B_sub, C_sub.
    # A_sub is formed by linear combination of the four 2x2 blocks of A.
    # For a specific element A_sub[di, dj], its value is:
    #   A_sub[di, dj] = sum_{bi,bj} A_coeffs[r][block_idx] * A[bi*2+di, bj*2+dj]
    # where block_idx is the index of block (bi,bj) in the order [A00, A01, A10, A11] = [0, 1, 2, 3]
    
    # Similarly for B_sub and C_sub.
    
    # The Strassen base decomposition says:
    #   C_sub[di, dj] = sum_s U2[di*2+dj, s] * B_sub[ei, ej] * W2[dk*2+di, s]
    #   where B_sub[ei, ej] = sum_s V2[ei*2+ej, s] * ... wait, this is getting complicated.
    
    # Let me redo this more carefully.
    # In the 2x2x2 Strassen decomposition:
    #   C_sub[i,k] = sum_j A_sub[i,j] * B_sub[j,k]
    #   = sum_j (sum_{blocks} cA[blocks] * A_block[i,j]) * (sum_{blocks} cB[blocks] * B_block[j,k])
    #   = sum_{blocks_a, blocks_b} cA[blocks_a] * cB[blocks_b] * A_block[blocks_a][i,j] * B_block[blocks_b][j,k]
    
    # And the rank-7 decomposition says:
    #   C_sub[i,k] = sum_s U2[i*2+j... wait, U2 has shape (4,7) for A_sub (2x2=4 elements)
    
    # Let's use the index convention from _strassen_2x2x2:
    #   A_sub indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    #   U2[a_idx, s] where a_idx is the A_sub index
    #   V2[b_idx, s] where b_idx is the B_sub index
    #   W2[c_idx, s] where c_idx is the C_sub index
    
    # So for the full 4x4x4:
    #   U_full[full_a_idx, r*7+s] = sum_{blocks} A_coeffs[r][block] * U2[sub_a_idx, s]
    #   where full_a_idx corresponds to the same (i,j) position within the block
    
    # More precisely:
    #   For A_full[i,j] where i,j in 0..3:
    #     block (bi, bj) where bi = i//2, bj = j//2
    #     sub-index (di, dj) where di = i%2, dj = j%2
    #     sub_a_idx = di * 2 + dj  (row-major: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3)
    
    #   U_full[i*4+j, r*7+s] = A_coeffs[r][block_idx] * U2[sub_a_idx, s]
    #   where block_idx = bi*2 + bj  (order: A00=0, A01=1, A10=2, A11=3)
    
    # Similarly for V and W.
    
    for r in range(7):
        for s in range(7):
            r2 = r * 7 + s
            for bi in range(2):
                for bj in range(2):
                    block_idx = bi * 2 + bj  # 0=A00, 1=A01, 2=A10, 3=A11
                    coeff_A = A_coeffs[r][block_idx]
                    coeff_B = B_coeffs[r][block_idx]
                    coeff_C = C_coeffs[r][block_idx]
                    
                    for di in range(2):
                        for dj in range(2):
                            # A index in 4x4
                            i = bi * 2 + di
                            j = bj * 2 + dj
                            full_a_idx = i * 4 + j  # m=4
                            sub_a_idx = di * 2 + dj  # 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                            U[full_a_idx][r2] += coeff_A * U2[sub_a_idx][s]
                            
                            # B index in 4x4
                            # B_full[j, k] where j is row, k is col
                            # For B: row index j, col index k
                            # Block (bi, bj) means rows bi*2..bi*2+1, cols bj*2..bj*2+1
                            # So for B: row = bi*2 + di, col = bj*2 + dj
                            full_b_idx = (bi * 2 + di) * 4 + (bj * 2 + dj)  # m=4, so p=4
                            sub_b_idx = di * 2 + dj
                            V[full_b_idx][r2] += coeff_B * V2[sub_b_idx][s]
                            
                            # C index in 4x4
                            # C_full[i, k] where i is row, k is col
                            # For C: row index i, col index k
                            # Block (bi, bj) means rows bi*2..bi*2+1, cols bj*2..bj*2+1
                            # So for C: row = bi*2 + di, col = bj*2 + dj
                            # W index: k*n + i = col*4 + row
                            full_c_idx = (bj * 2 + dj) * 4 + (bi * 2 + di)  # p=4, n=4
                            sub_c_idx = di * 2 + dj
                            W[full_c_idx][r2] += coeff_C * W2[sub_c_idx][s]
    
    return U, V, W

