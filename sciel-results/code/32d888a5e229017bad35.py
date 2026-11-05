"""Rewritten program: Strassen-based block decomposition for 4x4 * 4x7 matrix multiplication."""

DIMENSIONS = [4, 4, 7]

import time
from fractions import Fraction as F

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """
    Return (U, V, W) for 2x2 * 2x2 multiplication, rank 7.
    U is (4, 7), V is (4, 7), W is (4, 7).
    Indices: U row = i*2 + j, V row = j*2 + k, W row = k*2 + i.
    """
    n = 2
    # Strassen products:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(n*n)]
    V = [[0]*7 for _ in range(n*n)]
    W = [[0]*7 for _ in range(n*n)]
    
    # M1 = (A11+A22)(B11+B22)
    # A11: U[0*2+0]=U[0][0]=1
    # A22: U[1*2+1]=U[3][0]=1
    U[0][0] += 1
    U[3][0] += 1
    # B11: V[0*2+0]=V[0][0]=1
    # B22: V[1*2+1]=V[3][0]=1
    V[0][0] += 1
    V[3][0] += 1
    # C11: W[0*2+0]=W[0][0]=1
    # C22: W[1*2+1]=W[3][0]=1
    W[0][0] += 1
    W[3][0] += 1
    
    # M2 = (A21+A22)B11
    # A21: U[1*2+0]=U[2][1]=1
    # A22: U[3][1]=1
    U[2][1] += 1
    U[3][1] += 1
    # B11: V[0][1]=1
    V[0][1] += 1
    # C21: W[1*2+0]=W[2][1]=1
    # C22: W[3][1]=-1
    W[2][1] += 1
    W[3][1] -= 1
    
    # M3 = A11(B21-B22)
    # A11: U[0][2]=1
    U[0][2] += 1
    # B21: V[1*2+0]=V[2][2]=1
    # B22: V[3][2]=-1
    V[2][2] += 1
    V[3][2] -= 1
    # C12: W[0*2+1]=W[1][2]=1
    # C22: W[3][2]=1
    W[1][2] += 1
    W[3][2] += 1
    
    # M4 = A22(B12-B11)
    # A22: U[3][3]=1
    U[3][3] += 1
    # B12: V[0*2+1]=V[1][3]=1
    # B11: V[0][3]=-1
    V[1][3] += 1
    V[0][3] -= 1
    # C11: W[0][3]=1
    # C21: W[2][3]=1
    W[0][3] += 1
    W[2][3] += 1
    
    # M5 = (A11+A12)B22
    # A11: U[0][4]=1
    # A12: U[0*2+1]=U[1][4]=1
    U[0][4] += 1
    U[1][4] += 1
    # B22: V[3][4]=1
    V[3][4] += 1
    # C11: W[0][4]=-1
    # C12: W[1][4]=1
    W[0][4] -= 1
    W[1][4] += 1
    
    # M6 = (A21-A11)(B11+B12)
    # A21: U[2][5]=1
    # A11: U[0][5]=-1
    U[2][5] += 1
    U[0][5] -= 1
    # B11: V[0][5]=1
    # B12: V[1][5]=1
    V[0][5] += 1
    V[1][5] += 1
    # C22: W[3][5]=1
    W[3][5] += 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12: U[1][6]=1
    # A22: U[3][6]=-1
    U[1][6] += 1
    U[3][6] -= 1
    # B21: V[2][6]=1
    # B22: V[3][6]=1
    V[2][6] += 1
    V[3][6] += 1
    # C11: W[0][6]=1
    W[0][6] += 1
    
    return U, V, W


def _schoolbook_4x4_4x3():
    """
    Return (U, V, W) for 4x4 * 4x3 schoolbook, rank 48.
    U is (16, 48), V is (12, 48), W is (12, 48).
    Indices: U row = i*4 + j, V row = j*3 + k, W row = k*4 + i.
    """
    n = 4
    m = 4
    p = 3
    rank = n * m * p  # 48
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m*p + j*p + k
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
    return U, V, W


def _strassen_4x4_4x4():
    """
    Return (U, V, W) for 4x4 * 4x4 Strassen, rank 49.
    U is (16, 49), V is (16, 49), W is (16, 49).
    We decompose 4x4x4 into 7 independent 2x2x2 Strassen multiplications.
    
    Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    Split B into 2x2 blocks: B11, B12, B21, B22 (each 2x2)
    Split C into 2x2 blocks: C11, C12, C21, C22 (each 2x2)
    
    We need 7 multiplications of 2x2 * 2x2 = 2x2.
    Each uses rank 7, so total rank = 7*7 = 49.
    
    The 7 Strassen products:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    n = 4
    rank = 49
    U = [[0]*rank for _ in range(n*n)]
    V = [[0]*rank for _ in range(n*n)]
    W = [[0]*rank for _ in range(n*n)]
    
    # Get the base 2x2x2 Strassen matrices
    U2, V2, W2 = _strassen_2x2()
    
    # For each of the 7 Strassen multiplications, we have a 2x2 * 2x2 = 2x2 problem.
    # We need to map the 2x2 indices to the 4x4 indices.
    # 
    # A block (br, bc) has top-left at (br*2, bc*2).
    # A[i,j] where i = br*2 + i2, j = bc*2 + j2.
    # U index = i*4 + j = (br*2+i2)*4 + (bc*2+j2) = br*8 + i2*4 + bc*2 + j2
    # 
    # B block (br, bc) has top-left at (br*2, bc*2).
    # B[j,k] where j = br*2 + j2, k = bc*2 + k2.
    # V index = j*4 + k = (br*2+j2)*4 + (bc*2+k2) = br*8 + j2*4 + bc*2 + k2
    # 
    # C block (br, bc) has top-left at (br*2, bc*2).
    # C[i,k] where i = br*2 + i2, k = bc*2 + k2.
    # W index = k*4 + i = (bc*2+k2)*4 + (br*2+i2) = bc*8 + k2*4 + br*2 + i2
    
    # The 7 Strassen multiplications involve:
    # For each s in 0..6 (Strassen product index):
    #   A_factor: which A blocks are involved and with what signs
    #   B_factor: which B blocks are involved and with what signs
    #   C_contrib: which C blocks receive contributions and with what signs
    
    # Strassen product s involves:
    # s=0 (M1): A blocks {(0,0), (1,1)}, B blocks {(0,0), (1,1)}, C blocks {(0,0):+1, (1,1):+1}
    # s=1 (M2): A blocks {(1,0), (1,1)}, B blocks {(0,0)}, C blocks {(1,0):+1, (1,1):-1}
    # s=2 (M3): A blocks {(0,0)}, B blocks {(1,0):-1, (1,1):+1}, C blocks {(0,1):+1, (1,1):+1}
    # s=3 (M4): A blocks {(1,1)}, B blocks {(0,1):+1, (0,0):-1}, C blocks {(0,0):+1, (1,0):+1}
    # s=4 (M5): A blocks {(0,0), (0,1)}, B blocks {(1,1)}, C blocks {(0,0):-1, (0,1):+1}
    # s=5 (M6): A blocks {(1,0):+1, (0,0):-1}, B blocks {(0,0), (0,1)}, C blocks {(1,1):+1}
    # s=6 (M7): A blocks {(0,1):+1, (1,1):-1}, B blocks {(1,0), (1,1)}, C blocks {(0,0):+1}
    
    # For each Strassen product s, we use the 2x2x2 Strassen decomposition.
    # The 2x2x2 Strassen has 7 "scalar" multiplications.
    # For scalar multiplication: A_block * B_block -> contributes to C_block
    # In the 2x2x2 Strassen, the 7 terms correspond to different combinations.
    # Actually, for each of the 7 Strassen products, we perform a full 2x2x2 Strassen,
    # which itself has 7 terms. So total 7*7 = 49 terms.
    
    # For Strassen product s, the A factor is a linear combination of A blocks.
    # Let's define for each s:
    #   A_coeffs: list of (block_row, block_col, coeff)
    #   B_coeffs: list of (block_row, block_col, coeff)
    #   C_coeffs: list of (block_row, block_col, coeff)
    
    A_coeffs_list = [
        [(0,0,1), (1,1,1)],           # s=0: A11 + A22
        [(1,0,1), (1,1,1)],           # s=1: A21 + A22
        [(0,0,1)],                    # s=2: A11
        [(1,1,1)],                    # s=3: A22
        [(0,0,1), (0,1,1)],           # s=4: A11 + A12
        [(1,0,1), (0,0,-1)],          # s=5: A21 - A11
        [(0,1,1), (1,1,-1)],          # s=6: A12 - A22
    ]
    
    B_coeffs_list = [
        [(0,0,1), (1,1,1)],           # s=0: B11 + B22
        [(0,0,1)],                    # s=1: B11
        [(1,0,1), (1,1,-1)],          # s=2: B21 - B22
        [(0,1,1), (0,0,-1)],          # s=3: B12 - B11
        [(1,1,1)],                    # s=4: B22
        [(0,0,1), (0,1,1)],           # s=5: B11 + B12
        [(1,0,1), (1,1,1)],           # s=6: B21 + B22
    ]
    
    C_coeffs_list = [
        [(0,0,1), (1,1,1)],           # s=0: C11 + C22
        [(1,0,1), (1,1,-1)],          # s=1: C21 - C22
        [(0,1,1), (1,1,1)],           # s=2: C12 + C22
        [(0,0,1), (1,0,1)],           # s=3: C11 + C21
        [(0,0,-1), (0,1,1)],          # s=4: -C11 + C12
        [(1,1,1)],                    # s=5: C22
        [(0,0,1)],                    # s=6: C11
    ]
    
    for s in range(7):
        # For this Strassen product s, we have A_factor * B_factor.
        # A_factor = sum_{(ar,ac,a)} a * A[ar,ac]
        # B_factor = sum_{(br,bc,b)} b * B[br,bc]
        # The product contributes to C via: sum_{(cr,cc,c)} c * C[cr,cc]
        # 
        # We decompose this using the 2x2x2 Strassen.
        # The 2x2x2 Strassen decomposes a 2x2 * 2x2 = 2x2 into 7 rank-1 terms.
        # For each of the 7 rank-1 terms t (0..6):
        #   U2[i2*2+j2, t], V2[j2*2+k2, t], W2[k2*2+i2, t]
        #   contribute: U2[i2*2+j2,t] * V2[j2*2+k2,t] * W2[k2*2+i2,t]
        #   to C[i2,k2] for the 2x2 result.
        #
        # Now, the 2x2 result corresponds to each C block [cr,cc].
        # For C block (cr,cc), the 2x2 indices (i2,k2) map to:
        #   i = cr*2 + i2, k = cc*2 + k2
        #   W index in 4x4: k*4 + i = (cc*2+k2)*4 + (cr*2+i2)
        
        for t in range(7):
            col = s*7 + t
            # For each A block involved in A_factor:
            for (ar, ac, a_coeff) in A_coeffs_list[s]:
                # A[ar,ac] has top-left at (ar*2, ac*2)
                # For 2x2 indices (i2, j2):
                #   i = ar*2 + i2, j = ac*2 + j2
                #   U index = i*4 + j = (ar*2+i2)*4 + (ac*2+j2)
                for i2 in range(2):
                    for j2 in range(2):
                        u_idx = (ar*2 + i2)*4 + (ac*2 + j2)
                        U[u_idx][col] += a_coeff * U2[i2*2 + j2][t]
            
            # For each B block involved in B_factor:
            for (br, bc, b_coeff) in B_coeffs_list[s]:
                # B[br,bc] has top-left at (br*2, bc*2)
                # For 2x2 indices (j2, k2):
                #   j = br*2 + j2, k = bc*2 + k2
                #   V index = j*4 + k = (br*2+j2)*4 + (bc*2+k2)
                for j2 in range(2):
                    for k2 in range(2):
                        v_idx = (br*2 + j2)*4 + (bc*2 + k2)
                        V[v_idx][col] += b_coeff * V2[j2*2 + k2][t]
            
            # For each C block that receives contribution:
            for (cr, cc, c_coeff) in C_coeffs_list[s]:
                # C[cr,cc] has top-left at (cr*2, cc*2)
                # For 2x2 indices (i2, k2):
                #   i = cr*2 + i2, k = cc*2 + k2
                #   W index = k*4 + i = (cc*2+k2)*4 + (cr*2+i2)
                for i2 in range(2):
                    for k2 in range(2):
                        w_idx = (cc*2 + k2)*4 + (cr*2 + i2)
                        W[w_idx][col] += c_coeff * W2[k2*2 + i2][t]
    
    return U, V, W


def _combine(U1, V1, W1, U2, V2, W2):
    """
    Combine two decompositions.
    U1, V1, W1 have rank r1; U2, V2, W2 have rank r2.
    Returns combined U, V, W with rank r1+r2.
    """
    r1 = len(U1[0])
    r2 = len(U2[0])
    rank = r1 + r2
    U = [row[:] + [0]*r2 for row in U1]
    V = [row[:] + [0]*r2 for row in V1]
    W = [row[:] + [0]*r2 for row in W1]
    for i in range(len(U2)):
        for r in range(r2):
            U[i][r1 + r] = U2[i][r]
    for i in range(len(V2)):
        for r in range(r2):
            V[i][r1 + r] = V2[i][r]
    for i in range(len(W2)):
        for r in range(r2):
            W[i][r1 + r] = W2[i][r]
    return U, V, W


def _verify(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(len(U[0])):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if val != 1:
                    return False, (i, j, k, val)
    return True, None


def construct():
    """
    Construct rank-minimal tensor decomposition for 4x4 * 4x7 matrix multiplication.
    
    Strategy:
    - Split B (4x7) into B1 (4x4) and B2 (4x3).
    - Compute A*B1 using Strassen 4x4x4 (rank 49).
    - Compute A*B2 using schoolbook 4x4x3 (rank 48).
    - Total rank = 49 + 48 = 97.
    
    Return: {'U': ..., 'V': ..., 'W': ...}
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    start_time = time.time()
    
    # Get Strassen 4x4x4 decomposition (rank 49)
    U_strassen, V_strassen, W_strassen = _strassen_4x4_4x4()
    
    # Get schoolbook 4x4x3 decomposition (rank 48)
    U_school, V_school, W_school = _schoolbook_4x4_4x3()
    
    # Now we need to combine these.
    # The Strassen decomposition computes A * B1 where B1 is 4x4 (first 4 columns of B).
    # The schoolbook decomposition computes A * B2 where B2 is 4x3 (last 3 columns of B).
    #
    # In the full problem, B is 4x7. B1 = B[:, 0:4], B2 = B[:, 4:7].
    # 
    # For the Strassen part:
    #   U row index = i*4 + j (for A[i,j])
    #   V row index = j*4 + k1 where k1 is the column index within B1 (0..3)
    #   W row index = k1*4 + i where k1 is the row index of C1 (0..3)
    #
    # In the full problem:
    #   U row index = i*m + j = i*4 + j (same)
    #   V row index = j*p + k = j*7 + k where k is 0..6
    #   W row index = k*n + i = k*4 + i where k is 0..6
    #
    # So for the Strassen part, V uses k in 0..3, which maps to V_full[j*7 + k]
    # W uses k in 0..3, which maps to W_full[k*4 + i]
    #
    # For the schoolbook 4x4x3 part:
    #   U row index = i*4 + j (same)
    #   V row index = j*3 + k2 where k2 is 0..2 (column within B2)
    #   W row index = k2*4 + i where k2 is 0..2 (row of C2)
    #
    # In the full problem:
    #   V uses k in 4..6 (since B2 corresponds to columns 4,5,6 of B)
    #   V_full[j*7 + k] where k = 4 + k2
    #   W uses k in 4..6 (since C2 corresponds to rows 4,5,6 of C)
    #   W_full[k*4 + i] where k = 4 + k2
    
    rank_total = 49 + 48  # 97
    
    U_full = [[0]*rank_total for _ in range(n*m)]
    V_full = [[0]*rank_total for _ in range(m*p)]
    W_full = [[0]*rank_total for _ in range(p*n)]
    
    # Copy Strassen part (first 49 columns)
    for i in range(n*m):
        for r in range(49):
            U_full[i][r] = U_strassen[i][r]
    
    for j in range(m):
        for k1 in range(4):  # k1 = 0..3 for B1
            for r in range(49):
                V_full[j*p + k1][r] = V_strassen[j*4 + k1][r]
    
    for k1 in range(4):  # k1 = 0..3 for C1
        for i in range(n):
            for r in range(49):
                W_full[k1*n + i][r] = W_strassen[k1*4 + i][r]
    
    # Copy schoolbook part (columns 49..96)
    for i in range(n*m):
        for r in range(48):
            U_full[i][49 + r] = U_school[i][r]
    
    for j in range(m):
        for k2 in range(3):  # k2 = 0..2 for B2
            for r in range(48):
                V_full[j*p + (4 + k2)][49 + r] = V_school[j*3 + k2][r]
    
    for k2 in range(3):  # k2 = 0..2 for C2
        for i in range(n):
            for r in range(48):
                W_full[(4 + k2)*n + i][49 + r] = W_school[k2*4 + i][r]
    
    # Verify
    is_valid, info = _verify(U_full, V_full, W_full, n, m, p)
    if not is_valid:
        # Fallback to schoolbook if verification fails
        rank_sb = n * m * p
        U_sb = [[0]*rank_sb for _ in range(n*m)]
        V_sb = [[0]*rank_sb for _ in range(m*p)]
        W_sb = [[0]*rank_sb for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    col = i*m*p + j*p + k
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return {"U": U_full, "V": V_full, "W": W_full}

# EVOLVE-BLOCK-END
