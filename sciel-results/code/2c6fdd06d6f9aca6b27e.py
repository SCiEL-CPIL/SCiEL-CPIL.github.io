import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Decompose 2x5x6 into 2x1x6 + 2x4x6
    # 2x1x6: rank = 2*1*6 = 12 (schoolbook)
    # 2x4x6: decompose into 2x2x6 + 2x2x6
    #   2x2x6: rank = 3 (Strassen) * 6 = 18? No, Strassen for 2x2x2 is rank 7
    #   Actually 2x2x6 can be done with rank 7*6/2? No.
    #   Let's use the known result: 2x2x6 rank is 18? No.
    #   
    # From the inspiration: "2x4x6 (rank 42 via 2x2x6 blocks). 2x2x6 uses 3 Strassen instances (rank 21)"
    # So 2x2x6 rank = 21? That seems high. Schoolbook 2x2x6 = 24.
    # Wait, Strassen 2x2x2 has rank 7.
    # 2x2x6 can be seen as 3 instances of 2x2x2? No, 6 = 2+2+2? 
    # Or maybe 2x2x6 is decomposed differently.
    #
    # Let's just use the schoolbook for now to get a valid baseline, 
    # then try to optimize.
    #
    # Actually, the inspiration says:
    # "Decomposes 2x5x6 into 2x1x6 (rank 12) and 2x4x6 (rank 42 via 2x2x6 blocks). 2x2x6 uses 3 Strassen instances (rank 21)."
    # Total rank = 12 + 42 = 54.
    #
    # Let's verify: 2x2x6 schoolbook = 24. If we use 3 Strassen instances... 
    # Maybe 2x2x6 is split into 2x2x2 + 2x2x2 + 2x2x2? That would be 3*7 = 21.
    # Yes! 6 columns = 2+2+2. So we split the 6 columns into three groups of 2.
    # Each 2x2x2 block uses Strassen (rank 7).
    # So 2x2x6 rank = 21.
    #
    # Similarly, 2x4x6 = 2x2x6 + 2x2x6 = 21 + 21 = 42.
    #
    # Total rank = 12 + 42 = 54.

    # Build the decomposition
    # Part 1: 2x1x6 (rows 0, cols 0)
    # Part 2: 2x4x6 (rows 0, cols 1-4) -> split into 2x2x6 (cols 1-2) + 2x2x6 (cols 3-4)
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Helper: add schoolbook block for n x m x p
    def add_schoolbook(n, m, p, i_offset, j_offset, k_offset):
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[(i_offset + i) * m + (j_offset + j) - j_offset * 0, col] = 1  # This is wrong, need to map to global indices
                    # Actually, let's just build the full matrices directly.
                    col += 1
        return None
    
    # Let's build the full U, V, W matrices directly.
    # Global dimensions: U is (2*5, rank), V is (5*6, rank), W is (6*2, rank)
    # U index: i*m + j where i in [0,2), j in [0,5)
    # V index: j*p + k where j in [0,5), k in [0,6)
    # W index: k*n + i where k in [0,6), i in [0,2)
    
    # Part 1: 2x1x6
    # i in [0,2), j in [0,1), k in [0,6)
    rank_part1 = 2 * 1 * 6
    U1 = np.zeros((n * m, rank_part1))
    V1 = np.zeros((m * p, rank_part1))
    W1 = np.zeros((p * n, rank_part1))
    
    col = 0
    for i in range(2):
        for j in range(1):
            for k in range(6):
                U1[i * m + j, col] = 1
                V1[j * p + k, col] = 1
                W1[k * n + i, col] = 1
                col += 1
    
    # Part 2: 2x4x6
    # Split into two 2x2x6 blocks
    # Block 2a: i in [0,2), j in [1,3), k in [0,6)
    # Block 2b: i in [0,2), j in [3,5), k in [0,6)
    
    # For each 2x2x6, we split k into three groups of 2: k in [0,2), [2,4), [4,6)
    # Each 2x2x2 uses Strassen (rank 7)
    
    def strassen_2x2x2(U_base, V_base, W_base, i_offset, j_offset, k_offset, col_start):
        # Strassen decomposition for 2x2x2
        # C11 = A11*(B11-B22)
        # C12 = A11*B12 + A12*B22
        # C21 = A21*B11 + A22*B22
        # C22 = A22*(B11-B22)
        #
        # Actually, the standard Strassen is:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # This gives rank 7.
        #
        # Let's define the 7 terms:
        # Term 1: (A11+A22)(B11+B22)
        # Term 2: (A21+A22)B11
        # Term 3: A11(B12-B22)
        # Term 4: A22(B21-B11)
        # Term 5: (A11+A12)B22
        # Term 6: (A21-A11)(B11+B12)
        # Term 7: (A12-A22)(B21+B22)
        
        # Indices in the 2x2x2 block:
        # A indices: (0,0), (0,1), (1,0), (1,1) -> global (i_offset+0, j_offset+0), etc.
        # B indices: (0,0), (0,1), (1,0), (1,1) -> global (j_offset+0, k_offset+0), etc.
        # C indices: (0,0), (0,1), (1,0), (1,1) -> global (k_offset+0, i_offset+0), etc.
        
        # We need to add 7 terms to U, V, W
        # Each term is a rank-1 tensor: U[i,j] * V[j,k] * W[k,i]
        
        # Term 1: (A11+A22)(B11+B22)
        # U contributions: A11=1, A22=1 -> U[(i_offset+0)*m+(j_offset+0)] += 1, U[(i_offset+1)*m+(j_offset+1)] += 1
        # V contributions: B11=1, B22=1 -> V[(j_offset+0)*p+(k_offset+0)] += 1, V[(j_offset+1)*p+(k_offset+1)] += 1
        # W contributions: C11=1, C12=1, C21=1, C22=1? No, this term contributes to all C's with coefficients.
        #
        # Actually, the decomposition is:
        # C = sum_c U_c * V_c * W_c
        # where U_c is a 2x2 matrix, V_c is a 2x2 matrix, W_c is a 2x2 matrix.
        #
        # For Term 1:
        # U_1 = [[1, 0], [0, 1]]  (A11 + A22)
        # V_1 = [[1, 1], [1, 1]]  (B11 + B22? No, B11+B22 is not a matrix product)
        #
        # Wait, Strassen works by pre-computing products. The tensor decomposition is:
        # C_ij = sum_c U_ic * V_jc * W_kc? No, that's not right either.
        #
        # The matrix multiplication tensor is:
        # T_{ijk} = sum_c U_{ic} V_{jc} W_{kc}
        # where i is row of C, j is col of C, k is... wait.
        #
        # Actually, the standard form is:
        # C = A * B
        # C_{ij} = sum_k A_{ik} B_{kj}
        #
        # The tensor is T_{ijk} = A_{ik} B_{kj} C_{ji}? No.
        #
        # Let's use the convention from the task:
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        #
        # For matrix multiplication, T[i,j,k] = 1 if the indices match in the multiplication.
        # Specifically, C_{ij} = sum_k A_{ik} B_{kj}
        # So the tensor is T[i,j,k] = A_{ik} B_{kj} C_{ji}? 
        #
        # Actually, the tensor decomposition of matrix multiplication is:
        # sum_c U_c * V_c * W_c = T
        # where T_{ijk} = delta_{ik} delta_{kj}? No.
        #
        # Let's just use the schoolbook for now and return a valid answer.
        # We can optimize later.
        return col_start + 7  # 7 terms
    
    # For now, let's just use schoolbook for everything to ensure correctness.
    # Total rank = 60 (schoolbook)
    
    U_sb = np.zeros((n * m, n * m * p))
    V_sb = np.zeros((m * p, n * m * p))
    W_sb = np.zeros((p * n, n * m * p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    return {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
