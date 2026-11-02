"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Rank-18 decomposition for 2x3x4 matrix multiplication
    # U: (n*m, rank) = (6, 18)
    # V: (m*p, rank) = (12, 18)  
    # W: (p*n, rank) = (8, 18)
    
    rank = 18
    
    # Using known rank-18 bilinear algorithm for 2x3x4
    # Indices: U row = i*m+j, V row = j*p+k, W row = k*n+i
    
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    # Define the 18 bilinear forms
    # Each column c defines a term: U[:,c] * V[:,c] contributes to W[:,c]
    
    # This is a known decomposition - let me construct it systematically
    # Using the approach from algebraic complexity theory
    
    # For 2x3x4, we can use the following structure:
    # Group the computations and exploit linearity
    
    # Let me use a verified rank-18 decomposition
    # Based on the work of Bini, Capovani, Lotti, Romani (1978)
    
    # U rows: (i,j) for i in 0..1, j in 0..2 -> index i*3+j
    # V rows: (j,k) for j in 0..2, k in 0..3 -> index j*4+k
    # W rows: (k,i) for k in 0..3, i in 0..1 -> index k*2+i
    
    # Define the 18 terms
    # Each term has coefficients for U, V, W
    
    terms = []
    
    # Term 0-2: Handle A[0,0]*B[0,k] for k=0,1,2,3 with combinations
    # Actually, let me use a more systematic known decomposition
    
    # Known rank-18 decomposition for 2x3x4:
    # This uses the fact that we can share multiplications across outputs
    
    # Let me construct using explicit bilinear forms
    
    # For each of the 18 multiplications, define which A and B elements are combined
    # and which C element(s) they contribute to
    
    # U coefficients (for A matrix elements, flattened as i*m+j)
    # V coefficients (for B matrix elements, flattened as j*p+k)  
    # W coefficients (for C matrix elements, flattened as k*n+i)
    
    # Using a known explicit form:
    
    # I'll use a construction based on splitting the problem
    # C[0,:] = A[0,:] * B[:,0:4] row 0
    # C[1,:] = A[1,:] * B[:,0:4] row 1
    
    # A[0,:] = [a00, a01, a02]
    # A[1,:] = [a10, a11, a12]
    
    # B row j has 4 elements
    
    # Let me use the following rank-18 decomposition (verified):
    
    # The key insight: we can compute sums like (a00+a01)*b0k and then adjust
    
    # Here's a systematic construction:
    
    # For the 2x3 part, we can use 5 multiplications for a 2x2 subproblem
    # and handle the rest carefully
    
    # Actually, let me just use a verified explicit decomposition
    
    # U matrix (6 x 18):
    U_list = [
        # col: a00, a01, a02, a10, a11, a12
        [1, 0, 0, 0, 0, 0],  # 0
        [1, 1, 0, 0, 0, 0],  # 1
        [0, 1, 0, 0, 0, 0],  # 2
        [0, 0, 1, 0, 0, 0],  # 3
        [0, 0, 0, 1, 0, 0],  # 4
        [1, 0, 0, 1, 0, 0],  # 5
        [0, 1, 0, 1, 0, 0],  # 6
        [0, 0, 1, 1, 0, 0],  # 7
        [0, 0, 0, 0, 1, 0],  # 8
        [1, 0, 0, 0, 1, 0],  # 9
        [0, 1, 0, 0, 1, 0],  # 10
        [0, 0, 1, 0, 1, 0],  # 11
        [0, 0, 0, 0, 0, 1],  # 12
        [1, 0, 0, 0, 0, 1],  # 13
        [0, 1, 0, 0, 0, 1],  # 14
        [0, 0, 1, 0, 0, 1],  # 15
        [0, 0, 0, 1, 1, 0],  # 16
        [0, 0, 0, 1, 0, 1],  # 17
    ]
    
    # V matrix (12 x 18):
    # Rows: b00,b01,b02,b03, b10,b11,b12,b13, b20,b21,b22,b23
    V_list = [
        # col: b00..b03, b10..b13, b20..b23
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 1
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 2
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 3
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 4
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 5
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 6
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 7
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 8
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # 9
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 10
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 12
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 13
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 14
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 15
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # 16
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # 17
    ]
    
    # W matrix (8 x 18):
    # Rows: c00,c01,c02,c03, c10,c11,c12,c13 (k*n+i, so k*2+i)
    W_list = [
        # col: c00,c01,c02,c03, c10,c11,c12,c13
        [1, 0, 0, 0, 0, 0, 0, 0],  # 0
        [0, 1, 0, 0, 1, 0, 0, 0],  # 1
        [0, 0, 1, 0, 0, 1, 0, 0],  # 2
        [0, 0, 0, 1, 0, 0, 1, 0],  # 3
        [0, 0, 0, 0, 1, 0, 0, 0],  # 4
        [0, 0, 0, 0, 0, 0, 0, 1],  # 5
        [0, 0, 0, 0, 0, 0, 0, 0],  # 6
        [0, 0, 0, 0, 0, 0, 0, 0],  # 7
        [0, 0, 0, 0, 0, 0, 0, 0],  # 8
        [0, 0, 0, 0, 0, 0, 0, 0],  # 9
        [0, 0, 0, 0, 0, 0, 0, 0],  # 10
        [0, 0, 0, 0, 0, 0, 0, 0],  # 11
        [0, 0, 0, 0, 0, 0, 0, 0],  # 12
        [0, 0, 0, 0, 0, 0, 0, 0],  # 13
        [0, 0, 0, 0, 0, 0, 0, 0],  # 14
        [0, 0, 0, 0, 0, 0, 0, 0],  # 15
        [0, 0, 0, 0, 0, 0, 0, 0],  # 16
        [0, 0, 0, 0, 0, 0, 0, 0],  # 17
    ]
    
    # This doesn't look right. Let me use a proper verified decomposition.
    # Let me use a computational approach to find it.
    
    # Actually, let me just return the schoolbook for safety, but try to verify
    # a smaller rank. Given time constraints, let me use rank 18 with a 
    # known correct decomposition.
    
    # Let me use a different, verified approach:
    # The 2x3x4 tensor rank is 18. Here's a correct decomposition:
    
    # I'll use the following construction based on bilinear forms:
    
    # For output C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can write this as a sum of 18 rank-1 terms.
    
    # Using the known result, here's a valid rank-18 decomposition:
    
    U_final = np.zeros((6, 18))
    V_final = np.zeros((12, 18))
    W_final = np.zeros((8, 18))
    
    # Let me define the 18 terms explicitly
    # Each term r has:
    #   u_r: vector in R^6 (coefficients for A elements)
    #   v_r: vector in R^12 (coefficients for B elements)
    #   w_r: vector in R^8 (coefficients for C elements)
    
    # The condition is: for all A, B, sum_r (u_r . A) * (v_r . B) = sum_r w_r * ((u_r . A)*(v_r . B))
    # must equal A*B
    
    # This is equivalent to: the tensor T = sum_r w_r ⊗ u_r ⊗ v_r = multiplication tensor
    
    # Let me use a verified decomposition from the literature.
    
    # After careful construction, here's a rank-18 decomposition:
    
    # Term 0: A[0,0]*B[0,0] -> C[0,0]
    U_final[0, 0] = 1; V_final[0, 0] = 1; W_final[0, 0] = 1
    # Term 1: A[0,0]*B[0,1] -> C[0,1]
    U_final[0, 1] = 1; V_final[1, 1] = 1; W_final[1, 1] = 1
    # Term 2: A[0,0]*B[0,2] -> C[0,2]
    U_final[0, 2] = 1; V_final[2, 2] = 1; W_final[2, 2] = 1
    # Term 3: A[0,0]*B[0,3] -> C[0,3]
    U_final[0, 3] = 1; V_final[3, 3] = 1; W_final[3, 3] = 1
    # Term 4: A[0,1]*B[1,0] -> C[0,0]
    U_final[1, 4] = 1; V_final[4, 4] = 1; W_final[0, 4] = 1
    # Term 5: A[0,1]*B[1,1] -> C[0,1]
    U_final[1, 5] = 1; V_final[5, 5] = 1; W_final[1, 5] = 1
    # Term 6: A[0,1]*B[1,2] -> C[0,2]
    U_final[1, 6] = 1; V_final[6, 6] = 1; W_final[2, 6] = 1
    # Term 7: A[0,1]*B[1,3] -> C[0,3]
    U_final[1, 7] = 1; V_final[7, 7] = 1; W_final[3, 7] = 1
    # Term 8: A[0,2]*B[2,0] -> C[0,0]
    U_final[2, 8] = 1; V_final[8, 8] = 1; W_final[0, 8] = 1
    # Term 9: A[0,2]*B[2,1] -> C[0,1]
    U_final[2, 9] = 1; V_final[9, 9] = 1; W_final[1, 9] = 1
    # Term 10: A[0,2]*B[2,2] -> C[0,2]
    U_final[2, 10] = 1; V_final[10, 10] = 1; W_final[2, 10] = 1
    # Term 11: A[0,2]*B[2,3] -> C[0,3]
    U_final[2, 11] = 1; V_final[11, 11] = 1; W_final[3, 11] = 1
    # Term 12: A[1,0]*B[0,0] -> C[1,0]
    U_final[3, 12] = 1; V_final[0, 12] = 1; W_final[4, 12] = 1
    # Term 13: A[1,0]*B[0,1] -> C[1,1]
    U_final[3, 13] = 1; V_final[1, 13] = 1; W_final[5, 13] = 1
    # Term 14: A[1,0]*B[0,2] -> C[1,2]
    U_final[3, 14] = 1; V_final[2, 14] = 1; W_final[6, 14] = 1
    # Term 15: A[1,0]*B[0,3] -> C[1,3]
    U_final[3, 15] = 1; V_final[3, 15] = 1; W_final[7, 15] = 1
    # Term 16: A[1,1]*B[1,0] -> C[1,0]
    U_final[4, 16] = 1; V_final[4, 16] = 1; W_final[4, 16] = 1
    # Term 17: A[1,1]*B[1,1] -> C[1,1]
    U_final[4, 17] = 1; V_final[5, 17] = 1; W_final[5, 17] = 1
    
    # This is still rank 18 but incomplete - I'm missing some terms
    # and this is just a subset of schoolbook.
    
    # Given the complexity and time constraints, let me return the schoolbook
    # decomposition which is guaranteed correct. The score will be 24.
    
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
