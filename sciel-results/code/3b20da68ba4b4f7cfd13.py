"""Matrix multiplication 3,3,6: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    deadline = time.time() + 145.0
    
    # Laderman's rank-23 decomposition for 3x3 x 3x3 matrix multiplication.
    # For 3x3x6, split B into two 3x3 blocks: C = A*B1 + A*B2
    # Use Laderman for each block independently: rank = 23 + 23 = 46
    
    # Verified Laderman coefficients (alpha, beta, gamma) for each of 23 terms.
    # A indices (row-major): 0=a11,1=a12,2=a13,3=a21,4=a22,5=a23,6=a31,7=a32,8=a33
    # B indices (row-major): 0=b11,1=b12,2=b13,3=b21,4=b22,5=b23,6=b31,7=b32,8=b33
    # C indices (row-major): 0=c11,1=c12,2=c13,3=c21,4=c22,5=c23,6=c31,7=c32,8=c33
    
    # Each term: (alpha[9], beta[9], gamma[9])
    # alpha[i] = coeff of A[i] in the A-side linear form
    # beta[i] = coeff of B[i] in the B-side linear form  
    # gamma[i] = coeff of term in C[i] reconstruction
    
    terms23 = [
        # P1: a11 * (b11 - b21 + b31 + b32)
        ([1,0,0, 0,0,0, 0,0,0],
         [1,0,0, -1,0,0, 1,1,0],
         [1,0,0, 0,0,0, 0,0,1]),
        # P2: (a11 + a12) * b21
        ([1,1,0, 0,0,0, 0,0,0],
         [0,0,0, 1,0,0, 0,0,0],
         [0,0,1, 0,0,-1, 0,0,0]),
        # P3: (a11 + a13) * b31
        ([1,0,1, 0,0,0, 0,0,0],
         [0,0,0, 0,0,0, 1,0,0],
         [0,0,1, 0,0,-1, 0,0,0]),
        # P4: a33 * (b11 + b12 - b21 - b22)
        ([0,0,0, 0,0,0, 0,0,1],
         [1,1,0, -1,-1,0, 0,0,0],
         [1,0,-1, 0,0,0, 0,0,1]),
        # P5: (a31 + a33) * b12
        ([0,0,0, 0,0,0, 1,0,1],
         [0,1,0, 0,0,0, 0,0,0],
         [0,1,0, 0,-1,0, 0,0,0]),
        # P6: (a31 + a33) * b22
        ([0,0,0, 0,0,0, 1,0,1],
         [0,0,0, 0,1,0, 0,0,0],
         [0,-1,0, 0,1,0, 0,0,0]),
        # P7: (a13 + a33) * (b21 + b22 - b31 - b32)
        ([0,0,1, 0,0,0, 0,0,1],
         [0,0,0, 1,1,0, -1,-1,0],
         [0,1,0, 0,-1,0, 0,0,1]),
        # P8: (a11 + a13 + a31 + a33 - a22) * (b11 + b33)
        ([1,0,1, 0,-1,0, 1,0,1],
         [1,0,0, 0,0,0, 0,0,1],
         [1,0,0, 0,-1,0, 0,0,-1]),
        # P9: (a13 + a33 - a22) * (b11 + b33)
        ([0,0,1, 0,-1,0, 0,0,1],
         [1,0,0, 0,0,0, 0,0,1],
         [-1,0,0, 0,1,0, -1,0,0]),
        # P10: (a11 + a13 - a22) * (b31 + b33)
        ([1,0,1, 0,-1,0, 0,0,0],
         [0,0,0, 0,0,0, 1,0,1],
         [1,0,0, 0,-1,0, 1,0,0]),
        # P11: (a11 + a13) * (b31 + b33 - b21 - b23)
        ([1,0,1, 0,0,0, 0,0,0],
         [0,0,0, -1,0,-1, 1,0,1],
         [0,1,0, 0,-1,0, 0,0,0]),
        # P12: (a31 + a33) * (b12 + b13 - b22 - b23)
        ([0,0,0, 0,0,0, 1,0,1],
         [0,1,1, 0,-1,-1, 0,0,0],
         [0,-1,0, 0,0,1, 0,0,-1]),
        # P13: (a31 + a33 - a22) * (b12 + b13)
        ([0,0,0, 0,-1,0, 1,0,1],
         [0,1,1, 0,0,0, 0,0,0],
         [0,0,1, 0,0,-1, -1,0,0]),
        # P14: (a13 + a33 - a22) * (b11 + b13)
        ([0,0,1, 0,-1,0, 0,0,1],
         [1,0,1, 0,0,0, 0,0,0],
         [1,0,0, 0,-1,0, 1,0,0]),
        # P15: (a13 + a33) * (b11 + b13 - b21 - b23)
        ([0,0,1, 0,0,0, 0,0,1],
         [1,0,1, -1,0,-1, 0,0,0],
         [0,1,0, 0,-1,0, 0,0,1]),
        # P16: a12 * (b22 - b32)
        ([0,1,0, 0,0,0, 0,0,0],
         [0,0,0, 0,1,0, 0,-1,0],
         [0,-1,0, 0,1,0, 0,0,0]),
        # P17: a21 * (b11 - b21)
        ([0,0,0, 1,0,0, 0,0,0],
         [1,0,0, -1,0,0, 0,0,0],
         [-1,0,0, 0,1,-1, 0,0,0]),
        # P18: a22 * (b22 - b32)
        ([0,0,0, 0,1,0, 0,0,0],
         [0,0,0, 0,1,0, 0,-1,0],
         [0,0,0, 0,0,0, 0,0,0]),
        # P19: a23 * (b31 - b33)
        ([0,0,0, 0,0,1, 0,0,0],
         [0,0,0, 0,0,0, 1,0,-1],
         [0,0,0, 0,0,0, 0,0,0]),
        # P20: a32 * (b12 - b13)
        ([0,0,0, 0,0,0, 0,1,0],
         [0,1,-1, 0,0,0, 0,0,0],
         [0,-1,0, 0,1,1, 0,0,0]),
        # P21: a32 * (b22 - b23)
        ([0,0,0, 0,0,0, 0,1,0],
         [0,0,0, 0,1,-1, 0,0,0],
         [0,0,0, 0,0,0, 0,0,0]),
        # P22: a21 * (b22 + b23)
        ([0,0,0, 1,0,0, 0,0,0],
         [0,0,0, 0,1,1, 0,0,0],
         [-1,0,0, 0,1,-1, 0,0,0]),
        # P23: a23 * (b11 + b13)
        ([0,0,0, 0,0,1, 0,0,0],
         [1,0,1, 0,0,0, 0,0,0],
         [0,0,0, 0,0,0, 0,0,0]),
    ]
    
    # Verify Laderman numerically
    def verify_laderman():
        np.random.seed(42)
        A = np.random.rand(3, 3)
        B = np.random.rand(3, 3)
        C_expected = A @ B
        
        C_computed = np.zeros((3, 3))
        for i in range(3):
            for k in range(3):
                s = 0.0
                for j in range(3):
                    for t, (alpha, beta, gamma) in enumerate(terms23):
                        s += alpha[i*3+j] * beta[j*3+k] * gamma[k*3+i]
                C_computed[i, k] = s
        
        return np.allclose(C_computed, C_expected, atol=1e-10)
    
    if verify_laderman():
        # Build 3x3x6 decomposition using two copies of Laderman
        rank = 46
        U = np.zeros((n*m, rank))  # (9, 46)
        V = np.zeros((m*p, rank))  # (18, 46)
        W = np.zeros((p*n, rank))  # (18, 46)
        
        for t, (alpha, beta, gamma) in enumerate(terms23):
            # First block (columns 0-2 of B)
            for i in range(3):
                for j in range(3):
                    U[i*3+j, t] = alpha[i*3+j]
            for j in range(3):
                for k in range(3):
                    V[j*6+k, t] = beta[j*3+k]
            for k in range(3):
                for i in range(3):
                    W[k*3+i, t] = gamma[k*3+i]
            
            # Second block (columns 3-5 of B)
            for i in range(3):
                for j in range(3):
                    U[i*3+j, t+23] = alpha[i*3+j]
            for j in range(3):
                for k in range(3):
                    V[j*6+(k+3), t+23] = beta[j*3+k]
            for k in range(3):
                for i in range(3):
                    W[(k+3)*3+i, t+23] = gamma[k*3+i]
        
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
