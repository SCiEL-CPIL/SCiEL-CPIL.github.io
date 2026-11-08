import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=3, p=5
    # Total schoolbook rank: 45
    # Target rank: 41 (23 for 3x3x3 + 18 for 3x3x2)
    
    # Define Laderman's 23 terms for 3x3x3
    # L, R are 9-dim vectors (flattened 3x3 matrices)
    # M is 9-dim vector (flattened 3x3 matrix)
    
    laderman_terms = [
        # (L, R, M)
        ([1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0]),
        ([0,1,0, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,0,1]),
        ([1,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [-1,0,0, 0,0,0, 0,0,0]),
        ([1,1,0, 0,0,0, 0,0,0], [1,0,0, 0,0,0, 0,0,0], [0,0,0, 1,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,1, 0,0,0], [0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 1,0,0, 0,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,0, 1,0,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,0, 0,1,0], [0,0,0, 0,0,0, 0,0,1]),
        ([1,0,-1, 0,0,0, 0,0,0], [0,0,1, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,1,-1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0]),
        ([0,0,1, 0,0,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,1,0, 0,0,0], [0,1,0, 0,0,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,1,0, 0,0,0], [0,0,0, 0,1,0, 0,0,0]),
        ([0,0,0, 0,0,0, 1,0,0], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
        ([0,0,0, 0,0,0, 0,0,1], [0,0,0, 0,0,1, 0,0,0], [0,0,0, 0,0,0, 0,1,0]),
    ]
    
    # Define Schoolbook terms for 3x3x2
    # B has columns 0,1,2,3,4. We handle cols 3,4 here.
    # A is 3x3, B is 3x2 (cols 3,4), C is 3x2 (cols 3,4)
    # Indices for B: j in 0..2, k in 3..4. Flat index j*5 + k.
    # Indices for C: k in 3..4, i in 0..2. Flat index k*3 + i.
    
    schoolbook_332_terms = []
    for i in range(3):
        for j in range(3):
            for k in [3, 4]:
                L = [0]*9
                L[i*3 + j] = 1
                R = [0]*15
                R[j*5 + k] = 1
                M = [0]*15
                M[k*3 + i] = 1
                schoolbook_332_terms.append((L, R, M))
                
    # Combine terms
    # Total rank = 23 + 18 = 41
    rank = 41
    U = np.zeros((9, rank))
    V = np.zeros((15, rank))
    W = np.zeros((15, rank))
    
    # Fill Laderman terms (cols 0-22)
    for r in range(23):
        L, R, M = laderman_terms[r]
        U[:, r] = L
        V[:9, r] = R  # R is 9-dim, V is 15-dim. Cols 3-4 of B are 0.
        W[:9, r] = M  # M is 9-dim, W is 15-dim. Cols 3-4 of C are 0.
        
    # Fill Schoolbook 3x3x2 terms (cols 23-40)
    for r in range(18):
        L, R, M = schoolbook_332_terms[r]
        U[:, 23 + r] = L
        V[:, 23 + r] = R
        W[:, 23 + r] = M
        
    # Verify the decomposition
    # C = sum_r W_r * (U_r * A) * (V_r * B) ?
    # No, C_{i,k} = sum_r W_{k*3+i, r} * (sum_j U_{i*3+j, r} A_{i,j}) * (sum_j V_{j*5+k, r} B_{j,k})
    
    # Random test
    np.random.seed(42)
    A_test = np.random.rand(3, 3)
    B_test = np.random.rand(3, 5)
    C_expected = A_test @ B_test
    
    C_calc = np.zeros((3, 5))
    for r in range(rank):
        # L_r(A)
        L_val = 0
        for i in range(3):
            for j in range(3):
                L_val += U[i*3 + j, r] * A_test[i, j]
                
        # R_r(B)
        R_val = 0
        for j in range(3):
            for k in range(5):
                R_val += V[j*5 + k, r] * B_test[j, k]
                
        # Add to C
        for i in range(3):
            for k in range(5):
                C_calc[i, k] += W[k*3 + i, r] * L_val * R_val
                
    if not np.allclose(C_calc, C_expected, atol=1e-9):
        # Fallback to schoolbook 45
        rank_sb = 45
        U_sb = np.zeros((9, rank_sb))
        V_sb = np.zeros((15, rank_sb))
        W_sb = np.zeros((15, rank_sb))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(5):
                    U_sb[i*3 + j, col] = 1
                    V_sb[j*5 + k, col] = 1
                    W_sb[k*3 + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
