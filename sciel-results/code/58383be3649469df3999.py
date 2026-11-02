"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=12
    
    # Verify and potentially improve the Strassen-based rank-42 decomposition
    # The key insight: A is 2x2, B is 2x12, C is 2x12
    # We partition B's 12 columns into 6 pairs, apply Strassen to each pair
    # But the A-side Strassen factors are shared across all pairs
    
    # Strassen decomposition for 2x2 x 2x2:
    # C11 = M1 - M3 + M4 - M7
    # C12 = M3 + M5  
    # C21 = M2 + M4
    # C22 = M1 - M2 + M5 + M6
    # where:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B21)
    # M7 = (A12-A22)(B21+B22)
    
    # U tensor: indices (i,j) where i=row of A, j=col of A, so 4 entries
    # V tensor: indices (j,l) where j=row of B, l=col of B, so 24 entries
    # W tensor: indices (l,i) where l=col of B, i=row of C, so 24 entries
    # W convention: W[k*n+i, r] where k is column index, i is row index
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 4 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 24 x 42
    
    # For each of 6 column pairs (u from 0 to 5):
    # Columns 2u and 2u+1 of B
    # B11 = B[0, 2u], B12 = B[0, 2u+1]
    # B21 = B[1, 2u], B22 = B[1, 2u+1]
    
    # Strassen A-factors (shared):
    # A11+A22, A21+A22, A11, A22, A11+A12, A21-A11, A12-A22
    
    # For pair u, the 7 products:
    # M1_u = (A11+A22) * (B[0,2u] + B[0,2u+1])
    # M2_u = (A21+A22) * B[0,2u]
    # M3_u = A11 * (B[1,2u] - B[1,2u+1])
    # M4_u = A22 * (B[0,2u] - B[1,2u])
    # M5_u = (A11+A12) * B[1,2u+1]
    # M6_u = (A21-A11) * (B[0,2u] + B[1,2u])
    # M7_u = (A12-A22) * (B[1,2u] + B[1,2u+1])
    
    # U factors for each Strassen product (A-side):
    # t=0: A11+A22 -> U = [1, 0, 0, 1]
    # t=1: A21+A22 -> U = [0, 0, 1, 1]
    # t=2: A11 -> U = [1, 0, 0, 0]
    # t=3: A22 -> U = [0, 0, 0, 1]
    # t=4: A11+A12 -> U = [1, 1, 0, 0]
    # t=5: A21-A11 -> U = [-1, 0, 1, 0]
    # t=6: A12-A22 -> U = [0, 1, 0, -1]
    
    U_forms = [
        [1, 0, 0, 1],   # A11+A22
        [0, 0, 1, 1],   # A21+A22
        [1, 0, 0, 0],   # A11
        [0, 0, 0, 1],   # A22
        [1, 1, 0, 0],   # A11+A12
        [-1, 0, 1, 0],  # A21-A11
        [0, 1, 0, -1]   # A12-A22
    ]
    
    for u in range(6):
        base_v = 2 * u       # V index offset for column pair u
        base_w = 2 * u * n   # W index offset for column pair u
        
        for t in range(7):
            c = u * 7 + t
            
            # Set U factors
            for i in range(4):
                U[i][c] = U_forms[t][i]
            
            # Set V factors (B-side)
            # V index: j*p + l where j is B row, l is B column
            # For pair u: columns 2u and 2u+1
            # B[0, 2u] is at V index 0*p + 2u = 2u
            # B[0, 2u+1] is at V index 0*p + (2u+1) = 2u+1
            # B[1, 2u] is at V index 1*p + 2u = p + 2u
            # B[1, 2u+1] is at V index 1*p + (2u+1) = p + 2u + 1
            
            if t == 0:  # B[0,2u] + B[0,2u+1]
                V[2*u][c] = 1
                V[2*u+1][c] = 1
            elif t == 1:  # B[0,2u]
                V[2*u][c] = 1
            elif t == 2:  # B[1,2u] - B[1,2u+1]
                V[p + 2*u][c] = 1
                V[p + 2*u + 1][c] = -1
            elif t == 3:  # B[0,2u] - B[1,2u]
                V[2*u][c] = 1
                V[p + 2*u][c] = -1
            elif t == 4:  # B[1,2u+1]
                V[p + 2*u + 1][c] = 1
            elif t == 5:  # B[0,2u] + B[1,2u]
                V[2*u][c] = 1
                V[p + 2*u][c] = 1
            elif t == 6:  # B[1,2u] + B[1,2u+1]
                V[p + 2*u][c] = 1
                V[p + 2*u + 1][c] = 1
            
            # Set W factors (output-side)
            # W convention: W[k*n + i, r] where k is output column, i is output row
            # Output C[i, l] = sum_j A[i,j] B[j,l]
            # For pair u, output columns are 2u and 2u+1
            # C[0, 2u] is at W index 2u*n + 0 = 2u*2 + 0 = 4u
            # C[0, 2u+1] is at W index (2u+1)*n + 0 = (2u+1)*2 + 0 = 4u+2
            # C[1, 2u] is at W index 2u*n + 1 = 4u+1
            # C[1, 2u+1] is at W index (2u+1)*n + 1 = 4u+3
            
            # From Strassen:
            # C11 = M1 - M3 + M4 - M7  -> C[0, 2u]
            # C12 = M3 + M5            -> C[0, 2u+1]
            # C21 = M2 + M4            -> C[1, 2u]
            # C22 = M1 - M2 + M5 + M6  -> C[1, 2u+1]
            
            if t == 0:  # M1: contributes +1 to C[0,2u], +1 to C[1,2u+1]
                W[4*u][c] = 1       # C[0, 2u]
                W[4*u + 3][c] = 1   # C[1, 2u+1]
            elif t == 1:  # M2: contributes +1 to C[1,2u], -1 to C[1,2u+1]
                W[4*u + 1][c] = 1   # C[1, 2u]
                W[4*u + 3][c] = -1  # C[1, 2u+1]
            elif t == 2:  # M3: contributes -1 to C[0,2u], +1 to C[0,2u+1]
                W[4*u][c] = -1      # C[0, 2u]
                W[4*u + 2][c] = 1   # C[0, 2u+1]
            elif t == 3:  # M4: contributes +1 to C[0,2u], +1 to C[1,2u]
                W[4*u][c] = 1       # C[0, 2u]
                W[4*u + 1][c] = 1   # C[1, 2u]
            elif t == 4:  # M5: contributes +1 to C[0,2u+1], +1 to C[1,2u+1]
                W[4*u + 2][c] = 1   # C[0, 2u+1]
                W[4*u + 3][c] = 1   # C[1, 2u+1]
            elif t == 5:  # M6: contributes +1 to C[1,2u+1]
                W[4*u + 3][c] = 1   # C[1, 2u+1]
            elif t == 6:  # M7: contributes -1 to C[0,2u]
                W[4*u][c] = -1      # C[0, 2u]
    
    # Verify the decomposition
    if time.time() < deadline:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reconstruct tensor: T[i,j,l] = sum_r U[i,j,r] * V[j,l,r] * W[l,i,r]
        # Wait, the tensor for matrix multiplication A(2x2) x B(2x12) = C(2x12)
        # T[i,j,l] should be 1 if C[i,l] = sum_j A[i,j]*B[j,l]
        # So T[i,j,l] = 1 for all valid i,j,l (it's the identity tensor for this multiplication)
        # Actually no: the tensor is T[i,j,l] where the multiplication is
        # C[i,l] = sum_j A[i,j] B[j,l]
        # So the tensor has T[i,j,l] = 1 for all i in {0,1}, j in {0,1}, l in {0,...,11}
        # Wait, that's not right either. The tensor decomposes the bilinear map.
        
        # The multiplication tensor for A(n x m) x B(m x p) = C(n x p) is:
        # T_{i,j,l} = 1 where i is row of C/A, j is column of A / row of B, l is column of B/C
        # So T[i,j,l] = 1 for all valid indices
        
        # Decomposition: T[i,j,l] = sum_r U[i*m+j, r] * V[j*p+l, r] * W[l*n+i, r]
        
        # Let's verify
        T_reconstructed = np.zeros((n, m, p))
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for l in range(p):
                        T_reconstructed[i, j, l] += U_arr[i*m+j, r] * V_arr[j*p+l, r] * W_arr[l*n+i, r]
        
        T_expected = np.ones((n, m, p))
        
        if not np.allclose(T_reconstructed, T_expected):
            # Decomposition is incorrect, fall back to schoolbook
            rank = n * m * p  # 48
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            for i in range(n):
                for j in range(m):
                    for l in range(p):
                        c = i * m * p + j * p + l
                        U[i * m + j][c] = 1
                        V[j * p + l][c] = 1
                        W[l * n + i][c] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
