import numpy as np
from itertools import product

def construct():
    n, m, p = 2, 2, 2
    
    # Build the matrix multiplication tensor T of shape (4,4,4)
    # T[a_idx, b_idx, c_idx] = 1 if C[c_idx] = A[a_idx] * B[b_idx]
    # Using row-major: a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
    # For 2x2: 
    #   C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]  -> a=0,1; b=0,2; c=0
    #   C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]  -> a=0,1; b=1,3; c=2
    #   C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]  -> a=2,3; b=0,2; c=1
    #   C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]  -> a=2,3; b=1,3; c=3
    
    T = np.zeros((4, 4, 4), dtype=int)
    T[0, 0, 0] = 1  # C00 += A00*B00
    T[1, 2, 0] = 1  # C00 += A01*B10
    T[0, 1, 2] = 1  # C01 += A00*B01
    T[1, 3, 2] = 1  # C01 += A01*B11
    T[2, 0, 1] = 1  # C10 += A10*B00
    T[3, 2, 1] = 1  # C10 += A11*B10
    T[2, 1, 3] = 1  # C11 += A10*B01
    T[3, 3, 3] = 1  # C11 += A11*B11
    
    def verify_decomp(U_list, V_list, W_list):
        """Verify that U, V, W exactly decompose T"""
        r = len(U_list[0])
        U_arr = np.array(U_list, dtype=float)
        V_arr = np.array(V_list, dtype=float)
        W_arr = np.array(W_list, dtype=float)
        
        # Reconstruct tensor: T_recon[i,j,k] = sum_t U[i,t] * V[j,t] * W[k,t]
        T_recon = np.zeros((4, 4, 4), dtype=float)
        for t in range(r):
            T_recon += np.einsum('i,j,k->ijk', U_arr[:, t], V_arr[:, t], W_arr[:, t])
        
        return np.allclose(T_recon, T, atol=1e-10)
    
    # Attempt rank-6 search with small integer coefficients
    # Use a structured search: try all combinations of -2,-1,0,1,2 for each entry
    # This is computationally expensive, so use a smarter approach
    
    # Strategy: Use the known structure of bilinear decompositions
    # Each term t has U[:,t], V[:,t], W[:,t] as vectors
    # We need sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t] = T
    
    # Try a randomized local search approach
    import random
    random.seed(42)
    
    found_rank6 = False
    best_rank6_decomp = None
    
    # Try several random initializations and local improvements
    for trial in range(50):
        # Initialize with random small integers
        U = [[random.choice([-1, 0, 1]) for _ in range(6)] for _ in range(4)]
        V = [[random.choice([-1, 0, 1]) for _ in range(6)] for _ in range(4)]
        W = [[random.choice([-1, 0, 1]) for _ in range(6)] for _ in range(4)]
        
        # Local search: try to improve by changing one entry at a time
        best_obj = float('inf')
        for step in range(200):
            if verify_decomp(U, V, W):
                found_rank6 = True
                best_rank6_decomp = (U, V, W)
                break
            
            # Compute current error
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            T_recon = np.zeros((4, 4, 4), dtype=float)
            for t in range(6):
                T_recon += np.einsum('i,j,k->ijk', U_arr[:, t], V_arr[:, t], W_arr[:, t])
            obj = np.sum((T_recon - T) ** 2)
            
            if obj < best_obj:
                best_obj = obj
            
            # Pick a random position to modify
            mat = random.choice(['U', 'V', 'W'])
            row = random.randint(0, 3)
            col = random.randint(0, 5)
            
            old_val = U[row][col] if mat == 'U' else (V[row][col] if mat == 'V' else W[row][col])
            new_val = random.choice([-2, -1, 0, 1, 2])
            
            if mat == 'U':
                U[row][col] = new_val
            elif mat == 'V':
                V[row][col] = new_val
            else:
                W[row][col] = new_val
            
            if not verify_decomp(U, V, W):
                if mat == 'U':
                    U[row][col] = old_val
                elif mat == 'V':
                    V[row][col] = old_val
                else:
                    W[row][col] = old_val
        
        if found_rank6:
            break
    
    if found_rank6:
        U, V, W = best_rank6_decomp
        return {"U": U, "V": V, "W": W}
    
    # Fall back to Strassen rank-7
    rank = 7
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 0: (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1: (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # Term 2: A00(B01-B11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # Term 3: A11(B10-B00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # Term 4: (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # Term 5: (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6: (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return {"U": U, "V": V, "W": W}
