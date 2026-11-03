import numpy as np
import time
import random

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to get 2x2 Strassen factors
    # U, V, W shapes (4, 7)
    # Indices: 0:00, 1:01, 2:10, 3:11
    def get_strassen_2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # s=0: M1
        U[:, 0] = [1, 0, 0, 1]
        V[:, 0] = [1, 0, 0, 1]
        W[:, 0] = [1, 0, 0, 1] # C00+=1, C11+=1
        
        # s=1: M2
        U[:, 1] = [0, 0, 1, 1]
        V[:, 1] = [1, 0, 0, 0]
        W[:, 1] = [0, 0, 1, -1] # C10+=1, C11-=1
        
        # s=2: M3
        U[:, 2] = [1, 0, 0, 0]
        V[:, 2] = [0, 1, 0, -1]
        W[:, 2] = [0, 1, 0, 1] # C01+=1, C11+=1
        
        # s=3: M4
        U[:, 3] = [0, 0, 0, 1]
        V[:, 3] = [0, -1, 1, 0]
        W[:, 3] = [1, 0, 1, 0] # C00+=1, C10+=1
        
        # s=4: M5
        U[:, 4] = [1, 1, 0, 0]
        V[:, 4] = [0, 0, 0, 1]
        W[:, 4] = [-1, 1, 0, 1] # C00-=1, C01+=1, C11+=1
        
        # s=5: M6
        U[:, 5] = [-1, 0, 1, 0]
        V[:, 5] = [1, 1, 0, 0]
        W[:, 5] = [0, 0, 0, 1] # C11+=1
        
        # s=6: M7
        U[:, 6] = [0, 1, 0, -1]
        V[:, 6] = [0, 0, 1, 1]
        W[:, 6] = [1, 0, 0, 0] # C00+=1
        
        return U, V, W

    # Generate 4x4x4 Strassen factors
    # Rank 49
    def get_strassen_4x4():
        U2, V2, W2 = get_strassen_2x2()
        
        U4 = np.zeros((16, 49))
        V4 = np.zeros((16, 49))
        W4 = np.zeros((16, 49))
        
        col = 0
        for s in range(7):
            for t in range(7):
                # Outer coefficients
                u_outer = U2[:, s] # 4 values
                v_outer = V2[:, s]
                w_outer = W2[:, s]
                
                # Inner coefficients
                u_inner = U2[:, t]
                v_inner = V2[:, t]
                w_inner = W2[:, t]
                
                # Construct U4[:, col]
                # Global index i, j in 0..3
                # Block index p, q = i//2, j//2
                # Local index i', j' = i%2, j%2
                for i in range(4):
                    for j in range(4):
                        p = i // 2
                        q = j // 2
                        ip = i % 2
                        jp = j % 2
                        U4[i*4+j, col] = u_outer[p*2+q] * u_inner[ip*2+jp]
                        
                for j in range(4):
                    for k in range(4):
                        p = j // 2
                        q = k // 2
                        jp = j % 2
                        kp = k % 2
                        V4[j*4+k, col] = v_outer[p*2+q] * v_inner[jp*2+kp]
                        
                for k in range(4):
                    for i in range(4):
                        p = k // 2
                        q = i // 2
                        kp = k % 2
                        ip = i % 2
                        W4[k*4+i, col] = w_outer[p*2+q] * w_inner[kp*2+ip]
                        
                col += 1
                
        return U4, V4, W4

    U4, V4, W4 = get_strassen_4x4()
    
    # 4x4x1 factors
    # Rank 4
    U1 = np.zeros((16, 4))
    V1 = np.zeros((16, 4)) # Actually V is 4x1, so shape (4, 4) in 4x4 context? No, V is m*p = 4*1 = 4.
    # In the 4x4x1 subproblem, m=4, p=1.
    # U shape (16, 4), V shape (4, 4), W shape (4, 4).
    # Wait, in the main problem, V is (4, 9).
    # For the 4x4x1 part, we use columns 8 of B.
    # So V_local is (4, 1).
    # We need to embed this.
    
    # Let's just build the full U, V, W for the 102 rank case.
    rank = 49 + 49 + 4
    U = np.zeros((16, rank))
    V = np.zeros((36, rank))
    W = np.zeros((36, rank))
    
    # Part 1: 4x4x4 (cols 0-3 of B)
    # U is same as U4
    U[:, 0:49] = U4
    # V: embed V4 (16x49) into (36, 49)
    # V4[j, k] corresponds to j in 0..3, k in 0..3.
    # Global V index is j*9 + k.
    for c in range(49):
        for j in range(4):
            for k in range(4):
                V[j*9 + k, c] = V4[j*4 + k, c]
    # W: embed W4 (16x49) into (36, 49)
    # W4[k, i] corresponds to k in 0..3, i in 0..3.
    # Global W index is k*4 + i.
    for c in range(49):
        for k in range(4):
            for i in range(4):
                W[k*4 + i, c] = W4[k*4 + i, c]
                
    # Part 2: 4x4x4 (cols 4-7 of B)
    # U is same as U4
    U[:, 49:98] = U4
    # V: embed V4 into (36, 49) with offset 4 in k
    for c in range(49):
        for j in range(4):
            for k in range(4):
                V[j*9 + (k+4), 49+c] = V4[j*4 + k, c]
    # W: embed W4 into (36, 49) with offset 4 in k
    for c in range(49):
        for k in range(4):
            for i in range(4):
                W[(k+4)*4 + i, 49+c] = W4[k*4 + i, c]
                
    # Part 3: 4x4x1 (col 8 of B)
    # Rank 4
    # U[i, j] = 1 if c=j?
    # My previous derivation:
    # For c in 0..3 (local), U[i, j] = 1 if j == c.
    # V[j, 8] = 1 if j == c.
    # W[8, i] = 1.
    
    for c in range(4):
        # U
        for i in range(4):
            U[i*4 + c, 98+c] = 1
        # V
        V[c*9 + 8, 98+c] = 1
        # W
        for i in range(4):
            W[8*4 + i, 98+c] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
