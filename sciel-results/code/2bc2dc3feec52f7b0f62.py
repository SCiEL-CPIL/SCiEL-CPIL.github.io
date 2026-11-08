"""
Kronecker product decomposition for 2x4x5 matrix multiplication.
Uses M_{2,2,2} (Strassen) and M_{1,2,2} (Standard) for the main part.
Uses M_{2,2,1} (Standard) and M_{1,2,1} (Standard) for the tail.
"""

import numpy as np

def construct():
    # Define base decompositions as (U, V, W) matrices
    # Each is a list of columns (rank vectors)
    
    # 1. Strassen 2x2x2 (Rank 7)
    # U: 4x7, V: 4x7, W: 4x7
    # Indices: U[i*m+j], V[j*p+k], W[k*n+i]
    # n=2, m=2, p=2
    
    def get_strassen_222():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    # 2. Standard 1x2x2 (Rank 4)
    # n=1, m=2, p=2
    # U: 2x4, V: 4x4, W: 2x4
    def get_std_122():
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        # C_0 = A_0 B_0 + A_1 B_1
        # C_1 = A_0 B_2 + A_1 B_3
        # M0 = A_0 B_0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # M1 = A_1 B_1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        # M2 = A_0 B_2
        U[0, 2] = 1; V[2, 2] = 1; W[1, 2] = 1
        # M3 = A_1 B_3
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # 3. Standard 2x2x1 (Rank 4)
    # n=2, m=2, p=1
    # U: 4x4, V: 2x4, W: 2x4
    def get_std_221():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        
        # C_0 = A_0 B_0 + A_1 B_1
        # C_1 = A_2 B_0 + A_3 B_1
        # M0 = A_0 B_0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # M1 = A_1 B_1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        # M2 = A_2 B_0
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        # M3 = A_3 B_1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # 4. Standard 1x2x1 (Rank 2)
    # n=1, m=2, p=1
    # U: 2x2, V: 2x2, W: 1x2
    def get_std_121():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        
        # C_0 = A_0 B_0 + A_1 B_1
        # M0 = A_0 B_0
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        # M1 = A_1 B_1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        
        return U, V, W

    # Kronecker Product of two decompositions
    # T1: (n1, m1, p1, r1), T2: (n2, m2, p2, r2)
    # Result: (n1*n2, m1*m2, p1*p2, r1*r2)
    def kronecker_decomp(U1, V1, W1, U2, V2, W2):
        n1, m1, p1, r1 = U1.shape[0], U1.shape[0]//1, W1.shape[0]//1, U1.shape[1] 
        # Actually, shapes are given.
        # U1: (n1*m1, r1), V1: (m1*p1, r1), W1: (p1*n1, r1)
        n1 = U1.shape[0] // m1
        # We need to pass dimensions explicitly to avoid ambiguity
        pass 

    # Let's do it manually with loops for clarity and correctness
    def combine_kronecker(U1, V1, W1, dims1, U2, V2, W2, dims2):
        # dims1: (n1, m1, p1)
        # dims2: (n2, m2, p2)
        n1, m1, p1 = dims1
        n2, m2, p2 = dims2
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        r = r1 * r2
        
        n = n1 * n2
        m = m1 * m2
        p = p1 * p2
        
        U = np.zeros((n * m, r))
        V = np.zeros((m * p, r))
        W = np.zeros((p * n, r))
        
        for k1 in range(r1):
            for k2 in range(r2):
                k = k1 * r2 + k2
                
                # U part
                # Index in U1: i1 * m1 + j1
                # Index in U2: i2 * m2 + j2
                # Index in U: (i1 * n2 + i2) * m + (j1 * m2 + j2)
                for i1 in range(n1):
                    for j1 in range(m1):
                        val1 = U1[i1 * m1 + j1, k1]
                        if val1 == 0: continue
                        for i2 in range(n2):
                            for j2 in range(m2):
                                val2 = U2[i2 * m2 + j2, k2]
                                if val2 == 0: continue
                                val = val1 * val2
                                idx_u = (i1 * n2 + i2) * m + (j1 * m2 + j2)
                                U[idx_u, k] += val
                                
                # V part
                # Index in V1: j1 * p1 + k1_v
                # Index in V2: j2 * p2 + k2_v
                # Index in V: (j1 * p2 + j2) * p + (k1_v * p2 + k2_v)
                for j1 in range(m1):
                    for k1_v in range(p1):
                        val1 = V1[j1 * p1 + k1_v, k1]
                        if val1 == 0: continue
                        for j2 in range(m2):
                            for k2_v in range(p2):
                                val2 = V2[j2 * p2 + k2_v, k2]
                                if val2 == 0: continue
                                val = val1 * val2
                                idx_v = (j1 * p2 + j2) * p + (k1_v * p2 + k2_v)
                                V[idx_v, k] += val
                                
                # W part
                # Index in W1: k1_w * n1 + i1_w
                # Index in W2: k2_w * n2 + i2_w
                # Index in W: (k1_w * p2 + k2_w) * n + (i1_w * n2 + i2_w)
                for k1_w in range(p1):
                    for i1_w in range(n1):
                        val1 = W1[k1_w * n1 + i1_w, k1]
                        if val1 == 0: continue
                        for k2_w in range(p2):
                            for i2_w in range(n2):
                                val2 = W2[k2_w * n2 + i2_w, k2]
                                if val2 == 0: continue
                                val = val1 * val2
                                idx_w = (k1_w * p2 + k2_w) * n + (i1_w * n2 + i2_w)
                                W[idx_w, k] += val
                                
        return U, V, W

    # Get base components
    U_s, V_s, W_s = get_strassen_222()
    dims_s = (2, 2, 2)
    
    U_122, V_122, W_122 = get_std_122()
    dims_122 = (1, 2, 2)
    
    U_221, V_221, W_221 = get_std_221()
    dims_221 = (2, 2, 1)
    
    U_121, V_121, W_121 = get_std_121()
    dims_121 = (1, 2, 1)

    # Part 1: 2x4x4
    # Strassen (2,2,2) x Std (1,2,2) -> (2, 4, 4)
    U_p1, V_p1, W_p1 = combine_kronecker(U_s, V_s, W_s, dims_s, U_122, V_122, W_122, dims_122)
    
    # Part 2: 2x4x1
    # Std (2,2,1) x Std (1,2,1) -> (2, 4, 1)
    U_p2, V_p2, W_p2 = combine_kronecker(U_221, V_221, W_221, dims_221, U_121, V_121, W_121, dims_121)
    
    # Combine Part 1 and Part 2
    # Part 1 handles columns 0-3 of B, rows 0-1 of C.
    # Part 2 handles column 4 of B, rows 0-1 of C.
    # They are disjoint in W indices (k dimension).
    # Part 1 W shape: (4*2, 28) = (8, 28). Indices k in 0..3.
    # Part 2 W shape: (1*2, 8) = (2, 8). Indices k in 0..0 (local).
    # We need to map Part 2's W indices to the global k=4.
    
    n, m, p = 2, 4, 5
    r1 = U_p1.shape[1]
    r2 = U_p2.shape[1]
    r = r1 + r2
    
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Copy Part 1
    U[:, :r1] = U_p1
    V[:, :r1] = V_p1
    W[:, :r1] = W_p1
    
    # Copy Part 2
    U[:, r1:] = U_p2
    V[:, r1:] = V_p2
    
    # Map W_p2 to W
    # W_p2 has shape (1*2, 8). Local k=0.
    # Global k=4.
    # W index = k * n + i.
    # Local index = k_loc * n_loc + i_loc = 0 * 2 + i_loc = i_loc.
    # Global index = 4 * 2 + i_loc = 8 + i_loc.
    # W_p2[i_loc, col] -> W[8 + i_loc, r1 + col]
    for i in range(2):
        for c in range(r2):
            W[8 + i, r1 + c] = W_p2[i, c]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

DIMENSIONS = [2, 4, 5]

if __name__ == "__main__":
    # Validation
    res = construct()
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    
    n, m, p = 2, 4, 5
    rank = U.shape[1]
    print(f"Rank: {rank}")
    
    # Check exactness
    # T = sum_k U_k V_k W_k^T
    # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # We want T to be 1 if i,j,k correspond to A_ij B_jk C_ik?
    # No, the tensor is the multiplication tensor.
    # T[i, j, k] = 1 if A_ij and B_jk contribute to C_ik?
    # Actually, the decomposition satisfies:
    # C_ik = sum_j A_ij B_jk
    # This is equivalent to:
    # sum_{i,j,k} C_ik A_ij B_jk = sum_{i,j,k} (sum_r U_{ij,r} V_{jk,r} W_{ik,r}) A_ij B_jk
    # So we need to check if the tensor matches the multiplication tensor.
    
    T_decomp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A_ij B_jk -> C_ik
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                T_target[i*m + j, j*p + k, k*n + i] = 1
                
    if np.allclose(T_decomp, T_target):
        print("Valid decomposition")
    else:
        print("Invalid decomposition")
        print(np.nonzero(T_decomp != T_target))
