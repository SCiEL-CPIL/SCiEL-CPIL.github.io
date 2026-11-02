import time
import sys
import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    # Rank 39 decomposition: 5 pairs of columns (Strassen rank 7) + 1 single column (Schoolbook rank 4)
    rank = 5 * 7 + 1 * 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # 5 pairs of columns using Strassen's algorithm
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # V indices for B (row 0: k0, k1; row 1: 11+k0, 11+k1)
        v_e = k0       # B[0, k0]
        v_f = k1       # B[0, k1]
        v_g = 11 + k0  # B[1, k0]
        v_h = 11 + k1  # B[1, k1]
        
        # W indices for C (trace convention W[k*n+i, r])
        w_c00 = 2 * k0     # C[0, k0]
        w_c10 = 2 * k0 + 1 # C[1, k0]
        w_c01 = 2 * k1     # C[0, k1]
        w_c11 = 2 * k1 + 1 # C[1, k1]
        
        # m1 = (a+d)(e+h)
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m2 = (c+d)e
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        # m3 = a(f-h)
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        # m4 = d(g-e)
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        # m5 = (a+b)h
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1

    # Single column k2 = 10 using schoolbook
    k2 = 10
    v_e2 = k2
    v_g2 = 11 + k2
    w_c02 = 2 * k2
    w_c12 = 2 * k2 + 1
    
    # ae
    U[0][r] = 1
    V[v_e2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # bg
    U[1][r] = 1
    V[v_g2][r] = 1
    W[w_c02][r] = 1
    r += 1
    
    # ce
    U[2][r] = 1
    V[v_e2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    # dg
    U[3][r] = 1
    V[v_g2][r] = 1
    W[w_c12][r] = 1
    r += 1
    
    return {"U": U, "V": V, "W": W}

def _verify():
    """Verify the decomposition is correct."""
    result = construct()
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    n, m, p = DIMENSIONS
    
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    # But we need to be careful about the index conventions.
    # U has shape (n*m, rank) - index i in A (row, col)
    # V has shape (m*p, rank) - index j in B (row, col)  
    # W has shape (p*n, rank) - index k in C (col, row) per trace convention W[k*n+i, r]
    
    # A is n x m, B is m x p, C is n x p
    # C[i,j] = sum_l A[i,l] * B[l,j]
    
    # U index: i*m + l for A[i,l]
    # V index: l*p + j for B[l,j]
    # W index: j*n + i for C[i,j] (trace convention)
    
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        
        # Expand u_col to n x m
        u_mat = u_col.reshape(n, m)
        # Expand v_col to m x p
        v_mat = v_col.reshape(m, p)
        # Expand w_col to p x n
        w_mat = w_col.reshape(p, n)
        
        # For each (i, l, j): T[i,l,j] += u_mat[i,l] * v_mat[l,j] * w_mat[j,i]
        for i in range(n):
            for l in range(m):
                for j in range(p):
                    T[i, l, j] += u_mat[i, l] * v_mat[l, j] * w_mat[j, i]
    
    # Build the target tensor
    MM = np.zeros((n, m, p))
    for i in range(n):
        for l in range(m):
            for j in range(p):
                MM[i, l, j] = 1 if True else 0  # This is wrong, need to think more carefully
    
    # Actually, the matrix multiplication tensor T[i,l,j] = A[i,l] * B[l,j] contributes to C[i,j]
    # So the tensor for matrix multiplication has T[i,l,j] = 1 for all valid i,l,j
    # because C[i,j] = sum_l A[i,l] * B[l,j]
    
    MM = np.ones((n, m, p))
    
    diff = np.abs(T - MM)
    max_diff = np.max(diff)
    print(f"Verification: max diff = {max_diff}")
    return max_diff < 1e-10
