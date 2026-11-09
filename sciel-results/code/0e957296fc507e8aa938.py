"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_sb[u_idx][column] = 1
                V_sb[v_idx][column] = 1
                W_sb[w_idx][column] = 1
                column += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    deadline = 130  # safety margin
    
    # Try structured ansatz: U[i,j,r] = sum_a alpha[r][a] * i^a, V[j,k,r] = sum_b beta[r][b] * j^b * k^c
    # For n=2, i in {0,1}, so U[i,j,r] = A[r][j] + i * B[r][j]
    # V[j,k,r] parameterized as polynomial in j,k
    
    # Strategy: For each candidate rank R, set up U and V with polynomial structure,
    # then solve for W from the constraint that the tensor decomposition holds.
    
    # The tensor equation: sum_r U[i,j,r] * V[j,k,r] = C[i,k,r]... wait
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # where T[i,j,k] = delta (the multiplication tensor)
    # T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to (k,i)... 
    # Wait, let me re-read the convention.
    
    # W[k*n+i, r] is the output factor. The trace convention means:
    # sum_j A[i,j] B[j,k] = C[i,k]
    # Tensor: T[i,j,k] such that sum_{i,j,k} T[i,j,k] x_i y_j z_k = sum_{i,k} C[i,k] ... 
    # Actually the standard tensor for matmul (n x m) * (m x p) -> (n x p):
    # T[i,j,k] where i indexes rows of A (0..n-1), j indexes cols of A / rows of B (0..m-1), 
    # k indexes cols of B (0..p-1). T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Let's try ranks from a lower bound up to schoolbook.
    # Lower bound: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    
    # Try structured ansatz for U and V
    # U[i*m+j, r] = u0[r][j] + i * u1[r][j]  (since i in {0,1})
    # V[j*p+k, r] = v0[r][j][k] + v1[r][j][k] * j + v2[r][j][k] * k + v3[r][j][k] * j*k
    # But that's too many parameters. Let's use simpler structures.
    
    # Simpler: U[i*m+j, r] = a[r][j] * (1 + i * b[r][j])
    # V[j*p+k, r] = c[r][j] * (1 + k * d[r][j])
    
    # Actually, let's just do a direct numerical search.
    # For a given rank R, we have R columns. We need to find U (6 x R), V (21 x R), W (14 x R)
    # such that for all (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    
    # This is a system of 42 equations (2*3*7) with 6*R + 21*R + 14*R = 41*R unknowns.
    # For R=21: 861 unknowns, 42 equations. Very underdetermined.
    # But it's a cubic system, not linear.
    
    # Alternative: Fix U and V, solve for W linearly.
    # For each (i,k): sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
    # Wait no: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for EACH (i,j,k)
    
    # Let me define M_{(i,j,k), (i',k')} ... this is getting complex.
    
    # Let's use a different approach: random search with verification.
    
    def verify(U_mat, V_mat, W_mat, n, m, p):
        """Verify that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 for all i,j,k"""
        n, m, p = n, m, p
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U_mat[0])):
                        s += U_mat[i*m+j][r] * V_mat[j*p+k][r] * W_mat[k*n+i][r]
                    if abs(s - 1) > 1e-10:
                        return False
        return True
    
    # Try random search for lower ranks
    np.random.seed(42)
    
    for R in range(21, min(schoolbook_rank, 35)):
        if time.time() > deadline:
            break
        
        attempts = 0
        max_attempts = 50
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Random U, V, W
            U_r = np.random.randn(n*m, R)
            V_r = np.random.randn(m*p, R)
            W_r = np.random.randn(p*n, R)
            
            # Normalize to improve conditioning
            U_r = U_r / np.linalg.norm(U_r, axis=0, keepdims=True)
            V_r = V_r / np.linalg.norm(V_r, axis=0, keepdims=True)
            W_r = W_r / np.linalg.norm(W_r, axis=0, keepdims=True)
            
            # Check if this satisfies the tensor equation
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r in range(R):
                            s += U_r[i*m+j, r] * V_r[j*p+k, r] * W_r[k*n+i, r]
                        if abs(s - 1) > 1e-8:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            
            if success:
                # Convert to list of lists
                U_list = [[float(x) for x in row] for row in U_r]
                V_list = [[float(x) for x in row] for row in V_r]
                W_list = [[float(x) for x in row] for row in W_r]
                if verify(U_list, V_list, W_list, n, m, p):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = R
                    break
    
    return best_result
# EVOLVE-BLOCK-END
