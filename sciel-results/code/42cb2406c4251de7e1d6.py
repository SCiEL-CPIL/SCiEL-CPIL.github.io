import numpy as np
import time
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Schoolbook decomposition (Rank 60)
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_sb

    # Try to improve using a block-based Strassen approach
    # Split the m=4 dimension into two 2-dim blocks
    # For each block, use a 3x2 x 2x5 multiplication
    # Use Strassen-like optimization on the 2x2x2 core
    
    # Attempt 1: Structured decomposition using 2x2 Strassen blocks
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # = sum_{j=0}^{1} A[i,j] B[j,k] + sum_{j=2}^{3} A[i,j] B[j,k]
    # Let A1 = A[:,0:2], A2 = A[:,2:4], B1 = B[0:2,:], B2 = B[2:4,:]
    # C = A1*B1 + A2*B2
    # Each term is 3x2 * 2x5
    
    # For 3x2 * 2x5, we can use a rank-10 schoolbook (3*2*5=30) 
    # But we can do better by noting the structure.
    
    # Let's try a direct linear algebra approach to find a lower rank solution
    # We'll use a randomized approach: generate random U, V and solve for W
    
    def try_rank(R, max_iters=200):
        if time.time() > deadline - 30:
            return None
        
        # Generate random integer matrices U and V
        # Then solve for W from the linear equations
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # For fixed U, V, the equation is linear in W
        # For each (i,j,k): sum_r (U[i*m+j, r] * V[j*p+k, r]) * W[k*n+i, r] = 1
        # This is a linear system in W's entries
        
        for it in range(max_iters):
            if time.time() > deadline - 30:
                return None
            
            # Random small integer initialization
            U = np.random.randint(-1, 2, (n*m, R))
            V = np.random.randint(-1, 2, (m*p, R))
            
            # Build linear system for W
            # Unknowns: W[k*n+i, r] for k in [0,p), i in [0,n), r in [0,R)
            # Total unknowns: p*n*R
            # Equations: n*m*p = 60
            num_unknowns = p * n * R
            num_eqs = n * m * p
            
            # A_matrix: num_eqs x num_unknowns
            A_mat = np.zeros((num_eqs, num_unknowns))
            b_vec = np.ones(num_eqs)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(R):
                            coeff = U[i*m+j, r] * V[j*p+k, r]
                            if coeff != 0:
                                # W index: k*n + i, r
                                w_idx = (k*n + i) * R + r
                                A_mat[eq_idx, w_idx] = coeff
                        eq_idx += 1
            
            # Solve least squares (we need exact solution)
            try:
                W_flat, residuals, rank_svd, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                # Check if solution is exact
                W_mat = W_flat.reshape(p*n, R)
                
                # Verify exactness
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for r in range(R):
                                val += U[i*m+j, r] * V[j*p+k, r] * W_mat[k*n+i, r]
                            if abs(val - 1.0) > 1e-6:
                                error += abs(val - 1.0)
                
                if error < 1e-6:
                    # Found a valid solution! Convert to list of lists with rounded values
                    U_list = [[int(round(U[i, r])) for r in range(R)] for i in range(n*m)]
                    V_list = [[int(round(V[i, r])) for r in range(R)] for i in range(m*p)]
                    W_list = [[int(round(W_mat[i, r])) for r in range(R)] for i in range(p*n)]
                    
                    # Final verification with integers
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = 0
                                for r in range(R):
                                    val += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                                if val != 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    
                    if valid:
                        return {"U": U_list, "V": V_list, "W": W_list}
            except Exception:
                pass
        
        return None

    # Try ranks from 55 down to 40
    for R in range(55, 39, -1):
        if time.time() > deadline - 20:
            break
        result = try_rank(R)
        if result is not None:
            best_result = result
            best_rank = R
            break

    return best_result
# EVOLVE-BLOCK-END
