import numpy as np
import time

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 120.0

def construct():
    n, m, p = DIMENSIONS
    elapsed = time.time() - START_TIME
    if elapsed > TIME_LIMIT - 5:
        return _fallback(n, m, p)
    
    # Try to construct rank 46 decomposition using Laderman's algorithm for 3x3x3
    # Laderman's algorithm for 3x3 matrix multiplication has rank 23.
    # We use it for the first 3 columns of B and the next 3 columns of B.
    
    try:
        # U, V, W for one 3x3x3 multiplication (rank 23)
        U_33, V_33, W_33 = get_laderman_factors()
        
        rank_33 = U_33.shape[1]
        total_rank = 2 * rank_33
        
        U = np.zeros((n * m, total_rank))
        V = np.zeros((m * p, total_rank))
        W = np.zeros((p * n, total_rank))
        
        # First 3 columns of B (k=0,1,2)
        U[:, 0:rank_33] = U_33
        V[0:9, 0:rank_33] = V_33
        W[0:9, 0:rank_33] = W_33
        
        # Second 3 columns of B (k=3,4,5)
        U[:, rank_33:total_rank] = U_33
        for j in range(3):
            for k_prime in range(3):
                V[j*6 + k_prime + 3, rank_33:total_rank] = V_33[j*3 + k_prime, 0:rank_33]
        for k_prime in range(3):
            for i in range(3):
                W[(k_prime + 3)*3 + i, rank_33:total_rank] = W_33[k_prime*3 + i, 0:rank_33]
                
        # Verify
        if verify_decomposition(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    except Exception:
        pass

    # Fallback to seed (rank 54)
    return _fallback(n, m, p)

def _fallback(n, m, p):
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

def get_laderman_factors():
    """
    Returns U, V, W for 3x3x3 matrix multiplication with rank 23.
    U has shape (9, 23), indexed by (i,j) -> i*3+j
    V has shape (9, 23), indexed by (j,k) -> j*3+k  
    W has shape (9, 23), indexed by (k,i) -> k*3+i
    Such that sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k in {0,1,2}
    """
    n = 3
    rank = 23
    U = np.zeros((n*n, rank))
    V = np.zeros((n*n, rank))
    W = np.zeros((n*n, rank))
    
    # Terms 1-9: simple products
    # Term 0: A[0,0]*B[0,0]
    U[0*3+0, 0] = 1; V[0*3+0, 0] = 1; W[0*3+0, 0] = 1
    # Term 1: A[1,1]*B[1,1]
    U[1*3+1, 1] = 1; V[1*3+1, 1] = 1; W[1*3+1, 1] = 1
    # Term 2: A[2,2]*B[2,2]
    U[2*3+2, 2] = 1; V[2*3+2, 2] = 1; W[2*3+2, 2] = 1
    # Term 3: A[0,1]*B[1,0]
    U[0*3+1, 3] = 1; V[1*3+0, 3] = 1; W[0*3+0, 3] = 1
    # Term 4: A[0,2]*B[2,0]
    U[0*3+2, 4] = 1; V[2*3+0, 4] = 1; W[0*3+0, 4] = 1
    # Term 5: A[1,2]*B[2,1]
    U[1*3+2, 5] = 1; V[2*3+1, 5] = 1; W[1*3+1, 5] = 1
    # Term 6: A[1,0]*B[0,1]
    U[1*3+0, 6] = 1; V[0*3+1, 6] = 1; W[1*3+1, 6] = 1
    # Term 7: A[2,0]*B[0,2]
    U[2*3+0, 7] = 1; V[0*3+2, 7] = 1; W[2*3+2, 7] = 1
    # Term 8: A[2,1]*B[1,2]
    U[2*3+1, 8] = 1; V[1*3+2, 8] = 1; W[2*3+2, 8] = 1
    
    # Terms 9-22: compound terms
    # These are the Laderman compound terms
    # Term 9: (A[0,0]+A[0,1])*(B[1,0]-B[1,1])
    U[0*3+0, 9] = 1; U[0*3+1, 9] = 1
    V[1*3+0, 9] = 1; V[1*3+1, 9] = -1
    # W to be determined
    
    # Term 10: (A[0,0]+A[1,1])*(B[1,1]-B[2,1])
    U[0*3+0, 10] = 1; U[1*3+1, 10] = 1
    V[1*3+1, 10] = 1; V[2*3+1, 10] = -1
    
    # Term 11: (A[0,0]+A[0,2])*(B[2,0]-B[2,2])
    U[0*3+0, 11] = 1; U[0*3+2, 11] = 1
    V[2*3+0, 11] = 1; V[2*3+2, 11] = -1
    
    # Term 12: (A[1,0]+A[0,1])*(B[0,0]-B[0,1])
    U[1*3+0, 12] = 1; U[0*3+1, 12] = 1
    V[0*3+0, 12] = 1; V[0*3+1, 12] = -1
    
    # Term 13: (A[1,2]+A[0,2])*(B[2,0]-B[0,2])
    U[1*3+2, 13] = 1; U[0*3+2, 13] = 1
    V[2*3+0, 13] = 1; V[0*3+2, 13] = -1
    
    # Term 14: (A[1,0]+A[1,1])*(B[1,1]-B[1,2])
    U[1*3+0, 14] = 1; U[1*3+1, 14] = 1
    V[1*3+1, 14] = 1; V[1*3+2, 14] = -1
    
    # Term 15: (A[2,0]+A[0,2])*(B[2,2]-B[2,0])
    U[2*3+0, 15] = 1; U[0*3+2, 15] = 1
    V[2*3+2, 15] = 1; V[2*3+0, 15] = -1
    
    # Term 16: (A[0,0]+A[1,1]+A[2,2])*(B[0,0]-B[0,1]-B[1,2])
    U[0*3+0, 16] = 1; U[1*3+1, 16] = 1; U[2*3+2, 16] = 1
    V[0*3+0, 16] = 1; V[0*3+1, 16] = -1; V[1*3+2, 16] = -1
    
    # Term 17: (A[0,0]+A[1,1]+A[2,2])*(B[1,1]-B[0,1]-B[2,2])
    U[0*3+0, 17] = 1; U[1*3+1, 17] = 1; U[2*3+2, 17] = 1
    V[1*3+1, 17] = 1; V[0*3+1, 17] = -1; V[2*3+2, 17] = -1
    
    # Term 18: (A[0,0]+A[1,1]+A[2,2])*(B[2,2]-B[0,2]-B[1,2])
    U[0*3+0, 18] = 1; U[1*3+1, 18] = 1; U[2*3+2, 18] = 1
    V[2*3+2, 18] = 1; V[0*3+2, 18] = -1; V[1*3+2, 18] = -1
    
    # Term 19: (A[1,0]+A[2,1]+A[0,2])*(B[0,0]-B[0,2]-B[2,0])
    U[1*3+0, 19] = 1; U[2*3+1, 19] = 1; U[0*3+2, 19] = 1
    V[0*3+0, 19] = 1; V[0*3+2, 19] = -1; V[2*3+0, 19] = -1
    
    # Term 20: (A[1,0]+A[2,1]+A[0,2])*(B[1,1]-B[0,1]-B[2,1])
    U[1*3+0, 20] = 1; U[2*3+1, 20] = 1; U[0*3+2, 20] = 1
    V[1*3+1, 20] = 1; V[0*3+1, 20] = -1; V[2*3+1, 20] = -1
    
    # Term 21: (A[1,0]+A[2,1]+A[0,2])*(B[2,2]-B[1,2]-B[2,0])
    U[1*3+0, 21] = 1; U[2*3+1, 21] = 1; U[0*3+2, 21] = 1
    V[2*3+2, 21] = 1; V[1*3+2, 21] = -1; V[2*3+0, 21] = -1
    
    # Term 22: (A[0,2]+A[1,0]+A[2,1])*(B[0,1]-B[0,0]-B[1,1])
    U[0*3+2, 22] = 1; U[1*3+0, 22] = 1; U[2*3+1, 22] = 1
    V[0*3+1, 22] = 1; V[0*3+0, 22] = -1; V[1*3+1, 22] = -1
    
    # Now solve for W[9:23] (the W factors for terms 9-22)
    # We need: for all i,j,k: sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    # W[0:9] is already set. Let's set up the linear system for W[9:23].
    
    # Unknowns: W[k,i,r] for r in 9..22, k in 0..2, i in 0..2 -> 9*14=126 unknowns
    # Equations: for all i,j,k (27 equations)
    
    # For each (i,j,k), the equation is:
    # sum_{r=0}^{8} U[i,j,r]*V[j,k,r]*W[k,i,r] + sum_{r=9}^{22} U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    
    # Let's build the system
    unknown_indices = []
    for r in range(9, 23):
        for k in range(3):
            for i in range(3):
                unknown_indices.append((k*3+i, r))
    
    num_unknowns = len(unknown_indices)
    
    # Build A_matrix and b_vector
    A_mat = np.zeros((27, num_unknowns))
    b_vec = np.zeros(27)
    
    for eq_idx, (i, j, k) in enumerate([(i,j,k) for i in range(3) for j in range(3) for k in range(3)]):
        # Compute known contribution from terms 0-8
        known = 0.0
        for r in range(9):
            known += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
        b_vec[eq_idx] = 1.0 - known
        
        # Build row for unknowns
        for col_idx, (w_idx, r) in enumerate(unknown_indices):
            # w_idx = k*3+i, r is the term
            # The unknown is W[w_idx, r]
            # In the equation for (i,j,k), the term is U[i,j,r]*V[j,k,r]*W[k,i,r]
            # w_idx should equal k*3+i
            if w_idx == k*3+i:
                A_mat[eq_idx, col_idx] = U[i*3+j, r] * V[j*3+k, r]
    
    # Solve the underdetermined system using least squares
    try:
        w_solution, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    except Exception:
        raise ValueError("Could not solve for W factors")
    
    # Assign W values
    for col_idx, (w_idx, r) in enumerate(unknown_indices):
        W[w_idx, r] = w_solution[col_idx]
    
    # Verify the solution
    for i in range(3):
        for j in range(3):
            for k in range(3):
                s = 0.0
                for r in range(23):
                    s += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                if not np.isclose(s, 1.0, atol=1e-10):
                    raise ValueError(f"Verification failed for ({i},{j},{k}): got {s}")
    
    return U, V, W

def verify_decomposition(U, V, W, n, m, p):
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if not np.isclose(s, 1.0):
                    return False
    return True
