import numpy as np
import time
import random

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 32
    pn = p * n  # 16
    
    # Schoolbook as fallback
    rank_schoolbook = n * m * p  # 64
    U_school = np.zeros((nm, rank_schoolbook))
    V_school = np.zeros((mp, rank_schoolbook))
    W_school = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    def check_solution(U, V, W):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        r = U.shape[1]
        # For each output (i,k), compute the bilinear form
        # C[i,k] = sum_c W[k*n+i, c] * (sum_{i',j'} U[i'*m+j', c] * A[i',j']) * (sum_{j'',k''} V[j''*p+k'', c] * B[j'',k''])
        # The coefficient of A[i',j']*B[j'',k''] in C[i,k] is sum_c W[k*n+i, c] * U[i'*m+j', c] * V[j''*p+k'', c]
        # This must equal delta_{i',i} * delta_{j',j''} * delta_{k'',k}
        for i in range(n):
            for k in range(p):
                for ip in range(n):
                    for jp in range(m):
                        for jpp in range(m):
                            for kpp in range(p):
                                expected = 1 if (ip == i and jp == jpp and kpp == k) else 0
                                actual = 0
                                for c in range(r):
                                    actual += W[k*n + i, c] * U[ip*m + jp, c] * V[jpp*p + kpp, c]
                                if abs(actual - expected) > 1e-9:
                                    return False
        return True
    
    # Try to find lower rank solutions
    # Start from rank 32 and go up
    start_rank = 32
    max_rank = 64
    deadline = time.time() + 110  # 110 second deadline
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_schoolbook
    
    np.random.seed(42)
    
    for rank in range(start_rank, max_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            U = np.random.randint(-2, 3, size=(nm, rank))
            V = np.random.randint(-2, 3, size=(mp, rank))
            W = np.random.randint(-2, 3, size=(pn, rank))
            
            # Alternating least squares
            for iteration in range(200):
                if time.time() > deadline:
                    break
                
                # Fix U and V, solve for W
                # For each (i,k), we need: sum_c W[k*n+i, c] * (U[:, c]^T * V[:, c] mapped appropriately) = target
                # This is linear in W for fixed U, V
                
                # Build the linear system for W
                # For each output (i,k) and each input pair (ip,jp,jpp,kpp):
                # sum_c W[k*n+i, c] * U[ip*m+jp, c] * V[jpp*p+kpp, c] = delta_{ip,i} * delta_{jp,jpp} * delta_{kpp,k}
                
                # Group by (i,k): for each (i,k), we have equations in W[k*n+i, :]
                # Let M_{(i,k)} be a matrix where M_{(i,k), (ip,jp,jpp,kpp), c} = U[ip*m+jp, c] * V[jpp*p+kpp, c]
                # Then M_{(i,k)} @ W[k*n+i, :] = target_{(i,k)}
                
                solved = True
                new_W = W.copy()
                for i in range(n):
                    for k in range(p):
                        # Build the matrix for this (i,k)
                        # Rows: all (ip,jp,jpp,kpp) combinations
                        # Columns: c = 0..rank-1
                        A_mat = np.zeros((n*m*m*p, rank))
                        target = np.zeros(n*m*m*p)
                        row = 0
                        for ip in range(n):
                            for jp in range(m):
                                for jpp in range(m):
                                    for kpp in range(p):
                                        for c in range(rank):
                                            A_mat[row, c] = U[ip*m + jp, c] * V[jpp*p + kpp, c]
                                        target[row] = 1 if (ip == i and jp == jpp and kpp == k) else 0
                                        row += 1
                        
                        # Solve A_mat @ W[k*n+i, :] = target
                        # Use least squares
                        try:
                            sol, residuals, rank_mat, sv = np.linalg.lstsq(A_mat, target, rcond=None)
                            new_W[k*n + i, :] = sol
                        except:
                            solved = False
                            break
                    if not solved:
                        break
                
                W = new_W
                
                # Fix V and W, solve for U
                new_U = U.copy()
                for ip in range(n):
                    for jp in range(m):
                        A_mat = np.zeros((n*p*m*p, rank))
                        target = np.zeros(n*p*m*p)
                        row = 0
                        for i in range(n):
                            for k in range(p):
                                for jpp in range(m):
                                    for kpp in range(p):
                                        for c in range(rank):
                                            A_mat[row, c] = W[k*n + i, c] * V[jpp*p + kpp, c]
                                        target[row] = 1 if (ip == i and jp == jpp and kpp == k) else 0
                                        row += 1
                        
                        try:
                            sol, residuals, rank_mat, sv = np.linalg.lstsq(A_mat, target, rcond=None)
                            new_U[ip*m + jp, :] = sol
                        except:
                            solved = False
                            break
                    if not solved:
                        break
                
                U = new_U
                
                # Fix U and W, solve for V
                new_V = V.copy()
                for jpp in range(m):
                    for kpp in range(p):
                        A_mat = np.zeros((n*p*n*m, rank))
                        target = np.zeros(n*p*n*m)
                        row = 0
                        for i in range(n):
                            for k in range(p):
                                for ip in range(n):
                                    for jp in range(m):
                                        for c in range(rank):
                                            A_mat[row, c] = W[k*n + i, c] * U[ip*m + jp, c]
                                        target[row] = 1 if (ip == i and jp == jpp and kpp == k) else 0
                                        row += 1
                        
                        try:
                            sol, residuals, rank_mat, sv = np.linalg.lstsq(A_mat, target, rcond=None)
                            new_V[jpp*p + kpp, :] = sol
                        except:
                            solved = False
                            break
                    if not solved:
                        break
                
                V = new_V
                
                # Check if we have a solution
                if check_solution(U, V, W):
                    # Round to nearest integers/half-integers
                    U_round = np.round(U * 2) / 2
                    V_round = np.round(V * 2) / 2
                    W_round = np.round(W * 2) / 2
                    if check_solution(U_round, V_round, W_round):
                        best_U = U_round
                        best_V = V_round
                        best_W = W_round
                        best_rank = rank
                        break
                    else:
                        # Keep the floating point solution
                        best_U = U
                        best_V = V
                        best_W = W
                        best_rank = rank
                        break
                
                if not solved:
                    break
            
            if best_rank == rank:
                break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
