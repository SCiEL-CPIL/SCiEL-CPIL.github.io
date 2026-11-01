"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor: T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Schoolbook as fallback
    schoolbook_rank = n * m * p
    U_school = [[0] * schoolbook_rank for _ in range(nm)]
    V_school = [[0] * schoolbook_rank for _ in range(mp)]
    W_school = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower-rank decompositions using ALS
    # Try ranks from schoolbook-1 down to some minimum
    for rank in range(schoolbook_rank - 1, 15, -1):
        if time.time() > deadline - 5:
            break
        
        found = False
        # Try multiple random initializations
        for trial in range(20):
            if time.time() > deadline - 3:
                break
            
            np.random.seed(trial * 1000 + rank)
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # ALS: alternate fixing two matrices and solving for the third
            for iteration in range(500):
                if time.time() > deadline - 3:
                    break
                
                # Update U: for each row i*m+j, solve linear system
                # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # For fixed i*m+j row, this is: sum_k [sum_r U[row, r] * V[j*p+k, r] * W[k*n+i, r]] 
                # Actually, for fixed (i,j), we need: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for all k
                # This is: U[i*m+j, :] @ diag(V[j*p+k, r] * W[k*n+i, r]) ... not quite
                # Let me reformulate: for fixed i,j, define for each k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # This is a linear system in U[i*m+j, :]: 
                # For each k: sum_r U[i*m+j,r] * (V[j*p+k,r] * W[k*n+i,r]) = 1
                # So: A @ U[i*m+j, :]^T = 1 where A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        row = i * m + j
                        # A[k, r] = V[j*p+k, r] * W[k*n+i, r]
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve A @ u = ones(p)
                        try:
                            u = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            u = np.zeros(rank)
                        U[row, :] = u
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        row = j * p + k
                        # A[i, r] = U[i*m+j, r] * W[k*n+i, r]
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            v = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            v = np.zeros(rank)
                        V[row, :] = v
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        row = k * n + i
                        # A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            w = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        except:
                            w = np.zeros(rank)
                        W[row, :] = w
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-10:
                    found = True
                    break
            
            if found:
                break
        
        if found:
            # Convert to list format
            best_rank = rank
            best_U = U.tolist()
            best_V = V.tolist()
            best_W = W.tolist()
            break  # Found the lowest rank we tried
    
    # Ensure we have a valid result
    if best_rank == schoolbook_rank:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

