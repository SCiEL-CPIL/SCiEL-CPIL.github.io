"""Matrix multiplication 3,4,5 rank-minimal decomposition."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor T[a, b, c_idx] = 1 if a=i*m+j, b=j*p+k, c_idx=k*n+i
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                T[a, b, c_idx] = 1.0
    
    # Try ranks from low to high
    # Known: schoolbook is 60. Let's try to find lower.
    # For matrix multiplication, the minimum rank is at least max(n*p, m*p, n*m) = max(15, 20, 12) = 20
    # But actually the tensor rank is harder to bound. Let's try from 30 upward.
    
    best_result = None
    best_rank = nm * mp  # schoolbook
    
    # First, get the schoolbook solution as fallback
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_result = (U_school, V_school, W_school)
    best_rank = rank_school
    
    # Try to find lower rank solutions
    # Strategy: for a given rank r, randomly initialize U and V, then for each c_idx,
    # solve the linear system for W[c_idx, :]. Check consistency.
    
    for rank in range(30, best_rank):  # try ranks 30 to best_rank-1
        if time.time() > deadline:
            break
        
        for trial in range(200):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            U = np.random.randint(-2, 3, size=(nm, rank)).astype(np.float64)
            V = np.random.randint(-2, 3, size=(mp, rank)).astype(np.float64)
            
            # For each c_idx, we need to solve for W[c_idx, :]
            # The equation is: for all (a,b), sum_c U[a,c]*V[b,c]*W[c_idx,c] = T[a,b,c_idx]
            # This is a linear system: M[c_idx] @ W[c_idx,:]^T = T[:, :, c_idx].flatten()
            # where M[c_idx][a*mp+b, c] = U[a,c] * V[b,c]
            
            W = np.zeros((pn, rank))
            valid = True
            
            for c_idx in range(pn):
                # Build the linear system matrix
                M = np.zeros((nm * mp, rank))
                rhs = T[:, :, c_idx].flatten()
                
                for a in range(nm):
                    for b in range(mp):
                        for c in range(rank):
                            M[a * mp + b, c] = U[a, c] * V[b, c]
                
                # Solve M @ W[c_idx,:]^T = rhs
                # This is overdetermined (nm*mp equations, rank unknowns)
                # Use least squares, then check residual
                try:
                    sol, residuals, rank_s, sv = np.linalg.lstsq(M, rhs, rcond=None)
                    W[c_idx, :] = sol
                    
                    # Check if residual is small
                    residual = np.linalg.norm(M @ sol - rhs)
                    if residual > 1e-8:
                        valid = False
                        break
                except:
                    valid = False
                    break
            
            if valid:
                # Verify the full decomposition
                # Check: sum_c U[a,c]*V[b,c]*W[c_idx,c] == T[a,b,c_idx] for all a,b,c_idx
                max_err = 0
                for c_idx in range(pn):
                    # Compute the full product
                    prod = np.zeros((nm, mp))
                    for c in range(rank):
                        prod += U[:, c] * V[:, c] * W[c_idx, c]
                    err = np.max(np.abs(prod - T[:, :, c_idx]))
                    max_err = max(max_err, err)
                
                if max_err < 1e-8:
                    # Found a valid decomposition!
                    # Try to clean up to rational/integer entries
                    U_clean = np.round(U * 2) / 2
                    V_clean = np.round(V * 2) / 2
                    W_clean = np.round(W * 2) / 2
                    
                    # Verify cleaned version
                    max_err2 = 0
                    for c_idx in range(pn):
                        prod = np.zeros((nm, mp))
                        for c in range(rank):
                            prod += U_clean[:, c] * V_clean[:, c] * W_clean[c_idx, c]
                        err = np.max(np.abs(prod - T[:, :, c_idx]))
                        max_err2 = max(max_err2, err)
                    
                    if max_err2 < 1e-8:
                        best_result = (U_clean, V_clean, W_clean)
                        best_rank = rank
                        break  # Found lower rank, break trial loop
                    else:
                        # Use the uncleaned version
                        best_result = (U, V, W)
                        best_rank = rank
                        break
            # Continue trying if not valid
    
    if time.time() > deadline:
        pass  # Return best found
    
    U, V, W = best_result
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            result.append([float(x) for x in row])
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
