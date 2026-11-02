"""Matrix multiplication 3,4,5 rank-minimal decomposition."""
import numpy as np
import time
import sys
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
    
    # Schoolbook solution (rank 60) as fallback
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
    
    # Try to find a lower rank decomposition
    # The minimum rank for 3x4x5 matrix multiplication is known to be 49
    # Let's try from 49 upward to find the best we can
    
    best_result = (U_school, V_school, W_school)
    best_rank = rank_school
    
    # For the specific case of 3x4x5, we know that rank 49 is achievable
    # Let's try to construct it or find it via random search
    
    # Strategy: Randomly initialize U and V, then for each c_idx solve for W
    # But we need to ensure exact decomposition
    
    # Actually, let's try a different approach:
    # Use the known fact that the border rank of 3x4x5 is 49
    # We'll do a local search starting from random initializations
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W form an exact decomposition of the tensor"""
        max_err = 0
        for c_idx in range(pn):
            prod = np.zeros((nm, mp))
            for c in range(rank):
                prod += U[:, c] * V[:, c] * W[c_idx, c]
            err = np.max(np.abs(prod - T[:, :, c_idx]))
            max_err = max(max_err, err)
        return max_err
    
    # Try to find rank 49 solution
    # This is a well-known result: the tensor rank of 3x4x5 matrix multiplication is 49
    
    # Let's try random initialization and solve for W
    for rank in range(49, 60):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(500):
            if time.time() > deadline:
                break
            
            # Random initialization with small integers
            U = np.random.randint(-3, 4, size=(nm, rank)).astype(np.float64)
            V = np.random.randint(-3, 4, size=(mp, rank)).astype(np.float64)
            
            W = np.zeros((pn, rank))
            valid = True
            
            for c_idx in range(pn):
                # Build the linear system: for each (a,b), sum_c U[a,c]*V[b,c]*W[c_idx,c] = T[a,b,c_idx]
                # M[c_idx][a*mp+b, c] = U[a,c] * V[b,c]
                M = np.zeros((nm * mp, rank))
                rhs = T[:, :, c_idx].flatten()
                
                for a in range(nm):
                    for b in range(mp):
                        for c in range(rank):
                            M[a * mp + b, c] = U[a, c] * V[b, c]
                
                # Solve the overdetermined system
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
                max_err = verify_decomposition(U, V, W, rank)
                if max_err < 1e-8:
                    # Found a valid decomposition!
                    # Try to clean up to rational/integer entries
                    # Round to nearest 0.5
                    U_clean = np.round(U * 2) / 2
                    V_clean = np.round(V * 2) / 2
                    W_clean = np.round(W * 2) / 2
                    
                    # Verify cleaned version
                    max_err2 = verify_decomposition(U_clean, V_clean, W_clean, rank)
                    if max_err2 < 1e-8:
                        best_result = (U_clean, V_clean, W_clean)
                        best_rank = rank
                        found = True
                        break
                    else:
                        # Use the uncleaned version
                        best_result = (U, V, W)
                        best_rank = rank
                        found = True
                        break
        
        if found:
            break
    
    U, V, W = best_result
    
    # Convert to lists of lists with integer/half-integer entries
    def to_list(mat):
        result = []
        for row in mat:
            result.append([float(x) for x in row])
        return result
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
