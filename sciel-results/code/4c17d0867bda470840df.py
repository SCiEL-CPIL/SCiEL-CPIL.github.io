import numpy as np
from fractions import Fraction
import random
import time

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    num_a = n * m  # 16
    num_b = m * p  # 32
    num_c = p * n  # 32
    
    start_time = time.time()
    timeout = 115  # seconds
    
    # Try ranks from a lower bound up to schoolbook
    # Known lower bound for 4x4x8 is at least 48 (from 4x4x4 being 49, but let's be safe)
    # Actually, for 4x4x4, the minimum rank is 49. For 4x4x8, it should be higher.
    # Let's try a range.
    
    # First, let's try to use a structured approach with sparse random matrices
    # and solve the resulting linear system for W.
    
    # The key insight: for fixed U and V, the constraints on W are linear.
    # For each (a, b, c), we need sum_t U[a,t] * V[b,t] * W[c,t] = T[a,b,c]
    # This can be rewritten as: for each c, and for each (a,b), 
    # sum_t (U[a,t] * V[b,t]) * W[c,t] = T[a,b,c]
    
    # So for fixed U, V, we have a linear system for W.
    # The system is: A_c * W_c = rhs_c for each c
    # where A_c is a matrix with rows indexed by (a,b) and columns by t,
    # A_c[a,b,t] = U[a,t] * V[b,t]
    # and rhs_c[a,b] = T[a,b,c]
    
    # But this system is overdetermined (num_a * num_b = 512 equations, R unknowns)
    # So we need to find U, V such that this system is consistent for all c.
    
    # Alternative: Use the fact that the tensor has a special structure.
    # T[a,b,c] = 1 iff i_in == i_out and k_in == k_out and j_in == j_in'
    # where a = i_in*m + j_in, b = j_in'*p + k_in, c = k_out*n + i_out
    
    # Let's try a more systematic approach: use small integer entries for U and V,
    # and check if W can be solved exactly.
    
    # For each rank R, try random U, V with small integer entries
    for R in range(48, 81):
        if time.time() - start_time > timeout:
            break
        
        # Try many random initializations with small integers
        num_trials = 500
        for trial in range(num_trials):
            if time.time() - start_time > timeout:
                break
            
            # Use small integers to make exact arithmetic feasible
            # Random U, V with entries in {-2, -1, 0, 1, 2}
            U = np.random.randint(-2, 3, size=(num_a, R))
            V = np.random.randint(-2, 3, size=(num_b, R))
            
            # For each c, we need to solve for W[c, :]
            # The constraint is: for all (a,b), sum_t U[a,t] * V[b,t] * W[c,t] = T[a,b,c]
            
            # Let's build the linear system for W[c, :]
            # It's a system of num_a * num_b equations in R unknowns
            # But most equations are 0 = 0 (where T[a,b,c] = 0)
            # Only the equations where T[a,b,c] = 1 are non-trivial
            
            W = np.zeros((num_c, R))
            valid = True
            
            for c in range(num_c):
                i_out = c % n
                k_out = c // n
                
                # Build the linear system: find w such that
                # for all (a,b): sum_t U[a,t] * V[b,t] * w[t] = T[a,b,c]
                
                # Collect non-trivial equations (where T[a,b,c] = 1)
                # and trivial equations (where T[a,b,c] = 0)
                
                # For T[a,b,c] = 1: a = i_out*m + j, b = j*p + k_out
                # For T[a,b,c] = 0: all other (a,b)
                
                # Let's use least squares first to get a candidate, then verify exactly
                # Build the full matrix A (num_a * num_b) x R
                # A[row, t] where row = a * num_b + b, A[row, t] = U[a,t] * V[b,t]
                
                # This is large: 512 x R. Let's use a subset or solve directly.
                
                # Actually, let's just solve the system using all equations.
                # Build A and rhs
                A = np.zeros((num_a * num_b, R))
                rhs = np.zeros(num_a * num_b)
                
                for a in range(num_a):
                    for b in range(num_b):
                        row = a * num_b + b
                        A[row, :] = U[a, :] * V[b, :]
                        
                        # Compute T[a,b,c]
                        i_in = a // m
                        j_in = a % m
                        j_in_prime = b // p
                        k_in = b % p
                        
                        if i_in == i_out and k_in == k_out and j_in == j_in_prime:
                            rhs[row] = 1.0
                        else:
                            rhs[row] = 0.0
                
                # Solve A * W_c = rhs using least squares
                try:
                    W_c, residuals, rank_A, s = np.linalg.lstsq(A, rhs, rcond=1e-12)
                    # Check residual
                    residual = np.linalg.norm(A @ W_c - rhs)
                    if residual > 1e-8:
                        valid = False
                        break
                    W[c, :] = W_c
                except:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Verify the full decomposition exactly
            ok = True
            for a_idx in range(num_a):
                for b_idx in range(num_b):
                    for c_idx in range(num_c):
                        i_in = a_idx // m
                        j_in = a_idx % m
                        j_in_prime = b_idx // p
                        k_in = b_idx % p
                        k_out = c_idx // n
                        i_out = c_idx % n
                        
                        expected = 1.0 if (i_in == i_out and k_in == k_out and j_in == j_in_prime) else 0.0
                        val = np.dot(U[a_idx, :], V[b_idx, :] * W[c_idx, :])
                        if abs(val - expected) > 1e-8:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            # Success!
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
        
        print(f"Rank {R}: not found after {num_trials} trials")
    
    # Fallback to schoolbook
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
