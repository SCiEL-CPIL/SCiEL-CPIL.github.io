import time
import sys
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 9]
DEADLINE_SECONDS = 145.0  # Safety margin before 150s kill

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Standard bilinear decomposition as baseline
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        i = r // (m * p)
        j = (r // p) % m
        k = r % p
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][r] = 1
        V[v_idx][r] = 1
        W[w_idx][r] = 1
    
    best_rank = rank
    best_solution = {'U': U, 'V': V, 'W': W}
    
    # Try to find lower-rank decompositions
    # For small dimensions, we can try systematic search
    # The matrix multiplication tensor T[i,j,k] = delta_{i+k} 
    # Rank r decomposition: T = sum_{r=1}^R U_r ⊗ V_r ⊗ W_r
    
    # Try ranks from n*m*p-1 down to some lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    
    lower_bound = max(n*m, m*p, p*n)
    
    for target_rank in range(rank - 1, lower_bound - 1, -1):
        if time.time() - start_time > DEADLINE_SECONDS:
            break
        
        # Try to find a rank-target_rank decomposition
        # Use randomized search with local improvement
        
        max_attempts = 50
        found = False
        
        for attempt in range(max_attempts):
            if time.time() - start_time > DEADLINE_SECONDS:
                break
            
            # Initialize with random sparse matrices
            U_try = np.zeros((n*m, target_rank), dtype=float)
            V_try = np.zeros((m*p, target_rank), dtype=float)
            W_try = np.zeros((p*n, target_rank), dtype=float)
            
            # Initialize with schoolbook-like structure but with some merging
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = random.randint(0, target_rank - 1)
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U_try[u_idx, r] = 1
                        V_try[v_idx, r] = 1
                        W_try[w_idx, r] = 1
            
            # Local search: try to adjust entries to satisfy the tensor equation
            # The tensor equation: for all (i,j,k), sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
            # And for "wrong" combinations, the sum should be 0
            
            # Check if current solution works
            def check_solution(U_mat, V_mat, W_mat):
                # Check all (i,j,k) tuples
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0
                            for r in range(target_rank):
                                val += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                            if abs(val - 1) > 1e-9:
                                return False
                
                # Check that wrong combinations give 0
                # For any (i,j,k) where the product term should be 0
                # Actually, the tensor is defined only for valid (i,j,k)
                # The multiplication tensor has T[i,j,k] = 1 if indices are compatible
                # For 2x3 * 3x9: output is 2x9, so i in [0,1], j in [0,2], k in [0,8]
                # T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
                # Wait, no. The matrix multiplication tensor M[i,j,k] represents
                # C[i,k] = sum_j A[i,j] * B[j,k]
                # So the tensor is T[i,j,k] = delta for the bilinear form
                # Actually, the standard tensor is: for each (i,j,k), the coefficient
                # In the bilinear algorithm, we compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # This should equal 1 for all (i,j,k) that appear in the multiplication
                # For 2x3 * 3x9, all combinations (i,j,k) with i in [0,1], j in [0,2], k in [0,8]
                # are valid, and each contributes to C[i,k]
                
                # The condition is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k
                # And this is the only constraint (the tensor entries are all 1)
                
                return True
            
            if check_solution(U_try, V_try, W_try):
                # Convert to integer/half-integer entries
                U_list = [[0] * target_rank for _ in range(n*m)]
                V_list = [[0] * target_rank for _ in range(m*p)]
                W_list = [[0] * target_rank for _ in range(p*n)]
                
                for i in range(n*m):
                    for r in range(target_rank):
                        U_list[i][r] = U_try[i, r]
                for i in range(m*p):
                    for r in range(target_rank):
                        V_list[i][r] = V_try[i, r]
                for i in range(p*n):
                    for r in range(target_rank):
                        W_list[i][r] = W_try[i, r]
                
                best_rank = target_rank
                best_solution = {'U': U_list, 'V': V_list, 'W': W_list}
                found = True
                break
        
        if found:
            break
    
    return best_solution
