"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = 3, 3, 6
    
    # Build the target tensor: T[i,j,k] = 1 if we want C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor for matrix multiplication A(n,m) x B(m,p) -> C(n,p)
    # has entries T[i,j,k] = delta_{i,k} in the standard basis, but we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] = W[k*n+i, c] for all i,j,k
    # Actually, the decomposition requires:
    # For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] = delta_{i,k} (if we're doing C=A*B)
    # Wait, let me reconsider. The matrix multiplication tensor T has:
    # T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the "multiplication" sense)
    # No: T[i,j,k] = delta_{i,k} is wrong.
    # 
    # The correct formulation: We want to compute C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is defined such that C[i,k] = sum_{j,c} A[i,j] * B[j,k] * ... 
    # Actually, the tensor decomposition of matrix multiplication satisfies:
    # sum_c U[i*m+j, c] * V[j*p+k, c] = delta_{i,k} * (something)
    # 
    # Let me think again. The matrix multiplication tensor is T in R^{n x m x p} where
    # T[i,j,k] = 1 if and only if the product A[i,j]*B[j,k] contributes to C[i,k].
    # Since every A[i,j]*B[j,k] contributes to C[i,k] (with i fixed and k fixed),
    # T[i,j,k] = 1 for all i,j,k.
    # 
    # No wait. The tensor T has entries T[i,j,k] which represents the coefficient
    # such that C[i,k] = sum_{j} sum_{c} ... 
    # 
    # Standard formulation: The multiplication tensor T_{n,m,p} has entries
    # T[i,j,k] = delta_{i,k} ... no.
    # 
    # Let me use the standard: We decompose the tensor T where
    # T[i,j,k] = 1 for all i,j,k (since A[i,j]*B[j,k] always contributes to C[i,k])
    # But that's not right either because different (i,j,k) contribute to different C[i,k].
    #
    # Correct: The tensor T has T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B
    # multiply to contribute to C[i,k]. Since A[i,j]*B[j,k] contributes to C[i,k],
    # T[i,j,k] = 1 for all valid i,j,k.
    #
    # The rank-r decomposition means: there exist U (nm x r), V (mp x r), W (pn x r) such that
    # for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] = T[i,j,k] * (contribution to W)
    # 
    # Actually the standard is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # No, that's not right.
    #
    # The correct tensor decomposition: T is a 3-tensor with T[i,j,k] and we want
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But T[i,j,k] for matrix mult is: T[i,j,k] = delta_{i,k}? No.
    #
    # Let me look at this differently. For 2x2x2:
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    #
    # The tensor T has T[i,j,k] = 1 for all i,j,k in {0,1}^3 (8 entries all 1).
    # The rank-7 Strassen decomposition: sum_c U[i,j,c]*V[j,k,c]*W[i,k,c] = T[i,j,k]
    # where W[i,k,c] is indexed by output position.
    #
    # So for our case: T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,6)
    # We need: sum_c U[i*3+j, c] * V[j*6+k, c] * W[k*3+i, c] = 1 for all i,j,k
    #
    # This is a system of 3*3*6 = 54 equations.
    
    # Target: T[i,j,k] = 1 for all i,j,k
    # Constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # First, let's return the schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    def verify(U, V, W, tol=1e-9):
        """Check if the decomposition is valid."""
        if time.time() > deadline:
            return False
        rank = len(U[0])
        # Check all i,j,k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def try_random_search(max_rank, num_trials=1000):
        """Try random search for a valid decomposition."""
        nonlocal best_result, best_rank
        for trial in range(num_trials):
            if time.time() > deadline - 10.0:
                break
            # Try random small integer/half-integer values
            scale = random.choice([1, 2, 3])
            U = [[random.choice([-scale, -scale+0.5, 0, 0.5, 1, 1.5, scale]) for _ in range(max_rank)] for _ in range(n*m)]
            V = [[random.choice([-scale, -scale+0.5, 0, 0.5, 1, 1.5, scale]) for _ in range(max_rank)] for _ in range(m*p)]
            W = [[random.choice([-scale, -scale+0.5, 0, 0.5, 1, 1.5, scale]) for _ in range(max_rank)] for _ in range(p*n)]
            
            if verify(U, V, W):
                if max_rank < best_rank:
                    best_rank = max_rank
                    best_result = {"U": U, "V": V, "W": W}
                return True
        return False
    
    # Try to find lower rank decompositions
    # Start from lower ranks and go up
    for rank in range(20, best_rank):
        if time.time() > deadline - 20.0:
            break
        if try_random_search(rank, num_trials=200):
            # Found a valid decomposition at this rank
            break
    
    # If we found something better, return it
    if best_rank < n * m * p:
        return best_result
    
    # Otherwise return schoolbook
    return best_result

# EVOLVE-BLOCK-END
