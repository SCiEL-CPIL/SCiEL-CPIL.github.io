import numpy as np
import time
import random

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Schoolbook rank
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank decompositions using random projections
    # Strategy: randomly combine schoolbook terms and check if rank can be reduced
    
    def validate(U, V, W):
        """Validate that U, V, W correctly decompose the matrix multiplication tensor"""
        rank = len(U[0]) if U else 0
        # Check dimensions
        if len(U) != n * m:
            return False
        if len(V) != m * p:
            return False
        if len(W) != p * n:
            return False
        for r in range(rank):
            if len(U[r]) != rank:
                return False
            if len(V[r]) != rank:
                return False
            if len(W[r]) != rank:
                return False
        
        # Verify: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] should equal delta(i,j,k) for the tensor
        # The tensor T[i,j,k] = 1 if this is a valid multiplication term, else 0
        # Actually, the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix multiplication, T[i,j,k] = 1 for all i,j,k (each output C[i,k] = sum_j A[i,j]*B[j,k])
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try random rank reduction attempts
    # For each attempt, we'll try to find a rank-r decomposition by random sampling
    
    def try_find_decomposition(target_rank, max_iterations=50):
        """Try to find a rank decomposition of the given target rank"""
        if target_rank >= rank_schoolbook:
            return None
        
        for attempt in range(max_iterations):
            if time.time() > deadline:
                return None
            
            # Randomly initialize the factor matrices
            U_rand = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n * m)]
            V_rand = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m * p)]
            W_rand = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Check if this random initialization satisfies the constraints
            # This is a very rough check - we need sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for r in range(target_rank):
                            val += U_rand[i * m + j][r] * V_rand[j * p + k][r] * W_rand[k * n + i][r]
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return (U_rand, V_rand, W_rand)
        
        return None
    
    # Try to find decompositions with lower rank
    for target_rank in range(62, 55, -1):
        if time.time() > deadline:
            break
        
        result = try_find_decomposition(target_rank, max_iterations=20)
        if result is not None:
            U_found, V_found, W_found = result
            if validate(U_found, V_found, W_found):
                best_rank = target_rank
                best_U, best_V, best_W = U_found, V_found, W_found
                break
    
    if best_rank == rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": best_U, "V": best_V, "W": best_W}
