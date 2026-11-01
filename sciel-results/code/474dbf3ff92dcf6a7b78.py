import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin from 150s limit

    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 42

    # Trivial decomposition as fallback
    U_triv = [[0] * trivial_rank for _ in range(n * m)]
    V_triv = [[0] * trivial_rank for _ in range(m * p)]
    W_triv = [[0] * trivial_rank for _ in range(p * n)]

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i * m + j][col] = 1
                V_triv[j * p + k][col] = 1
                W_triv[k * n + i][col] = 1
                col += 1

    best_rank = trivial_rank
    best_result = {"U": U_triv, "V": V_triv, "W": W_triv}

    def check_decomposition(U, V, W, rank):
        """Verify that sum_t U[i*m+j][t]*V[j*p+k][t]*W[k*n+i][t] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if s != 1:
                        return False
        return True

    # Strategy: exploit m=3 by treating the computation as
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0..6}
    # 
    # For each k, we compute a 2x1 vector from 2x3 * 3x1.
    # The bilinear map for one k has 6 inputs to 2 outputs.
    # We can try to find a shared decomposition across all k's.
    #
    # Approach: Try ranks from 21 (lower bound) upward, using
    # random search with half-integer entries and verification.
    
    # First, try a structured Winograd-like approach for m=3
    # For multiplying (2x3) by (3x7), we can group the 3 intermediate
    # terms and use the fact that we only have 2 rows.
    
    # Let's try rank 30, 28, 25, 21 with random search
    # Using half-integer entries (multiply by 2 to work with integers)
    
    def check_decomposition(U, V, W, rank):
        """Verify that sum_t U[i*m+j][t]*V[j*p+k][t]*W[k*n+i][t] == 1 for all i,j,k"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if s != 1:
                        return False
        return True

    # Try to find lower rank decompositions using random search
    # Focus on ranks that might be achievable: 35, 30, 28, 25, 21
    
    target_ranks = [35, 32, 30, 28, 25, 22, 21]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
        
        # Multiple random attempts for each target rank
        max_attempts = 500 if target_rank >= 30 else 2000
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random half-integer values (as integers * 0.5)
            # Use small integers for efficiency
            U_r = [[random.randint(-3, 3) for _ in range(target_rank)] for _ in range(n * m)]
            V_r = [[random.randint(-3, 3) for _ in range(target_rank)] for _ in range(m * p)]
            W_r = [[random.randint(-3, 3) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Scale: we want sum of products = 1, so let's work with integers
            # and check if sum == 1 (or adjust scaling)
            
            # Quick check: compute tensor contraction
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for t in range(target_rank):
                            s += U_r[i * m + j][t] * V_r[j * p + k][t] * W_r[k * n + i][t]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                break
        
        if best_rank < trivial_rank:
            break

    # If random search didn't find anything, try a more structured approach
    # based on the block decomposition for m=3
    
    if best_rank == trivial_rank and time.time() < deadline - 20:
        # Try structured approach: for each k, the 2x3 * 3x1 multiplication
        # can be done with at most 6 mults, but we can share across k's.
        # 
        # Idea: Precompute combinations of A's columns that are shared.
        # Let A = [a0, a1, a2] where aj is a 2x1 column.
        # C[i,k] = a0[i]*B[0,k] + a1[i]*B[1,k] + a2[i]*B[2,k]
        #
        # We can precompute: s0 = a0+a1, s1 = a0-a1, s2 = a2, etc.
        # But this doesn't directly reduce the bilinear rank unless we
        # can express the result using fewer products.
        
        # Try rank 35: remove 7 terms by merging where possible
        # For each i (2 rows), we have 3*7=21 products. Total 42.
        # If we can share some products between the two rows, we save.
        
        # Specifically: C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
        #              C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
        #
        # Let x_j = A[0,j] + A[1,j], y_j = A[0,j] - A[1,j]
        # Then C[0,k] + C[1,k] = sum_j x_j * B[j,k]
        #      C[0,k] - C[1,k] = sum_j y_j * B[j,k]
        # This uses 6 products per k instead of 6, no savings.
        
        # The minimum rank for 2x3x7 is likely 42 or close to it.
        # Let's just return the best we found.
        pass

    return best_result
