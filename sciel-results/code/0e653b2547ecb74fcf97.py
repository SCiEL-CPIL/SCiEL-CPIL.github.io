<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill

    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W

    def validate(U_mat, V_mat, W_mat):
        """Check if the decomposition exactly represents matrix multiplication."""
        n, m, p = DIMENSIONS
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        rank = U_arr.shape[1]

        # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    # Try to find a lower rank using structured random search
    # The key insight: if we can find linear dependencies among the schoolbook terms,
    # we can reduce the rank.
    
    # For now, attempt a simple improvement: check if any column is redundant
    # This is a heuristic - in general, finding the minimal rank is NP-hard.
    
    # Try small rank reductions by zeroing out columns and checking validity
    for reduction in range(1, 10):  # Try to reduce rank by 1 to 9
        if time.time() > deadline - 60:
            break
        if rank_schoolbook - reduction < 25:  # Don't go below theoretical lower bound
            break
        
        target_rank = rank_schoolbook - reduction
        # Remove the last 'reduction' columns
        U_red = [row[:target_rank] for row in U]
        V_red = [row[:target_rank] for row in V]
        W_red = [row[:target_rank] for row in W]
        
        if validate(U_red, V_red, W_red):
            best_rank = target_rank
            best_U, best_V, best_W = U_red, V_red, W_red
            break  # Found a valid reduction, stop
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from itertools import combinations

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill

    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W

    def validate(U_mat, V_mat, W_mat):
        """Check if the decomposition exactly represents matrix multiplication."""
        n, m, p = DIMENSIONS
        U_arr = np.array(U_mat, dtype=float)
        V_arr = np.array(V_mat, dtype=float)
        W_arr = np.array(W_mat, dtype=float)
        rank = U_arr.shape[1]

        # Check: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                    if abs(s - 1.0) > 1e-6:
                        return False
        return True

    # Strategy: Try to find linear dependencies among the schoolbook terms
    # by attempting to express some terms as linear combinations of others.
    
    # The tensor T[i,j,k] = 1 for all i,j,k (since we're multiplying identity-like structures in this simplified view)
    # Actually, the matrix multiplication tensor is T[i,j,k] = 1 if we think of it as 
    # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # So the tensor has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # For this specific case (all entries are 1), we can potentially find a lower rank.
    # Let's try a systematic search for lower rank decompositions.
    
    # First, let's check if there's a trivial lower rank.
    # For a 3x5x5 tensor with all 1s, the rank is at most min(3*5, 5*5, 3*5) = 15
    # But actually, for all-ones tensor, the rank is 1 if we allow scaling.
    # Wait, no. The tensor is T[i,j,k] = 1.
    # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all i,j,k.
    
    # Let's try rank 1: U[i,j,0] * V[j,k,0] * W[k,i,0] = 1 for all i,j,k
    # This implies U[i,j,0] = 1/a_{ij}, V[j,k,0] = 1/b_{jk}, W[k,i,0] = 1/c_{ki}
    # and a_{ij} * b_{jk} * c_{ki} = 1 for all i,j,k.
    # This is possible if we set a_{ij} = 1, b_{jk} = 1, c_{ki} = 1.
    # So rank 1 works! But wait, the entries must be integer or half-integer.
    # 1 is an integer, so this should work.
    
    # Let's verify: U[i,j,0] = 1, V[j,k,0] = 1, W[k,i,0] = 1
    # Then sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1*1*1 = 1 for all i,j,k.
    # This is valid!
    
    U_rank1 = [[1] for _ in range(n * m)]
    V_rank1 = [[1] for _ in range(m * p)]
    W_rank1 = [[1] for _ in range(p * n)]
    
    if validate(U_rank1, V_rank1, W_rank1):
        best_rank = 1
        best_U, best_V, best_W = U_rank1, V_rank1, W_rank1
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # If rank 1 doesn't work (shouldn't happen for all-ones), try rank 2, 3, etc.
    # But first, let's try a more systematic approach.
    
    # For small ranks, we can try random initializations and gradient descent.
    # But since we need exact integer/half-integer entries, let's try small integer values.
    
    # Try ranks from 2 to min(rank_schoolbook, 20)
    max_rank_to_try = min(rank_schoolbook, 20)
    
    for target_rank in range(2, max_rank_to_try + 1):
        if time.time() > deadline - 30:
            break
        
        # Try random integer/half-integer entries
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            # Generate random entries from {-2, -1, -0.5, 0, 0.5, 1, 2}
            U_try = [[random.choice([-2, -1, -0.5, 0, 0.5, 1, 2]) for _ in range(target_rank)] 
                    for _ in range(n * m)]
            V_try = [[random.choice([-2, -1, -0.5, 0, 0.5, 1, 2]) for _ in range(target_rank)] 
                    for _ in range(m * p)]
            W_try = [[random.choice([-2, -1, -0.5, 0, 0.5, 1, 2]) for _ in range(target_rank)] 
                    for _ in range(p * n)]
            
            if validate(U_try, V_try, W_try):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break
        
        if best_rank == target_rank:
            break  # Found a valid decomposition at this rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
