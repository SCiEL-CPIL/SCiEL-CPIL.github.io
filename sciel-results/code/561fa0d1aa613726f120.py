"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]
import time
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Schoolbook as fallback
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
    
    # Try to find lower rank via finite field search
    # The tensor is T[i,j,k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We need: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # And for all other (i,j,k) combinations that don't correspond to valid products, sum = 0
    
    # Actually, the tensor entries are: T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k]
    # The decomposition must satisfy: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = delta_{valid}
    
    # For 4x4 * 4x5, the valid (i,j,k) triples are all i in [0,4), j in [0,4), k in [0,5)
    # So actually ALL combinations are valid - this is the full multiplication tensor
    
    # Let's search for rank r decompositions over small fields
    # We'll try GF(2), GF(3), etc.
    
    best_result = None
    best_rank = rank_schoolbook
    
    # Known lower bound: for 4x4*4x5, we can try ranks from 40 down
    # Theoretical lower bound is around n*m*p/2 for some cases
    
    # Try a structured search: use the fact that we can have entries from {-1, 0, 1}
    # and search for sparse solutions
    
    # First, let's try to find a rank-40 or lower solution using algebraic methods
    # For small matrices, we can use the "winograd" style optimizations
    
    # Winograd optimization for 4x4 * 4x5:
    # We can use tensor rank reductions from known algorithms
    
    # Let's try a direct search for rank r where r < 80
    # We'll use the fact that the decomposition must satisfy bilinear constraints
    
    # Actually, let's use a smarter approach: search over small ranks with structured U,V,W
    # where entries are from {-1, 0, 1}
    
    # For rank r, we need:
    # For all (i,j,k): sum_{t=0}^{r-1} U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    
    # This is a system of equations in the products U*V*W
    
    # Let's try rank 64, 60, 56, etc. with a greedy/optimization approach
    
    # Actually, for this specific case, let's use known results:
    # The rank of 4x4*4x5 matrix multiplication is known to be at most 70 (from literature)
    # Let's try to find a rank-70 or better solution
    
    # Use a structured approach: partition the computation into blocks
    
    # For now, let's implement a search that tries to find rank r solutions
    # where r is in a reasonable range
    
    # Try ranks from 50 to 79
    for target_rank in range(50, 80):
        if time.time() > deadline - 10:
            break
        if target_rank >= best_rank:
            break
        
        # Try to find a decomposition of this rank
        # We'll use a randomized approach with local search
        
        found = _find_decomposition(n, m, p, target_rank, deadline - 10)
        if found is not None:
            U, V, W = found
            best_rank = target_rank
            best_result = {"U": U, "V": V, "W": W}
            break
    
    if best_result is not None:
        return best_result
    else:
        return {"U": U_school, "V": V_school, "W": W_school}

def _find_decomposition(n, m, p, rank, time_limit):
    """Try to find a rank decomposition using structured search."""
    deadline = time.time() + time_limit
    
    # Initialize with random sparse matrices
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # We need to satisfy: for all i,j,k: sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    
    # This is a 3-way tensor completion problem
    
    # Use a greedy approach: assign values to satisfy constraints one by one
    
    # Create the list of constraints
    constraints = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                constraints.append((i, j, k))
    
    # For each constraint, we need sum_t U[i*m+j,t]*V[j*p+k,t]*W[k*n+i,t] = 1
    
    # Let's try a different approach: use the fact that we can have half-integer solutions
    
    # Actually, let's use a known construction: for 4x4*4x5, we can achieve rank 72
    # using block-Winograd
    
    # Construct a rank-72 solution:
    # Split into 2x2 blocks
    
    # For now, return None to use schoolbook
    return None

# EVOLVE-BLOCK-END
