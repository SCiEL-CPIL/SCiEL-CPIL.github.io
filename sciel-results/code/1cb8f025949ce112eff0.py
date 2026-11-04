"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Attempt to find a rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication.
    The schoolbook rank is 3*4*7 = 84. We attempt to find a lower rank via random search
    combined with local optimization, falling back to schoolbook if no improvement is found
    within the time budget.
    """
    n, m, p = DIMENSIONS
    total_dim = n * m * p  # 84
    
    # Time budget guard
    deadline = time.time() + 120.0
    
    # First, construct the schoolbook decomposition as a fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # The multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # We need to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Try random initializations with rank < 84
    # Start with ranks from 60 down to 40
    for target_rank in range(83, 39, -1):
        if time.time() > deadline - 5:
            break
        
        # Try a few random initializations
        max_attempts = 50
        attempts = 0
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Random integer initialization
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Check if this is a valid decomposition
            # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
            valid = True
            for i in range(n):
                if not valid:
                    break
                for j in range(m):
                    if not valid:
                        break
                    for k in range(p):
                        s = 0
                        for c in range(target_rank):
                            s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                        if s != 1:
                            valid = False
                            break
            
            if valid:
                best_rank = target_rank
                best_U = U
                best_V = V
                best_W = W
                # Found a valid decomposition, try to go lower
                break
        
        # If we found a valid decomposition at this rank, continue to lower ranks
        # If not, we can stop if we've spent too much time
        if time.time() > deadline - 10:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
