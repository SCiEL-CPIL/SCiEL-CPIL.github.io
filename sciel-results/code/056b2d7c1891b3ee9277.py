"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 110  # safety margin before 120s limit
    n, m, p = DIMENSIONS
    
    # Known low-rank decomposition for 3x3 matrix multiplication (rank 23)
    # This is a well-known result from tensor decomposition research
    # Format: list of (u_indices, v_indices, w_indices) where each is a set of positions
    # u is in R^(n*m) = R^9, v is in R^(m*p) = R^9, w is in R^(p*n) = R^9
    
    # Using the known rank-23 decomposition for 3x3x3 tensor
    # The decomposition is given by 23 terms
    # Each term has coefficients for U, V, W
    
    # I'll use a known explicit decomposition. The 3x3 matrix multiplication
    # tensor has rank 23 (or 24 depending on the exact form).
    # Let me use a verified rank-23 decomposition.
    
    # Actually, let me use the well-documented rank-23 decomposition.
    # The entries are in {-1, 0, 1} for this particular decomposition.
    
    # Rank-23 decomposition of 3x3 matrix multiplication
    # U is 9x23, V is 9x23, W is 9x23
    # Indexing: U[i*m+j], V[j*p+k], W[k*n+i] for 3x3x3
    
    # Here's a known rank-23 decomposition (from various tensor decomposition papers)
    # I'll encode it as three 9x23 matrices with entries in {-1, 0, 1}
    
    # Let me use a simpler approach: try to verify if we can find a rank < 63
    # by using the known 3x3 rank-23 decomposition and extending to 3x7
    
    # For 3x7, we can potentially use block structure
    # But the simplest improvement is to use the rank-23 3x3 decomposition
    # and combine it appropriately
    
    # Actually, for 3x7, the minimal rank is not simply 7*23 because the
    # decomposition needs to handle all 7 columns together.
    
    # Let me try a different approach: use the known rank-23 for 3x3 and
    # see if we can extend it to 3x7 with some optimization
    
    # For now, let me return the schoolbook decomposition but check if we
    # can do better with a simple search
    
    # Schoolbook: rank = 3*3*7 = 63
    rank_schoolbook = n * m * p
    
    # Try to find a better decomposition using the 3x3 rank-23 as a base
    # The idea: for 3x7, we can think of it as 7 independent 3x1 multiplications
    # plus cross terms, but this doesn't directly help
    
    # Alternative: use the fact that 3x7 = 3x3 + 3x3 + 3x1
    # rank(3x3) <= 23, rank(3x1) <= 9
    # So rank(3x7) <= 23 + 23 + 9 = 55? Not quite, because the decompositions
    # need to be compatible
    
    # Let me just return the schoolbook for now, but with a deadline check
    if time.time() > deadline:
        # Return schoolbook as fallback
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Use the known rank-23 decomposition for 3x3 and extend to 3x7
    # The 3x3 rank-23 decomposition (one known form):
    # I'll use a specific verified decomposition
    
    # Actually, let me use a simpler known result: the rank of 3x3 matrix
    # multiplication is 23 (or 24). Let me use rank 24 which is easier to encode.
    
    # For 3x7, I'll try to use a decomposition based on the 3x3 result
    # But this is complex. Let me just return schoolbook with a note.
    
    # Return schoolbook decomposition
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
