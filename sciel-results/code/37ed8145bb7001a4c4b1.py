"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def verify(U, V, W, rank):
        """Verify that the decomposition is exact."""
        if time.time() > deadline:
            return False
        # For each output element (k, i), check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] 
        # equals the indicator that this is a valid matrix multiplication term
        # Actually, we need to verify that the tensor decomposition is correct
        # The matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
        # and the decomposition should satisfy:
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    def try_reduce_rank(U, V, W, rank, max_attempts=1000):
        """Try to reduce rank by finding linear dependencies."""
        if time.time() > deadline:
            return U, V, W, rank
        
        best_U, best_V, best_W, best_rank = U, V, W, rank
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Try to find if we can eliminate one column by expressing it as a combination
            # of others. For each column c, check if there exist coefficients for other
            # columns such that the tensor contribution matches.
            
            # A simpler approach: try random linear combinations to see if we can
            # find a lower rank decomposition
            
            # Try removing a random column and checking if the remaining columns
            # can still span the same tensor (approximately)
            c_remove = np.random.randint(rank)
            
            # Create reduced matrices without column c_remove
            U_red = [row[:c_remove] + row[c_remove+1:] for row in U]
            V_red = [row[:c_remove] + row[c_remove+1:] for row in V]
            W_red = [row[:c_remove] + row[c_remove+1:] for row in W]
            new_rank = rank - 1
            
            # Check if the reduced decomposition still works
            if verify(U_red, V_red, W_red, new_rank):
                U, V, W, rank = U_red, V_red, W_red, new_rank
                best_U, best_V, best_W, best_rank = U, V, W, rank
        
        return best_U, best_V, best_W, best_rank
    
    # Start with schoolbook decomposition
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
    
    # Verify initial decomposition
    if not verify(U, V, W, rank):
        # Should not happen, but return anyway
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    U, V, W, rank = try_reduce_rank(U, V, W, rank, max_attempts=5000)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
