import time
import sys
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    global START_TIME, TIME_LIMIT
    
    # Trivial rank decomposition (baseline)
    rank_trivial = n * m * p
    U_trivial = [[0] * rank_trivial for _ in range(n * m)]
    V_trivial = [[0] * rank_trivial for _ in range(m * p)]
    W_trivial = [[0] * rank_trivial for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j][col] = 1
                V_trivial[j * p + k][col] = 1
                W_trivial[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_trivial
    best_result = {"U": U_trivial, "V": V_trivial, "W": W_trivial}
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(rank):
                        val += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if val != 1:
                        return False
        return True

    def try_random_search(r, n, m, p, max_iterations=1000000, coeff_set=None):
        """Try to find a rank-r decomposition with random coefficients."""
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        if coeff_set is None:
            coeff_set = [-1, 0, 1]
        
        for _ in range(max_iterations):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                return None
            
            U = [[random.choice(coeff_set) for _ in range(r)] for _ in range(num_u)]
            V = [[random.choice(coeff_set) for _ in range(r)] for _ in range(num_v)]
            W = [[random.choice(coeff_set) for _ in range(r)] for _ in range(num_w)]
            
            if verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        return None

    def try_structured_search(r, n, m, p):
        """
        Try a structured approach based on the fact that for matrix multiplication,
        we can exploit the structure. For 3x4x4, try to find decompositions where
        the tensors have a specific pattern.
        
        Key insight: The tensor T[i,j,k] = 1 for all valid (i,j,k).
        We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1
        
        One approach: Try to use the "rank-1" building blocks more cleverly.
        For example, we can try to combine multiple rank-1 terms.
        """
        num_u = n * m
        num_v = m * p
        num_w = p * n
        
        # Try a hybrid approach: fix some structure and search for the rest
        # For instance, try to make U, V, W have many zeros and specific patterns
        
        # Strategy 1: Try rank-r where r is close to known bounds
        # For 3x4x4, the trivial rank is 48. Known lower bounds suggest it could be lower.
        # Let's try a more systematic search with different coefficient sets.
        
        # Try with coefficients in {-1, 0, 1, 2}
        result = try_random_search(r, n, m, p, max_iterations=20000, coeff_set=[-1, 0, 1, 2])
        if result is not None:
            return result
        
        # Try with coefficients in {0, 1} only (binary)
        result = try_random_search(r, n, m, p, max_iterations=20000, coeff_set=[0, 1])
        if result is not None:
            return result
        
        return None

    # Search for lower rank decompositions
    # Start from a reasonable lower bound and go up
    # The trivial rank is 48. Let's try to find something better.
    
    # First, try some structured approaches for lower ranks
    # Based on known results, the rank of 3x4x4 matrix multiplication is likely around 20-30
    
    # Try ranks from 20 to 47 (just below trivial)
    for r in range(20, min(rank_trivial, 48)):
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        
        # Try structured search first
        result = try_structured_search(r, n, m, p)
        if result is not None:
            best_rank = r
            best_result = result
            break
        
        # If structured search fails, try pure random search with more iterations
        if time.time() - START_TIME > TIME_LIMIT - 15:
            break
        
        result = try_random_search(r, n, m, p, max_iterations=50000)
        if result is not None:
            best_rank = r
            best_result = result
            break
    
    # If no improvement found with the above, try a few more ranks with different strategies
    if best_rank == rank_trivial:
        for r in [15, 18, 25, 30, 35, 40, 45]:
            if time.time() - START_TIME > TIME_LIMIT - 10:
                break
            if r >= rank_trivial:
                continue
            
            result = try_random_search(r, n, m, p, max_iterations=30000)
            if result is not None:
                best_rank = r
                best_result = result
                break
    
    return best_result
# EVOLVE-BLOCK-END
