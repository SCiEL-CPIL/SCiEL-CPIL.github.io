"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
from itertools import product

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Schoolbook decomposition: rank = n*m*p = 24
    def schoolbook():
        rank = n * m * p
        U = [[Fraction(0)] * rank for _ in range(n * m)]
        V = [[Fraction(0)] * rank for _ in range(m * p)]
        W = [[Fraction(0)] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = Fraction(1)
                    V[j * p + k][col] = Fraction(1)
                    W[k * n + i][col] = Fraction(1)
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition is correct using exact arithmetic."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if s != Fraction(1):
                        return False
        return True
    
    # Try to find a lower rank decomposition using random search
    # Strategy: randomly initialize U, V, W with small rational values and try to solve
    # for a valid decomposition
    
    # For 2x3 * 3x4, known results suggest rank 22 or 23 might be achievable
    # Let's try a more systematic approach using the structure of the problem
    
    # The tensor T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Try rank 23 first, then 22
    for target_rank in [23, 22]:
        if time.time() > deadline - 5:
            break
        
        # Use a structured initialization based on the schoolbook decomposition
        # with some columns merged or modified
        
        # Approach: Start with schoolbook and try to eliminate one column
        # by finding linear dependencies
        
        # For rank 23: try to find if one schoolbook term can be expressed
        # using the others
        
        # A simpler approach: random search with rational entries
        max_attempts = 500
        attempts = 0
        
        while attempts < max_attempts and time.time() < deadline - 5:
            attempts += 1
            
            # Initialize with small random rational values
            # Use values from {-1, 0, 1} for simplicity
            U = [[Fraction(random.choice([-1, 0, 1])) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[Fraction(random.choice([-1, 0, 1])) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[Fraction(random.choice([-1, 0, 1])) for _ in range(target_rank)] for _ in range(p * n)]
            
            if verify_decomposition(U, V, W):
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < n * m * p:
            break
    
    # If we found a better rank, return it; otherwise return schoolbook
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
