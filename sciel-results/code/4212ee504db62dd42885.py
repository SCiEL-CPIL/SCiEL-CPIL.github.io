"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all (i,j,k) in [0,n) x [0,m) x [0,p)
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # This is a 3-way tensor of size n*m x m*p x p*n with 1s at specific positions.
    
    # Schoolbook rank is n*m*p = 36.
    # For (2,3,6), we try to find lower rank.
    
    # Let's use a randomized search with exact rational arithmetic.
    # We'll try to find a rank-r decomposition by solving linear systems.
    
    # First, let's try known constructions and then random search.
    
    def verify(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(r):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if s != 1:
                        return False
        return True
    
    def schoolbook_decomp(n, m, p):
        """Return the schoolbook decomposition of rank n*m*p."""
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
        return U, V, W
    
    # Start with schoolbook as fallback
    U_fallback, V_fallback, W_fallback = schoolbook_decomp(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_fallback, V_fallback, W_fallback
    
    # Try to find lower rank decompositions.
    # For (2,3,6), the border rank is at most 20 (from literature).
    # Let's try ranks from 20 up to 35.
    
    for target_rank in range(20, min(36, n*m*p)):
        if time.time() > deadline - 10:
            break
        
        # Try random search for this rank
        found = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random sparse initialization
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # For each column, pick random sparse patterns
            for c in range(target_rank):
                # Pick random non-zero entries
                u_idx = random.randint(0, n*m-1)
                v_idx = random.randint(0, m*p-1)
                w_idx = random.randint(0, p*n-1)
                
                U[u_idx][c] = 1
                V[v_idx][c] = 1
                W[w_idx][c] = 1
            
            # Verify
            if verify(U, V, W, n, m, p):
                found = True
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if found:
            break
    
    # If we found a better decomposition, use it; otherwise use schoolbook
    if best_rank < n * m * p:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        return {"U": U_fallback, "V": V_fallback, "W": W_fallback}
# EVOLVE-BLOCK-END
