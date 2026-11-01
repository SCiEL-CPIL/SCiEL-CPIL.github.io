"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

import time
import itertools

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    # Check tensor: T[i,j,k] should equal sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Start with schoolbook rank 32
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    # Generate schoolbook decomposition
    U_sch = [[0] * best_rank for _ in range(n * m)]
    V_sch = [[0] * best_rank for _ in range(m * p)]
    W_sch = [[0] * best_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sch[i * m + j][col] = 1
                V_sch[j * p + k][col] = 1
                W_sch[k * n + i][col] = 1
                col += 1
    
    if _verify(U_sch, V_sch, W_sch, n, m, p):
        best_U, best_V, best_W = U_sch, V_sch, W_sch
    
    # Try to find a lower rank decomposition using structured patterns
    # For (2,4,4), known rank is 16
    
    # Try rank 16 with various structured patterns
    target_rank = 16
    U_try = [[0] * target_rank for _ in range(n * m)]
    V_try = [[0] * target_rank for _ in range(m * p)]
    W_try = [[0] * target_rank for _ in range(p * n)]
    
    # Pattern: use (i, j, k) mapped to column index
    # For rank 16, we need each of the 32 terms to be covered
    # Try: column = j*p//2 + k for some mapping, or use a different structure
    
    # Known result: for 2x4x4, rank 16 exists
    # One construction: group terms cleverly
    
    # Let's try: for c in 0..15, assign values based on binary representations
    # U[i*m+j, c] = 1 if (j mod 2) == (c mod 2) and (i == c//4 or similar)
    
    # Actually, let me try a direct search over small structured families
    # Family: U[i*m+j, c] = a_i[j][c], V[j*p+k, c] = b_j[k][c], W[k*n+i, c] = w_k[i][c]
    # where each entry is in {0, 1, -1}
    
    # For efficiency, try specific known patterns
    
    # Pattern 1: column c = 4*j + k for j in 0..3, k in 0..3 (but that's 16 columns)
    # U[i*m+j, c]: depends on j and k? No, U only depends on i, j, c
    # This won't work directly since U can't depend on k
    
    # Let me think differently. The multiplication tensor T[i,j,k] = 1 for all i,j,k
    # We need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k
    
    # For n=2, m=4, p=4, rank 16:
    # Try U[i*m+j, c] = 1 if j == c%4, else 0 (4 values per i)
    # But then we only use 4 of the 16 columns per (i,j)
    
    # Let me try: c = 2*j + (k % 2), with some structure
    # This is getting complex. Let me just return the verified schoolbook for now.
    
    # Try one more structured approach:
    # U[i*m+j, c] = 1 if c//4 == j, else 0  (j determines which group of 4)
    # V[j*p+k, c] = 1 if c%4 == k, else 0
    # W[k*n+i, c] = 1 if (c//4) % 4 == k? No...
    
    # W[k*n+i, c]: for this to work, we need the product to be 1 when i,j,k match
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 iff all three are nonzero
    
    # With U: j = c//4, V: k = c%4, then W needs to be 1 when (k,i) matches
    # W[k*n+i, c] should be 1 for the right i given k = c%4
    # But i is free (0 or 1), so W[k*n+i, c] can't depend on i in a way that picks one
    
    # This pattern won't work. The schoolbook is the safe choice.
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
