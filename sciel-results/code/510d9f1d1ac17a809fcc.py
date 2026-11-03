"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x10 matrix.
    
    We attempt to find a lower rank decomposition than the schoolbook rank of 60.
    The condition is: for all i in [0,n), j in [0,m), k in [0,p):
    sum_{c=0}^{r-1} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    We'll try various ranks and use a heuristic search to find valid decompositions.
    """
    import numpy as np
    from fractions import Fraction
    import time
    
    start_time = time.time()
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 60
    
    # First, let's try to find a lower rank decomposition
    # We'll use a randomized approach with verification
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition satisfies the tensor equation."""
        U = np.array(U, dtype=object)
        V = np.array(V, dtype=object)
        W = np.array(W, dtype=object)
        
        # Check: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for c in range(rank):
                        total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if total != 1:
                        return False
        return True
    
    # Try to find a decomposition with rank < 60
    # We'll use a structured approach: try to exploit the small dimensions
    
    # For 2x3 * 3x10, let's try rank 59, 58, etc.
    # This is a hard problem, so we'll use a heuristic:
    # Start with schoolbook and try to merge columns
    
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Create schoolbook decomposition
    def make_schoolbook():
        U = [[0] * schoolbook_rank for _ in range(n * m)]
        V = [[0] * schoolbook_rank for _ in range(m * p)]
        W = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    U_school, V_school, W_school = make_schoolbook()
    
    # Try to reduce rank by finding linear dependencies
    # For each pair of columns, check if we can combine them
    # This is a heuristic - we'll try to find if any column is redundant
    
    # Given the complexity, let's try a different approach:
    # Use the fact that for small matrices, we can sometimes find
    # decompositions using Strassen-like methods or other algebraic tricks
    
    # For now, let's try a randomized search for lower rank
    # We'll try ranks from 59 down to 50
    
    import random
    
    for target_rank in range(schoolbook_rank - 1, max(40, schoolbook_rank - 10), -1):
        if time.time() - start_time > 100:  # Leave time for finalization
            break
        
        # Try multiple random attempts
        for attempt in range(5):
            if time.time() - start_time > 105:
                break
            
            # Initialize with random values
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p * n)]
            
            # Convert to Fraction for exact arithmetic
            U_frac = [[Fraction(U[i][j]) for j in range(target_rank)] for i in range(n * m)]
            V_frac = [[Fraction(V[i][j]) for j in range(target_rank)] for i in range(m * p)]
            W_frac = [[Fraction(W[i][j]) for j in range(target_rank)] for i in range(p * n)]
            
            if verify_decomposition(U_frac, V_frac, W_frac, target_rank):
                best_rank = target_rank
                best_decomp = (U_frac, V_frac, W_frac)
                break
        
        if best_decomp is not None:
            break
    
    # If we found a better decomposition, return it
    if best_decomp is not None:
        U_frac, V_frac, W_frac = best_decomp
        U_out = [[str(x) if isinstance(x, Fraction) else x for x in row] for row in U_frac]
        V_out = [[str(x) if isinstance(x, Fraction) else x for x in row] for row in V_frac]
        W_out = [[str(x) if isinstance(x, Fraction) else x for x in row] for row in W_frac]
        # Convert back to int if possible
        U_out = [[int(x) if isinstance(x, Fraction) and x.denominator == 1 else x for x in row] for row in U_frac]
        V_out = [[int(x) if isinstance(x, Fraction) and x.denominator == 1 else x for x in row] for row in V_frac]
        W_out = [[int(x) if isinstance(x, Fraction) and x.denominator == 1 else x for x in row] for row in W_frac]
        return {"U": U_out, "V": V_out, "W": W_out}
    
    # Otherwise, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
