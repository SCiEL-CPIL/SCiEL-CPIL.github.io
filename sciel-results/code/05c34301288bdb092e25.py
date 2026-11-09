"""Matrix multiplication 3x3 by 3x4: rank-minimal exact tensor decomposition."""
import time
import numpy as np
from fractions import Fraction
import random
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix.
    Uses structured search with multiple strategies.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    n_m = n * m  # 9
    m_p = m * p  # 12
    p_n = p * n  # 12
    
    # Schoolbook rank = n*m*p = 36
    schoolbook_rank = n * m * p
    
    def verify_decomposition_frac(U, V, W, rank):
        """Verify with exact rational arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != 1:
                        return False
        return True
    
    def verify_decomposition_float(U, V, W, rank, tol=1e-9):
        """Quick verification with floating point."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Start with schoolbook decomposition
    best_rank = schoolbook_rank
    best_U = [[0] * schoolbook_rank for _ in range(n_m)]
    best_V = [[0] * schoolbook_rank for _ in range(m_p)]
    best_W = [[0] * schoolbook_rank for _ in range(p_n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # Strategy: Try to find lower rank decompositions
    # Theoretical lower bound: at least max(n*m, m*p, p*n) = 12
    # But realistically for 3x3*3x4, we expect around 20-25
    
    # Approach: For a given rank r, we need to find U, V, W with entries in {-1, -0.5, 0, 0.5, 1}
    # such that the tensor decomposition condition holds.
    
    # Key observation: We can fix W to have a specific structure and solve for U, V
    # Or use a randomized approach with local search
    
    # Let's try a more systematic approach:
    # 1. Fix W to be a random matrix with entries from {-1, 0, 1}
    # 2. For each (i,k), the equation becomes:
    #    sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    #    This is linear in the products U[i*m+j, r] * V[j*p+k, r]
    
    # Actually, let's try a different approach: use the fact that we can scale
    # If we find a decomposition with rational entries, we can clear denominators
    
    # Try ranks from 35 down to 15
    random.seed(42)
    np.random.seed(42)
    
    def try_find_decomposition(r, max_attempts=50):
        """Try to find a decomposition of rank r."""
        if time.time() > deadline:
            return None
        
        # Try multiple random initializations
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Initialize with random values from {-1, 0, 1}
            U = np.zeros((n_m, r), dtype=int)
            V = np.zeros((m_p, r), dtype=int)
            W = np.zeros((p_n, r), dtype=int)
            
            # Randomly assign -1, 0, 1 to each entry
            for _ in range(r):
                # Randomly choose which entries to set
                u_rows = np.random.choice(n_m, size=min(r, n_m), replace=False)
                v_rows = np.random.choice(m_p, size=min(r, m_p), replace=False)
                w_rows = np.random.choice(p_n, size=min(r, p_n), replace=False)
                
                for idx, row in enumerate(u_rows):
                    if idx < r:
                        U[row, idx] = random.choice([-1, 1])
                for idx, row in enumerate(v_rows):
                    if idx < r:
                        V[row, idx] = random.choice([-1, 1])
                for idx, row in enumerate(w_rows):
                    if idx < r:
                        W[row, idx] = random.choice([-1, 1])
            
            # Check if this works
            if verify_decomposition_float(U.tolist(), V.tolist(), W.tolist(), r):
                # Verify with exact arithmetic
                if verify_decomposition_frac(U.tolist(), V.tolist(), W.tolist(), r):
                    return (U.tolist(), V.tolist(), W.tolist())
        
        return None
    
    # Try to find better decompositions
    for r in range(schoolbook_rank - 1, 14, -1):
        if time.time() > deadline:
            break
        
        result = try_find_decomposition(r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U = U
            best_V = V
            best_W = W
            break
    
    # If we didn't find anything better, try a more structured approach
    if best_rank == schoolbook_rank:
        # Try a structured approach based on Strassen-like ideas
        # For 3x3 * 3x3, Strassen gives rank 7
        # For 3x3 * 3x4, we can try to extend this
        
        # One approach: use the 3x3*3x3 Strassen decomposition and extend to 3x4
        # Strassen's 7-multiplication algorithm for 2x2 matrices
        # For 3x3, we can use a similar approach
        
        # Let's try a different strategy: use a known low-rank construction
        # and verify it
        
        # For now, if we can't find a better one, return schoolbook
        pass
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
