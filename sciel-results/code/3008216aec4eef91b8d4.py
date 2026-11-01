"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition exactly represents matrix multiplication."""
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 for all i,j,k.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _to_fraction(x):
    """Convert a number to Fraction if possible."""
    if isinstance(x, Fraction):
        return x
    try:
        return Fraction(x)
    except:
        return Fraction(0)

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 2*2*5 = 20
    
    # For 2x2 * 2x5 matrix multiplication:
    # The tensor has dimensions 2x2x5 = 20 entries
    # We want to find the minimum rank decomposition
    
    # Known results: For 2x2 * 2x2, the rank is 7 (Strassen)
    # For 2x2 * 2x5, we can use a combination of Strassen-like approaches
    
    # Let's try to find a decomposition of rank < 20
    
    # Strategy: Use the fact that we're multiplying (2x2) by (2x5)
    # Let A be 2x2, B be 2x5, C = A*B is 2x5
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # The tensor T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,1,2,3,4}
    
    # We can use a block approach: split the 2x5 into two 2x2 and one 2x1
    # Or use Strassen for the 2x2 part and handle the rest
    
    # Let's try rank 15, 14, 13, ... down to some lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(4, 10, 10) = 10
    
    best_rank = schoolbook_rank
    best_decomp = None
    
    # Generate schoolbook decomposition as fallback
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, r
    
    U_sb, V_sb, W_sb, r_sb = schoolbook()
    best_decomp = (U_sb, V_sb, W_sb, r_sb)
    
    # Try to find better decompositions using randomized/structured approaches
    # For this specific case, let's try a few known constructions
    
    # Approach: Use Strassen-like ideas
    # For 2x2 * 2x2, Strassen uses rank 7
    # For the extra columns (2x1 part), we need additional terms
    
    # Let's try to construct a rank-15 decomposition:
    # Use 7 terms for the first 2 columns of B (treating it as 2x2),
    # and handle the remaining 3 columns
    
    # Actually, let's try a more systematic approach with small integer/half-integer entries
    
    # Try ranks from 12 to 19
    for target_rank in range(12, schoolbook_rank):
        if time.time() > deadline:
            break
        
        # Try random constructions with small integer/half-integer entries
        found = False
        max_attempts = 5000
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            # Generate random U, V, W with entries from {-1, 0, 1, 0.5, -0.5}
            values = [-1, -0.5, 0, 0.5, 1]
            
            U = [[random.choice(values) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.choice(values) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.choice(values) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify(U, V, W, n, m, p, target_rank):
                found = True
                best_decomp = (U, V, W, target_rank)
                break
        
        if found:
            break
    
    U, V, W, rank = best_decomp
    
    # Convert to proper format
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
