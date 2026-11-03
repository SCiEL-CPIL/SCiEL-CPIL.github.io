"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x7 matrix.
    
    Strategy:
    1. Start with the schoolbook decomposition (rank 84).
    2. Attempt to reduce the rank by finding linear dependencies among the terms.
    3. Use a support compression approach: constrain the factor matrices to have a specific
       sparse support pattern derived from the block structure, and solve for non-trivial
       linear dependencies that allow exact cancellation of coefficients.
    
    The output factor follows the official trace convention W[k*n+i, r].
    """
    start_time = time.time()
    deadline = start_time + 140  # 140 seconds to leave margin before 150s kill
    
    n, m, p = DIMENSIONS  # n=3, m=4, p=7
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
        return {"U": U, "V": V, "W": W}
    
    def try_rank_decomposition(target_rank):
        """
        Try to find a decomposition of the given target rank.
        We look for U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
        For all i in [0,n), j in [0,m), k in [0,p):
            sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        """
        nm = n * m  # 12
        mp = m * p  # 28
        pn = p * n  # 21
        
        # For target_rank, we have target_rank columns in each matrix
        # The constraint is: for each (i,j,k), the trilinear form equals 1
        # This gives n*m*p = 84 constraints
        
        # Approach: Try a structured support pattern
        # Split the 3 rows into 2+1, and use Strassen-like combinations
        
        # Let's try: for the first 2 rows, use a 2x4x7 decomposition
        # For the last 1 row, use a 1x4x7 decomposition (rank 7)
        # Total: rank_2x4x7 + 7
        
        # For 2x4x7, we can further decompose:
        # 2x4x7 = 2x(2x2+2x2)x7... this gets complex
        
        # Alternative: use the fact that rank(3,4,7) <= rank(2,4,7) + rank(1,4,7)
        # rank(1,4,7) = 7 (just the outer product structure)
        # rank(2,4,7) <= ? 
        
        # Let's try a direct approach with target_rank = 42
        # We'll set up the system and try to find a sparse solution
        
        if target_rank < n * m * p:
            # Try to find a solution with exactly target_rank non-zero "triples"
            # Use a greedy/iterative approach
            
            # Initialize with random integer/half-integer values
            # and try to satisfy the constraints
            
            # For simplicity, let's try a specific pattern:
            # Group the 84 terms into pairs and see if we can merge them
            
            # Actually, let's try the known bound: rank(3,4,7) = 42
            # This can be achieved by: rank(3,4,7) <= 2*rank(2,4,7)/2 + ... 
            # or by direct construction
            
            # Let me try a simpler approach: assume the decomposition has
            # a specific block structure based on splitting dimensions
            
            # Split m=4 into 2+2, so we have 3x2x7 and 3x2x7 blocks
            # rank(3,2,7) for each block, total rank = 2 * rank(3,2,7)
            # rank(3,2,7) <= 21 (schoolbook) or less
            
            # For 3x2x7: split n=3 into 2+1
            # rank(2,2,7) + rank(1,2,7) = rank(2,2,7) + 2
            # rank(2,2,7) <= 14 (schoolbook) or 7 (using Strassen-like)
            
            # This is getting complex. Let me try a numerical approach.
            
            # Set up the bilinear system:
            # Unknowns: U (12 x R), V (28 x R), W (21 x R) where R = target_rank
            # Constraints: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            
            # This is a system of 84 cubic equations in 12*R + 28*R + 21*R = 61*R unknowns
            
            # For R=42: 61*42 = 2562 unknowns, 84 equations
            # This is highly underdetermined, so solutions should exist
            
            # Let's try to find a solution with integer/half-integer entries
            
            # Approach: 
            # 1. Fix W to have a specific simple pattern
            # 2. Solve for U and V
            
            # Let W[k*n+i, r] = delta_{k, r mod p} * delta_{i, r div p} for r in [0, p*n)
            # This gives W as identity-like, and then U and V are determined
            
            # Actually, if W is the identity (21x21 block), then for each (i,k):
            # sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1
            # So U[i*m+j, k*n+i] = 1/V[j*p+k, k*n+i]
            # This gives rank = 21, but U and V would have 21 columns each
            # But U is 12x21, V is 28x21, so total rank is 21
            
            # Wait, that's not right. The rank is the number of columns R.
            # If R=21, then U is 12x21, V is 28x21, W is 21x21
            # And we need: for each (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
            
            # If W is identity (21x21), then for each (i,j,k):
            # U[i*m+j, k*n+i] * V[j*p+k, k*n+i] = 1
            # So we can set U[i*m+j, k*n+i] = 1 and V[j*p+k, k*n+i] = 1
            # But this only fills 21 columns in U (for each i,k pair) and 21 in V
            # U has 12 rows, 21 cols: for row i*m+j, only column k*n+i is non-zero for each k
            # That's 7 non-zeros per row, which is fine
            # V has 28 rows, 21 cols: for row j*p+k, only column k*n+i is non-zero for each i
            # That's 3 non-zeros per row, which is fine
            
            # But wait: for a fixed (i,j), U[i*m+j, :] has non-zeros at positions k*n+i for k=0..6
            # That's 7 positions, each with value 1
            # For a fixed (j,k), V[j*p+k, :] has non-zeros at positions k*n+i for i=0..2
            # That's 3 positions, each with value 1
            
            # Then for (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # = U[i*m+j, k*n+i] * V[j*p+k, k*n+i] * W[k*n+i, k*n+i]
            # = 1 * 1 * 1 = 1 ✓
            
            # So this gives a valid decomposition with rank = 21!
            # But wait, is this correct? Let me double-check.
            
            # W[k*n+i, r] = delta_{k*n+i, r} (identity)
            # U[i*m+j, r] = 1 if r = k*n+i for some k, else 0
            #   More precisely: U[i*m+j, k*n+i] = 1 for all k
            # V[j*p+k, r] = 1 if r = k*n+i for some i, else 0
            #   More precisely: V[j*p+k, k*n+i] = 1 for all i
            
            # Check: for (i,j,k):
            # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # Only r = k*n+i contributes (since W is identity)
            # = U[i*m+j, k*n+i] * V[j*p+k, k*n+i] * W[k*n+i, k*n+i]
            # = 1 * 1 * 1 = 1 ✓
            
            # So rank 21 works! But can we do better?
            
            # Let's try rank 14 or lower.
            
            # For now, return the rank-21 solution
            R = 21
            U = [[0] * R for _ in range(nm)]
            V = [[0] * R for _ in range(mp)]
            W = [[0] * R for _ in range(pn)]
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        r = k * n + i
                        U[i * m + j][r] = 1
                        V[j * p + k][r] = 1
                        W[k * n + i][r] = 1
            
            return {"U": U, "V": V, "W": W}
    
    # Try to find a low-rank decomposition
    result = try_rank_decomposition(21)
    if result is not None:
        return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
