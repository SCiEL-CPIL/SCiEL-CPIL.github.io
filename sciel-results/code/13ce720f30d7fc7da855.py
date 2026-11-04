"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]
import numpy as np
import time
import itertools
import random

# EVOLVE-BLOCK-START
def construct():
    """
    Attempt to find a low-rank decomposition for 2x5 x 5x5 matrix multiplication.
    
    The tensor for (n,m,p) matrix mult has indices (i,j,k) for i in [n], j in [m], k in [p].
    U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank.
    The decomposition satisfies: T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Try Laderman's 3x3 algorithm (rank 7) as a building block
    # Laderman's algorithm for 3x3 matrix multiplication:
    # T1 = A11*B11 + A21*B12 + A12*B21 + A22*B22  (not quite - let me recall correctly)
    
    # Actually, Laderman's 7-term algorithm for 3x3:
    # Let's use the standard formulation:
    # C = A*B where A, B are 3x3
    # 
    # Define:
    # M1 = A11*B11
    # M2 = (A11 + A21)*B12  
    # M3 = A21*(B12 + B22)
    # M4 = A11*(B11 - B22)
    # M5 = (A12 + A22)*B21
    # M6 = A22*(B21 - B11)
    # M7 = (A11 + A22)*(B11 + B22)
    # 
    # Then:
    # C11 = M1 + M4 + M5 + M6 - M7  ... this is getting complicated.
    # Let me use a cleaner formulation.
    
    # Known rank for 3x3: Laderman's algorithm has rank 7.
    # For 2x5, we can embed 3x3 in a 5x5 block and use the remaining dimensions.
    
    # Strategy: 
    # 1. Start with schoolbook (rank 50)
    # 2. Try to find rank-15 or rank-12 decompositions by leveraging structure
    # 3. Use algebraic methods: look for decompositions with integer/half-integer entries
    
    # The matrix multiplication tensor for (2,5,5) has rank at most 50 (schoolbook).
    # Known lower bounds: rank >= 2*5*5 / (some factor). 
    # For (2,5,5), the border rank and rank are studied. 
    # 
    # Let's try a direct approach: assume a decomposition with rank R and try R=15, 16, ...
    # Use the fact that we need integer/half-integer entries.
    
    # First, let's verify the schoolbook decomposition and return it as fallback.
    rank_school = n * m * p  # 50
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def _verify(U, V, W):
        """Verify that U, V, W decompose the matrix multiplication tensor."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # T[i*m+j, j*p+k, k*n+i] should be 1
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
                    # Also check that other entries are 0
                    # T[i', j', k'] for (i',j',k') != (i,j,k) should be 0
                    # But we only need to check the "1" positions for correctness
        # Check a few zero positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if (i, j, k) != (i2, j2, k2):
                                    s = 0
                                    for r in range(rank):
                                        s += U[i2*m+j2][r] * V[j2*p+k2][r] * W[k2*n+i2][r]
                                    if abs(s) > 1e-9:
                                        return False
        return True
    
    # Try to find a lower rank decomposition
    # Start with small rank candidates
    for R in [15, 16, 17, 18, 20]:
        if time.time() > deadline:
            break
        
        # For each rank R, we need to find U (10 x R), V (25 x R), W (10 x R)
        # such that the tensor is decomposed.
        # 
        # Use the structure: the tensor is the matrix multiplication tensor, which is
        # a sum of rank-1 tensors. Each rank-1 term is u * v * w where u is a vector in R^10,
        # v in R^25, w in R^10.
        #
        # Key insight: For matrix multiplication, the tensor T satisfies:
        # T[i,j,k] = 1 if the (i,j) of A and (j,k) of B produce (i,k) of C
        #
        # Let's try a structured approach: assume the decomposition uses terms that
        # are "aligned" with the matrix multiplication structure.
        
        # Try a specific family: decompositions based on 3x3 blocks
        # For 2x5x5, we can think of A as 2x5, B as 5x5, C as 2x5.
        # 
        # One approach: use the fact that 5 = 2 + 3, and use a 2x2 x 2x2 block (rank 7 by Strassen)
        # and a 3x3 x 3x3 block (rank 7 by Laderman), but the coupling makes this complex.
        
        # Alternative: try a greedy / iterative approach
        # Start with a random decomposition and try to reduce rank by merging terms
        
        # For now, let's try a structured low-rank decomposition based on
        # the observation that the matrix multiplication tensor for (2,5,5)
        # might have rank 15 or 16.
        
        # Try a specific construction:
        # Use the 5x5 matrix multiplication rank 19 (or 20) and adapt to 2x5.
        # 
        # Actually, let's just return the schoolbook decomposition for now,
        # but try a quick search for a lower rank one.
        
        if R <= 20:
            # Try to construct a rank-R decomposition using a known algorithm
            # For (2,5,5), one known result is that the rank is at most 25 (trivially)
            # and possibly lower.
            #
            # Let's try a decomposition based on 2x2 Strassen blocks:
            # A = [A1 A2] where A1 is 2x2, A2 is 2x3
            # B = [B1; B2] where B1 is 2x5, B2 is 3x5
            # C = A1*B1 + A2*B2
            #
            # This doesn't directly help with rank reduction.
            
            pass
    
    # Return the schoolbook decomposition as the guaranteed valid result
    return {"U": U_school, "V": V_school, "W": W_school}
# EVOLVE-BLOCK-END
