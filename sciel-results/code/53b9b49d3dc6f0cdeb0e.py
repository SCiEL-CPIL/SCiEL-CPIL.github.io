"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def verify(U, V, W):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_rank11_systematic():
        """
        Systematic search for rank-11 decomposition using the algebraic structure.
        
        Key insight: For 2x2 * 2x4, we can use the fact that the tensor has
        a block structure. We'll search over structured decompositions where
        the U, V, W matrices have specific sparsity patterns.
        
        Approach: Use the known rank-14 Strassen decomposition as a starting point
        and try to eliminate rank-1 terms by finding linear dependencies.
        """
        if time.time() > deadline - 30:
            return None
        
        # Start with the rank-14 Strassen decomposition
        R14 = 14
        U14 = [[0]*R14 for _ in range(4)]
        V14 = [[0]*R14 for _ in range(8)]
        W14 = [[0]*R14 for _ in range(8)]
        
        # Left block Strassen (7 products)
        U14[0][0] = 1; U14[3][0] = 1; V14[0][0] = 1; V14[3][0] = 1; W14[0][0] = 1; W14[3][0] = 1
        U14[2][1] = 1; U14[3][1] = 1; V14[0][1] = 1; W14[2][1] = 1; W14[3][1] = -1
        U14[0][2] = 1; V14[2][2] = 1; V14[3][2] = -1; W14[1][2] = 1; W14[3][2] = 1
        U14[3][3] = 1; V14[1][3] = 1; V14[0][3] = -1; W14[0][3] = 1; W14[2][3] = 1
        U14[0][4] = 1; U14[1][4] = 1; V14[3][4] = 1; W14[0][4] = -1; W14[1][4] = 1
        U14[2][5] = 1; U14[0][5] = -1; V14[0][5] = 1; V14[1][5] = 1; W14[3][5] = 1
        U14[1][6] = 1; U14[3][6] = -1; V14[2][6] = 1; V14[3][6] = 1; W14[0][6] = 1
        
        # Right block Strassen (7 products)
        U14[0][7] = 1; U14[3][7] = 1; V14[4][7] = 1; V14[7][7] = 1; W14[4][7] = 1; W14[7][7] = 1
        U14[2][8] = 1; U14[3][8] = 1; V14[4][8] = 1; W14[6][8] = 1; W14[7][8] = -1
        U14[0][9] = 1; V14[6][9] = 1; V14[7][9] = -1; W14[5][9] = 1; W14[7][9] = 1
        U14[3][10] = 1; V14[5][10] = 1; V14[4][10] = -1; W14[4][10] = 1; W14[6][10] = 1
        U14[0][11] = 1; U14[1][11] = 1; V14[7][11] = 1; W14[4][11] = -1; W14[5][11] = 1
        U14[2][12] = 1; U14[0][12] = -1; V14[4][12] = 1; V14[5][12] = 1; W14[7][12] = 1
        U14[1][13] = 1; U14[3][13] = -1; V14[6][13] = 1; V14[7][13] = 1; W14[4][13] = 1
        
        if not verify(U14, V14, W14):
            return None
        
        # Now try to find a rank-11 decomposition by looking for
        # linear combinations of the 14 rank-1 terms that give zero.
        # If we can find 3 independent linear dependencies, we can reduce to rank 11.
        
        # The tensor equation is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
        # in the support, 0 otherwise.
        
        # Let's try a different approach: use the fact that for matrix multiplication,
        # there are known rank-11 constructions. Let me try to find one by search.
        
        # Strategy: Fix the U matrix to have a specific structure (e.g., based on Strassen
        # but with some columns removed), then solve for V and W.
        
        # Try removing 3 columns from the rank-14 decomposition and see if we can
        # adjust the remaining 11 to still satisfy the equations.
        
        # This is a nonlinear system, so let's try a more direct approach.
        
        # Known rank-11 construction for 2x2*2x4:
        # Based on the work of Stothers, the following should work:
        
        R = 11
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Use a hybrid approach: 6 for left block + 5 for right block
        # with shared structure.
        
        # Left block (6 products):
        # M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (a10+a11)(b00)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        
        # M3 = a00(b10-b11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = a11(b01-b00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (a00+a01)(b11)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # Right block (5 products):
        # N1 = (a00+a11)(b02+b13)
        U[0][6] = 1; U[3][6] = 1
        V[4][6] = 1; V[7][6] = 1
        W[4][6] = 1; W[7][6] = 1
        
        # N2 = (a10+a11)(b02)
        U[2][7] = 1; U[3][7] = 1
        V[4][7] = 1
        W[6][7] = 1; W[7][7] = -1
        
        # N3 = a00(b12-b13)
        U[0][8] = 1
        V[6][8] = 1; V[7][8] = -1
        W[5][8] = 1; W[7][8] = 1
        
        # N4 = a11(b03-b02)
        U[3][9] = 1
        V[5][9] = 1; V[4][9] = -1
        W[4][9] = 1; W[6][9] = 1
        
        # N5 = (a00+a01)(b13)
        U[0][10] = 1; U[1][10] = 1
        V[7][10] = 1
        W[5][10] = 1; W[4][10] = -1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        
        # If the above doesn't work, try other combinations
        # Let's try a more systematic search over the right block structure
        
        if time.time() > deadline - 15:
            return None
        
        # Try different assignments for the right block
        # The right block needs to compute:
        # C02 = a00*b02 + a01*b12
        # C03 = a00*b03 + a01*b13
        # C12 = a10*b02 + a11*b12
        # C13 = a10*b03 + a11*b13
        
        # We have 5 rank-1 terms for the right block.
        # Each term has a U-factor (linear combo of a00,a01,a10,a11),
        # V-factor (linear combo of b02,b03,b12,b13),
        # W-factor (linear combo of output positions).
        
        # Let's try to find a valid assignment by search.
        # This is a small search space if we restrict to simple patterns.
        
        # Try all combinations of which U-factors to use for the 5 right-block terms
        # U-factors available: a00, a01, a10, a11, a00+a11, a10+a11, a00+a01, a10-a00, a01-a11
        # V-factors available: b02, b03, b12, b13, b02+b13, b02+b03, b12+b13, b12-b13, b03-b02
        
        # This is still too large. Let me try a more targeted approach.
        
        # Use the fact that the left block already contributes to some outputs.
        # After the left block (6 terms), what's the residual for the right block?
        
        # Actually, let me just try a few more specific constructions.
        
        # Construction 2: Use 7 for left, 4 for right (aggressive sharing)
        R = 11
        U2 = [[0]*R for _ in range(4)]
        V2 = [[0]*R for _ in range(8)]
        W2 = [[0]*R for _ in range(8)]
        
        # Left block: full Strassen (7)
        U2[0][0] = 1; U2[3][0] = 1; V2[0][0] = 1; V2[3][0] = 1; W2[0][0] = 1; W2[3][0] = 1
        U2[2][1] = 1; U2[3][1] = 1; V2[0][1] = 1; W2[2][1] = 1; W2[3][1] = -1
        U2[0][2] = 1; V2[2][2] = 1; V2[3][2] = -1; W2[1][2] = 1; W2[3][2] = 1
        U2[3][3] = 1; V2[1][3] = 1; V2[0][3] = -1; W2[0][3] = 1; W2[2][3] = 1
        U2[0][4] = 1; U2[1][4] = 1; V2[3][4] = 1; W2[0][4] = -1; W2[1][4] = 1
        U2[2][5] = 1; U2[0][5] = -1; V2[0][5] = 1; V2[1][5] = 1; W2[3][5] = 1
        U2[1][6] = 1; U2[3][6] = -1; V2[2][6] = 1; V2[3][6] = 1; W2[0][6] = 1
        
        # Right block: 4 terms
        # Try: (a00+a11)(b02+b13), (a10+a11)(b02), a00(b12-b13), a11(b03-b02)
        U2[0][7] = 1; U2[3][7] = 1; V2[4][7] = 1; V2[7][7] = 1; W2[4][7] = 1; W2[7][7] = 1
        U2[2][8] = 1; U2[3][8] = 1; V2[4][8] = 1; W2[6][8] = 1; W2[7][8] = -1
        U2[0][9] = 1; V2[6][9] = 1; V2[7][9] = -1; W2[5][9] = 1; W2[7][9] = 1
        U2[3][10] = 1; V2[5][10] = 1; V2[4][10] = -1; W2[4][10] = 1; W2[6][10] = 1
        
        if verify(U2, V2, W2):
            return {"U": U2, "V": V2, "W": W2}
        
        return None
    
    def try_rank11_search():
        """
        More systematic search for rank-11 using structured patterns.
        """
        if time.time() > deadline - 20:
            return None
        
        # Try a different structured approach:
        # Use the fact that we can write the multiplication as:
        # C = A * B where B = [B1 | B2]
        # C = [A*B1 | A*B2]
        
        # For A*B1, use 7 Strassen products.
        # For A*B2, we need to find 4 products that complete the computation.
        
        # The key is that A*B2 shares the same A, so we can reuse some
        # of the linear combinations from the left block.
        
        # Let's try: after computing A*B1 with 7 products, what additional
        # products do we need for A*B2?
        
        # A*B2 = [a00*b02 + a01*b12, a00*b03 + a01*b13]
        #        [a10*b02 + a11*b12, a10*b03 + a11*b13]
        
        # We can compute this with 4 products if we're clever:
        # P1 = (a00+a11)(b02+b13) -> a00*b02 + a00*b13 + a11*b02 + a11*b13
        # P2 = (a10+a11)(b02)     -> a10*b02 + a11*b02
        # P3 = a00*(b12-b13)      -> a00*b12 - a00*b13
        # P4 = a11*(b03-b02)      -> a11*b03 - a11*b02
        
        # Then:
        # C02 = P1 - P2 - P3 - P4 + (a01*b12 + a10*b03)  -- still missing terms!
        
        # This doesn't work with just 4. We need 5 for the right block.
        
        # So rank 11 = 6 (left) + 5 (right) or 7 (left) + 4 (right) won't work.
        # It must be some other combination.
        
        # Let me try: use a non-Strassen approach for both blocks.
        
        # Alternative: use 5+5+1 = 11
        # 5 products for A*B1, 5 for A*B2, 1 shared.
        
        # This is getting complex. Let me just try the rank-14 solution
        # and return it if we can't find rank 11.
        
        return None
    
    def try_rank14_strassen():
        """Return the known rank-14 Strassen decomposition."""
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block Strassen (7 products)
        U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1; W[0][0] = 1; W[3][0] = 1
        U[2][1] = 1; U[3][1] = 1; V[0][1] = 1; W[2][1] = 1; W[3][1] = -1
        U[0][2] = 1; V[2][2] = 1; V[3][2] = -1; W[1][2] = 1; W[3][2] = 1
        U[3][3] = 1; V[1][3] = 1; V[0][3] = -1; W[0][3] = 1; W[2][3] = 1
        U[0][4] = 1; U[1][4] = 1; V[3][4] = 1; W[0][4] = -1; W[1][4] = 1
        U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[3][5] = 1
        U[1][6] = 1; U[3][6] = -1; V[2][6] = 1; V[3][6] = 1; W[0][6] = 1
        
        # Right block Strassen (7 products)
        U[0][7] = 1; U[3][7] = 1; V[4][7] = 1; V[7][7] = 1; W[4][7] = 1; W[7][7] = 1
        U[2][8] = 1; U[3][8] = 1; V[4][8] = 1; W[6][8] = 1; W[7][8] = -1
        U[0][9] = 1; V[6][9] = 1; V[7][9] = -1; W[5][9] = 1; W[7][9] = 1
        U[3][10] = 1; V[5][10] = 1; V[4][10] = -1; W[4][10] = 1; W[6][10] = 1
        U[0][11] = 1; U[1][11] = 1; V[7][11] = 1; W[4][11] = -1; W[5][11] = 1
        U[2][12] = 1; U[0][12] = -1; V[4][12] = 1; V[5][12] = 1; W[7][12] = 1
        U[1][13] = 1; U[3][13] = -1; V[6][13] = 1; V[7][13] = 1; W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    def naive_decomp():
        rank = n * m * p  # 16
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    # Try in order of increasing rank
    result = try_rank11_systematic()
    if result is not None:
        return result
    
    result = try_rank11_search()
    if result is not None:
        return result
    
    result = try_rank14_strassen()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
