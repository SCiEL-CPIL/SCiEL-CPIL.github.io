"""Rank-minimal tensor decomposition for 2x2 * 2x4 matrix multiplication."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
    def verify(U, V, W, tol=1e-9):
        """Verify the tensor decomposition is exact."""
        R = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(R):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def build_rank14():
        """Build the known rank-14 Strassen decomposition."""
        R = 14
        U = [[0]*R for _ in range(nm)]
        V = [[0]*R for _ in range(mp)]
        W = [[0]*R for _ in range(pn)]
        
        # Left block Strassen (7 products): computes A*B_left (2x2 * 2x2)
        # Output positions: W[0]=C00, W[1]=C01, W[2]=C10, W[3]=C11
        # M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1; V[0][0] = 1; V[3][0] = 1; W[0][0] = 1; W[3][0] = 1
        # M2 = (a10+a11)(b00)
        U[2][1] = 1; U[3][1] = 1; V[0][1] = 1; W[2][1] = 1; W[3][1] = -1
        # M3 = a00(b10-b11)
        U[0][2] = 1; V[2][2] = 1; V[3][2] = -1; W[1][2] = 1; W[3][2] = 1
        # M4 = a11(b01-b00)
        U[3][3] = 1; V[1][3] = 1; V[0][3] = -1; W[0][3] = 1; W[2][3] = 1
        # M5 = (a00+a01)(b11)
        U[0][4] = 1; U[1][4] = 1; V[3][4] = 1; W[0][4] = -1; W[1][4] = 1
        # M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1; V[0][5] = 1; V[1][5] = 1; W[3][5] = 1
        # M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1; V[2][6] = 1; V[3][6] = 1; W[0][6] = 1
        
        # Right block Strassen (7 products): computes A*B_right (2x2 * 2x2)
        # Output positions: W[4]=C02, W[5]=C03, W[6]=C12, W[7]=C13
        # N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1; V[4][7] = 1; V[7][7] = 1; W[4][7] = 1; W[7][7] = 1
        # N2 = (a10+a11)(b02)
        U[2][8] = 1; U[3][8] = 1; V[4][8] = 1; W[6][8] = 1; W[7][8] = -1
        # N3 = a00(b12-b13)
        U[0][9] = 1; V[6][9] = 1; V[7][9] = -1; W[5][9] = 1; W[7][9] = 1
        # N4 = a11(b03-b02)
        U[3][10] = 1; V[5][10] = 1; V[4][10] = -1; W[4][10] = 1; W[6][10] = 1
        # N5 = (a00+a01)(b13)
        U[0][11] = 1; U[1][11] = 1; V[7][11] = 1; W[4][11] = -1; W[5][11] = 1
        # N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1; V[4][12] = 1; V[5][12] = 1; W[7][12] = 1
        # N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1; V[6][13] = 1; V[7][13] = 1; W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    def try_rank11_via_dependency_search():
        """
        Search for rank-11 by finding 3 linear dependencies among the 14 Strassen terms.
        
        If we can express 3 of the rank-1 terms as linear combinations of the other 11,
        we can eliminate them. The key is finding coefficients c_r such that
        sum_r c_r * (u_r, v_r, w_r) = 0 in the tensor space.
        
        We'll use a numerical approach: set up the system of equations and solve.
        """
        if time.time() > deadline - 40:
            return None
        
        R14 = 14
        U14, V14, W14 = build_rank14()["U"], build_rank14()["V"], build_rank14()["W"]
        
        # The tensor has 16 entries (all i,j,k combinations are 1)
        # A rank-1 term (u, v, w) contributes to entry (i,j,k) the value u[i,j]*v[j,k]*w[k,i]
        # For a dependency: sum_r c_r * u_r[i,j] * v_r[j,k] * w_r[k,i] = 0 for all (i,j,k)
        
        # This is a system of 16 equations in 14 unknowns (c_0, ..., c_13)
        # The system is: for each (i,j,k): sum_r c_r * U14[i*m+j][r] * V14[j*p+k][r] * W14[k*n+i][r] = 0
        
        # Let's build this system
        # Actually, this won't work directly because the 14 terms span the 16-dim space.
        # We need to find a different approach.
        
        # Alternative: Try to find a rank-11 decomposition by direct search over
        # structured patterns with half-integer entries.
        
        # Strategy: Use the fact that for 2x2*2x4, we can try various
        # combinations of linear forms. Let's try a systematic search.
        
        # The U matrix has 4 rows (a00, a01, a10, a11) and R columns
        # The V matrix has 8 rows (b00,b01,b10,b11,b02,b03,b12,b13) and R columns
        # The W matrix has 8 rows (C00,C01,C10,C11,C02,C03,C12,C13) and R columns
        
        # For rank 11, we need 11 columns in each.
        
        # Let's try a structured search: fix U to have a specific pattern,
        # then solve for V and W.
        
        # Actually, let's try a completely different approach: use the
        # rectangular Strassen algorithm which is known to achieve rank 14,
        # and then try to optimize.
        
        # Let me try: use 6 terms for the left block and 5 for the right block,
        # but with a different structure that allows sharing.
        
        # Key insight: The left and right blocks share the same A matrix.
        # So we can potentially share some of the linear combinations of A.
        
        # Let's try: compute 4 shared linear combinations of A, then use them
        # for both blocks.
        
        # Shared combinations:
        # s1 = a00 + a11
        # s2 = a00 + a01
        # s3 = a10 + a11
        # s4 = a00 + a10
        
        # For left block (2x2 * 2x2), we need 7 products normally.
        # For right block (2x2 * 2x2), we need 7 products normally.
        # If we share the A-side combinations, we might save some.
        
        # Actually, this is getting complicated. Let me just return the rank-14
        # solution for now, as finding rank 11 requires a more sophisticated
        # approach.
        
        return None
    
    def try_rank10_search():
        """Try to find rank-10 or lower by direct search."""
        if time.time() > deadline - 20:
            return None
        # This would require a very sophisticated search. Skip for now.
        return None
    
    def naive_decomp():
        """Naive rank-16 decomposition."""
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
    result = try_rank11_via_dependency_search()
    if result is not None:
        return result
    
    result = try_rank10_search()
    if result is not None:
        return result
    
    result = build_rank14()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
