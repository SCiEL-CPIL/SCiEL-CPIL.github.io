"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 4, 8, 8
    
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
    
    def try_strassen_rank14():
        """
        Rank-14 decomposition for 2x2 * 2x4 using two Strassen 2x2 blocks.
        """
        R = 14
        U = [[0]*R for _ in range(4)]
        V = [[0]*R for _ in range(8)]
        W = [[0]*R for _ in range(8)]
        
        # Left block (cols 0,1 of B) - standard Strassen
        # r=0: M1 = (a00+a11)(b00+b11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[5][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # r=1: M2 = (a10+a11)b00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # r=2: M3 = a00(b01-b11)
        U[0][2] = 1
        V[1][2] = 1; V[5][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # r=3: M4 = a11(b10-b00)
        U[3][3] = 1
        V[4][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # r=4: M5 = (a00+a01)b11
        U[0][4] = 1; U[1][4] = 1
        V[5][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # r=5: M6 = (a10-a00)(b00+b01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # r=6: M7 = (a01-a11)(b10+b11)
        U[1][6] = 1; U[3][6] = -1
        V[4][6] = 1; V[5][6] = 1
        W[0][6] = 1
        
        # Right block (cols 2,3 of B) - standard Strassen
        # r=7: N1 = (a00+a11)(b02+b13)
        U[0][7] = 1; U[3][7] = 1
        V[2][7] = 1; V[7][7] = 1
        W[4][7] = 1; W[7][7] = 1
        # r=8: N2 = (a10+a11)b02
        U[2][8] = 1; U[3][8] = 1
        V[2][8] = 1
        W[5][8] = 1; W[7][8] = -1
        # r=9: N3 = a00(b03-b13)
        U[0][9] = 1
        V[3][9] = 1; V[7][9] = -1
        W[6][9] = 1; W[7][9] = 1
        # r=10: N4 = a11(b12-b02)
        U[3][10] = 1
        V[6][10] = 1; V[2][10] = -1
        W[4][10] = 1; W[5][10] = 1
        # r=11: N5 = (a00+a01)b13
        U[0][11] = 1; U[1][11] = 1
        V[7][11] = 1
        W[4][11] = -1; W[6][11] = 1
        # r=12: N6 = (a10-a00)(b02+b03)
        U[2][12] = 1; U[0][12] = -1
        V[2][12] = 1; V[3][12] = 1
        W[7][12] = 1
        # r=13: N7 = (a01-a11)(b12+b13)
        U[1][13] = 1; U[3][13] = -1
        V[6][13] = 1; V[7][13] = 1
        W[4][13] = 1
        
        if verify(U, V, W):
            return {"U": U, "V": V, "W": W}
        return None
    
    def try_rank13_search():
        """
        Search for rank-13 decomposition by trying to share one rank-1 term
        between the two Strassen blocks. We fix U and W with a shared term
        structure and solve for V using linear algebra.
        
        Strategy: The rank-14 solution has 14 independent rank-1 terms.
        For rank 13, we need one term to serve both blocks. We try all
        possible "merge" patterns where a left-block term and a right-block
        term are combined into a single term.
        """
        if time.time() > deadline - 10:
            return None
        
        # Standard Strassen factors for a 2x2 block
        # Each term r has (U_vec, V_vec, W_vec) where U_vec is length 4, V_vec length 2, W_vec length 2
        strassen_terms = [
            # (U, V, W) for each of the 7 Strassen terms
            ([1,0,0,1], [1,1], [1,-1]),   # M1: (a00+a11)(b00+b11) -> C[0,0]+C[1,1]... 
            # Actually let me use the standard Strassen formulation:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            # C00 = M1 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
        ]
        
        # Let me define the Strassen terms more carefully for the 2x2 case
        # U indices: 0=a00, 1=a01, 2=a10, 3=a11
        # V indices: 0=b00, 1=b01 (for left block), 2=b02, 3=b03 (for right block)
        # W indices: 0=C00, 1=C10, 2=C01, 3=C11 (left), 4=C02, 5=C12, 6=C03, 7=C13 (right)
        
        # For the left block, Strassen terms (U_4, V_2, W_2):
        left_terms = [
            ([1,0,0,1], [1,1], [1,0,0,-1]),    # M1: (a00+a11)(b00+b11) -> C00 + C11
            ([0,0,1,1], [1,0], [0,1,-1,0]),    # M2: (a10+a11)b00 -> C10 - C11
            ([1,0,0,0], [0,1], [0,0,1,1]),     # M3: a00(b01-b11) -> C01 + C11
            ([0,0,0,1], [1,0], [1,-1,0,0]),    # M4: a11(b10-b00) -> C00 + C10
            ([1,1,0,0], [0,1], [-1,0,1,0]),    # M5: (a00+a01)b11 -> -C00 + C01
            ([1,-1,0,0], [1,1], [0,0,0,1]),    # M6: (a10-a00)(b00+b01) -> C11
            ([0,1,-1,0], [1,1], [1,0,0,0]),    # M7: (a01-a11)(b10+b11) -> C00
        ]
        
        # Verify left block:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # For the right block, similar structure with b02,b03 and C02,C12,C03,C13
        right_terms = [
            ([1,0,0,1], [1,1], [0,0,0,0,1,0,0,-1]),    # N1
            ([0,0,1,1], [1,0], [0,0,0,0,0,1,-1,0]),    # N2
            ([1,0,0,0], [0,1], [0,0,0,0,0,0,1,1]),     # N3
            ([0,0,0,1], [1,0], [0,0,0,0,1,-1,0,0]),    # N4
            ([1,1,0,0], [0,1], [0,0,0,0,-1,0,0,1]),    # N5
            ([1,-1,0,0], [1,1], [0,0,0,0,0,0,0,1]),    # N6
            ([0,1,-1,0], [1,1], [0,0,0,0,1,0,0,0]),    # N7
        ]
        
        # Now try to find a rank-13 solution by merging one left term with one right term
        # The merged term must have the same U vector (up to scalar) for both blocks
        # and the V/W vectors must be compatible
        
        # For a merge to work, the U vectors of the left and right terms must be proportional
        # (since they multiply the same A matrix entries)
        
        # Check which pairs of left/right terms have proportional U vectors
        for li in range(7):
            for ri in range(7):
                U_left = np.array(left_terms[li][0], dtype=float)
                U_right = np.array(right_terms[ri][0], dtype=float)
                
                # Check if U_left and U_right are proportional
                # They need to be linearly dependent
                if np.linalg.matrix_rank(np.column_stack([U_left, U_right])) < 2:
                    # They are proportional - potential merge candidate
                    # The merged U is U_left (or U_right, they're proportional)
                    # V_left is length 2 (for b00,b01), V_right is length 2 (for b02,b03)
                    # Merged V is length 4: [V_left[0], V_left[1], V_right[0], V_right[1]]
                    # W_left is length 4 (for C00,C10,C01,C11), W_right is length 4 (for C02,C12,C03,C13)
                    # Merged W is length 8: [W_left[0], W_left[1], W_left[2], W_left[3], W_right[0], W_right[1], W_right[2], W_right[3]]
                    
                    # Build the rank-13 decomposition
                    R = 13
                    U_mat = np.zeros((4, R))
                    V_mat = np.zeros((8, R))
                    W_mat = np.zeros((8, R))
                    
                    # Column 0: merged term
                    U_mat[:, 0] = U_left
                    V_mat[:2, 0] = left_terms[li][1]  # b00, b01
                    V_mat[2:4, 0] = right_terms[ri][1]  # b02, b03
                    W_mat[:4, 0] = left_terms[li][2]  # C00,C10,C01,C11
                    W_mat[4:8, 0] = right_terms[ri][2]  # C02,C12,C03,C13
                    
                    # Remaining left terms (excluding li)
                    col = 1
                    for l_idx in range(7):
                        if l_idx == li:
                            continue
                        U_mat[:, col] = left_terms[l_idx][0]
                        V_mat[:2, col] = left_terms[l_idx][1]
                        W_mat[:4, col] = left_terms[l_idx][2]
                        col += 1
                    
                    # Remaining right terms (excluding ri)
                    for r_idx in range(7):
                        if r_idx == ri:
                            continue
                        U_mat[:, col] = right_terms[r_idx][0]
                        V_mat[2:4, col] = right_terms[r_idx][1]
                        W_mat[4:8, col] = right_terms[r_idx][2]
                        col += 1
                    
                    # Now solve for V_mat using the constraint:
                    # For each (i,j,k): sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
                    # This is: for each (i,j,k), sum_r (U[i*m+j][r] * W[k*n+i][r]) * V[j*p+k][r] = 1
                    # Let A[i,j,k,r] = U[i*m+j][r] * W[k*n+i][r]
                    # Then for each (i,j,k): sum_r A[i,j,k,r] * V[j*p+k][r] = 1
                    # This gives a system: for each (j,k), sum_r A[i,j,k,r] * V[j*p+k][r] = 1 for all i
                    # Wait, it's per (i,j,k) triple.
                    
                    # Actually: for fixed (j,k), we need V[j*p+k][r] for all r.
                    # The constraints are: for each i, sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] = 1
                    # This is a linear system in V[j*p+k][r]:
                    # sum_r (U[i*m+j][r] * W[k*n+i][r]) * V[j*p+k][r] = 1, for each i
                    # So for each (j,k), we have n=2 equations in R=13 unknowns.
                    # We have 8*(j,k) pairs, each giving 2 equations = 16 equations total.
                    # But V has 8*13 = 104 unknowns.
                    # This is underdetermined, so we need to check if a solution exists.
                    
                    # Actually, the system is:
                    # For each (i,j,k): sum_r U[i*m+j][r] * W[k*n+i][r] * V[j*p+k][r] = 1
                    # Let's form the matrix B where B[(i,j,k), r] = U[i*m+j][r] * W[k*n+i][r]
                    # Then B @ V_flat = 1 (vector of ones)
                    # V_flat is of length 8*13 = 104
                    # B is 16 x 104
                    # We need to find V such that B @ V = ones(16)
                    
                    B = np.zeros((16, R * 8))
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                idx = i * m * p + j * p + k
                                for r in range(R):
                                    B[idx, r * 8 + j * p + k] = U_mat[i * m + j, r] * W_mat[k * n + i, r]
                    
                    # Solve B @ V_flat = ones(16) using least squares
                    rhs = np.ones(16)
                    try:
                        V_flat, residuals, rank, sv = np.linalg.lstsq(B, rhs, rcond=None)
                    except:
                        continue
                    
                    # Check if the solution is exact
                    B_V = B @ V_flat
                    if not np.allclose(B_V, rhs, atol=1e-9):
                        continue
                    
                    # Reshape V_flat
                    V_sol = V_flat.reshape(R, 8).T  # shape (8, R)
                    
                    # Convert to list of lists
                    U_list = U_mat.tolist()
                    V_list = V_sol.tolist()
                    W_list = W_mat.tolist()
                    
                    # Verify
                    if verify(U_list, V_list, W_list):
                        # Check if entries are "nice" (integers or half-integers)
                        # For now, accept any exact solution
                        return {"U": U_list, "V": V_list, "W": W_list}
        
        return None
    
    # Try rank-13 search first
    result = try_rank13_search()
    if result is not None:
        return result
    
    # Try the structured rank-14 decomposition
    result = try_strassen_rank14()
    if result is not None:
        return result
    
    # Fallback to naive rank 16
    return naive_decomp()
# EVOLVE-BLOCK-END
