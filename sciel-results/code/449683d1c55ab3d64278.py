import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]
DEADLINE = time.time() + 145.0  # 5s safety margin

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Start with naive rank-96 decomposition
    naive_rank = n * m * p
    U_naive = [[0] * naive_rank for _ in range(n * m)]
    V_naive = [[0] * naive_rank for _ in range(m * p)]
    W_naive = [[0] * naive_rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i * m + j][col] = 1
                V_naive[j * p + k][col] = 1
                W_naive[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_naive, "V": V_naive, "W": W_naive}
    best_rank = naive_rank
    
    def verify(U, V, W, rank):
        """Check if the decomposition is valid"""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        
        # Compute the tensor from the decomposition
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
        
        # The tensor should be all ones
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True
    
    # Verify naive solution
    if not verify(best_result["U"], best_result["V"], best_result["W"], best_rank):
        return best_result
    
    # Try to improve using Winograd-like block decomposition
    # Strategy: Split the multiplication into blocks and use Strassen for 2x2x2 sub-problems
    # 
    # Decompose 3x4 x 4x8 as:
    # Split m=4 into 2+2
    # For each pair (i,k), C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    #                      = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #
    # First sum: 3x2 x 2x8 product (naive rank 48)
    # Second sum: 3x2 x 2x8 product (naive rank 48)
    #
    # For each 3x2 x 2x8, split n=3 into 2+1, p=8 into 4+4:
    # (2+1) x 2 x (4+4) = 2x2x4 + 2x2x4 + 1x2x4 + 1x2x4
    #
    # For 2x2x4: split p=4 into 2+2, get two 2x2x2 problems
    # Each 2x2x2 can use Strassen rank 7 instead of 8
    #
    # Total rank calculation:
    # Two 3x2x2x8 blocks:
    #   Each has: 2*(2x2x2) + 2*(1x2x2) = 2*7 + 2*4 = 22
    #   Wait, let me recalculate.
    #
    # Actually, let me just try to construct a rank-88 decomposition by finding
    # 8 linear dependencies in the naive 96-rank decomposition.
    #
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # A rank-r decomposition: sum_{r=1}^{R} U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
    #
    # In the naive decomposition, term (i,j,k) has U[i,j]=1, V[j,k]=1, W[k,i]=1.
    # We can try to replace some terms with linear combinations.
    #
    # Key insight: For fixed i and k, sum_j A[i,j]*B[j,k] = C[i,k]
    # The naive terms for fixed (i,k) are: for each j, term (i,j,k)
    # These 4 terms (j=0,1,2,3) all contribute 1 to C[i,k].
    #
    # If we can find a linear relation among these terms, we can reduce the rank.
    # But in the naive decomposition, each term is independent.
    #
    # Let me try a different approach: use the fact that we can factor some terms.
    #
    # For now, let me just return the naive solution with proper deadline checking.
    
    if time.time() > DEADLINE:
        return best_result
    
    # Try a simple improvement: use rank-7 Strassen for 2x2x2 sub-problems
    # by carefully constructing the decomposition.
    
    # Let me define a helper to add a 2x2x2 Strassen block
    def add_strassen_2x2x2(U_list, V_list, W_list, offset_i, offset_j, offset_k, col_start):
        """Add a 2x2x2 Strassen decomposition at the given offsets.
        Returns the new column index."""
        # For 2x2x2: i in {0,1}, j in {0,1}, k in {0,1}
        # U index: (offset_i + i)*m + (offset_j + j) where m=2 for this sub-block
        # But we're in the full problem, so m=4, p=8, n=3
        # U row: (offset_i + i)*4 + (offset_j + j)
        # V row: (offset_j + j)*8 + (offset_k + k)
        # W row: (offset_k + k)*3 + (offset_i + i)
        
        # Strassen 7 terms for 2x2x2:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B11)
        # M4 = A11(B01-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A10)(B10+B11)
        #
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        terms = [
            # (U_coeffs, V_coeffs, W_coeffs)
            # M1: (A00+A11)(B00+B11) -> C00:+1, C11:+1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # M2: (A10+A11)B00 -> C10:+1, C11:-1
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            # M3: A00(B10-B11) -> C01:+1, C11:+1
            ([(0,0,1)], [(1,0,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            # M4: A11(B01-B00) -> C00:+1, C10:+1
            ([(1,1,1)], [(0,1,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            # M5: (A00+A01)B11 -> C00:-1, C01:+1
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            # M6: (A10-A00)(B00+B01) -> not used in C formulas above
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], []),
            # M7: (A01-A10)(B10+B11) -> C00:-1, C11:+1
            ([(0,1,1), (1,0,-1)], [(1,0,1), (1,1,1)], [(0,0,-1), (1,1,1)]),
        ]
        
        col = col_start
        for u_coeffs, v_coeffs, w_coeffs in terms:
            if not w_coeffs:
                # M6 is not used, skip it
                continue
            for (ui, uj, uval) in u_coeffs:
                row_u = (offset_i + ui) * m + (offset_j + uj)
                U_list[row_u][col] += uval
            for (vj, vk, vval) in v_coeffs:
                row_v = (offset_j + vj) * p + (offset_k + vk)
                V_list[row_v][col] += vval
            for (wk, wi, wval) in w_coeffs:
                row_w = (offset_k + wk) * n + (offset_i + wi)
                W_list[row_w][col] += wval
            col += 1
        
        return col
    
    # Let's try to build a better decomposition
    # Split the problem into blocks:
    # n=3 -> 2+1, m=4 -> 2+2, p=8 -> 4+4
    #
    # Block (i_block, j_block, k_block) where:
    # i_block: 0 (rows 0-1), 1 (row 2)
    # j_block: 0 (cols 0-1), 1 (cols 2-3)
    # k_block: 0 (cols 0-3), 1 (cols 4-7)
    #
    # For i_block=0, j_block=0, k_block=0: 2x2x4 -> split k into 2+2: two 2x2x2
    # For i_block=0, j_block=1, k_block=0: 2x2x4 -> split k into 2+2: two 2x2x2
    # For i_block=0, j_block=0, k_block=1: 2x2x4 -> split k into 2+2: two 2x2x2
    # For i_block=0, j_block=1, k_block=1: 2x2x4 -> split k into 2+2: two 2x2x2
    # For i_block=1, j_block=0, k_block=0: 1x2x4 -> naive rank 8
    # For i_block=1, j_block=1, k_block=0: 1x2x4 -> naive rank 8
    # For i_block=1, j_block=0, k_block=1: 1x2x4 -> naive rank 8
    # For i_block=1, j_block=1, k_block=1: 1x2x4 -> naive rank 8
    #
    # Total: 8 * 7 + 4 * 8 = 56 + 32 = 88
    #
    # Let's construct this!
    
    U_new = [[0] * 88 for _ in range(n * m)]
    V_new = [[0] * 88 for _ in range(m * p)]
    W_new = [[0] * 88 for _ in range(p * n)]
    
    col = 0
    # Process the 2x2x2 blocks for i_block=0
    for j_block in [0, 1]:
        for k_block in [0, 1]:
            # Two 2x2x2 sub-blocks for each (j_block, k_block)
            for k_sub in [0, 1]:
                offset_i = 0
                offset_j = j_block * 2
                offset_k = k_block * 4 + k_sub * 2
                col = add_strassen_2x2x2(U_new, V_new, W_new, offset_i, offset_j, offset_k, col)
    
    # Process the 1x2x4 blocks for i_block=1 (naive)
    for j_block in [0, 1]:
        for k_block in [0, 1]:
            offset_i = 2
            offset_j = j_block * 2
            offset_k = k_block * 4
            for j in range(2):
                for k in range(4):
                    row_u = (offset_i) * m + (offset_j + j)
                    row_v = (offset_j + j) * p + (offset_k + k)
                    row_w = (offset_k + k) * n + (offset_i)
                    U_new[row_u][col] = 1
                    V_new[row_v][col] = 1
                    W_new[row_w][col] = 1
                    col += 1
    
    # Verify the new decomposition
    if col == 88 and verify(U_new, V_new, W_new, 88):
        best_result = {"U": U_new, "V": V_new, "W": W_new}
        best_rank = 88
    
    if time.time() > DEADLINE:
        return best_result
    
    return best_result
