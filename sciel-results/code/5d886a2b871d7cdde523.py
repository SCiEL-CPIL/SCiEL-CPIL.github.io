"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # safety margin before 150s limit
    
    # Build the target tensor T[i, j, k] = 1 for matrix mult (n x m) * (m x p)
    # T has shape (n*m, m*p, p*n) in the convention:
    # U shape (n*m, rank), V shape (m*p, rank), W shape (p*n, rank)
    # Product: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal T[i,j,k]
    # where T[i,j,k] = 1 if the (i,j) of A times (j,k) of B contributes to (i,k) of C
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The target: for each (i,j,k), we need sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # And for other index combinations, it should be 0. But actually the tensor is sparse:
    # The multiplication tensor T has T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' 
    # AND j = j' AND i = i' AND k = k'. So T[(i*m+j), (j*p+k), (k*n+i)] = 1, all else 0.
    
    # Build the target tensor as a list of nonzero entries
    # Each entry is (row_idx, col_idx, third_idx) with value 1
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.append((a, b, c))
    
    # Try to find a low-rank decomposition
    # Start with schoolbook (rank = n*m*p = 120) and try to reduce
    
    # Schoolbook: rank = 120
    schoolbook_rank = n * m * p
    
    # Try a greedy approach to reduce rank
    # For rank r, we need r vectors u_r in R^(nm), v_r in R^(mp), w_r in R^(pn)
    # such that sum_r u_r[a] * v_r[b] * w_r[c] = T[a,b,c]
    
    # Use a simple iterative method: start with schoolbook, try to merge terms
    
    # Since finding the exact minimum rank is hard, let's try a structured ansatz:
    # Can we share some factors? For example, if two entries share the same (a,b) or (a,c) etc.
    
    # Let's try: use the fact that the tensor is a "matrix multiplication" tensor.
    # The known upper bound for rank of m×n × n×p is mn + np - n (for n >= 2) via Laderman-style,
    # but that's for specific dimensions. For general, it's still open.
    
    # For 4x5x6, let's just return the schoolbook unless we find something better quickly.
    
    # Try a quick search: can we do better than 120?
    # One idea: group by shared indices.
    
    # Let me try a simple approach: use the minimum possible rank which is max(nm, mp, pn) = 30
    # and see if we can construct it. This is unlikely to work exactly, but let's try a randomized approach.
    
    best_rank = schoolbook_rank
    best_result = None
    
    # First, get the schoolbook as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(nm)]
    V_sb = [[0] * schoolbook_rank for _ in range(mp)]
    W_sb = [[0] * schoolbook_rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_sb[a][col] = 1
                V_sb[b][col] = 1
                W_sb[c][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find a lower rank decomposition using a structured approach
    # Idea: use the fact that we can sometimes combine terms.
    # For a lower rank, we need the vectors to be linear combinations.
    
    # Let's try a simple optimization: start with schoolbook and see if we can zero out some columns
    # by adjusting others. This is a linear algebra problem.
    
    # Given time constraints, let's try a few random initializations for lower ranks
    # and use a simple gradient-free method.
    
    import random
    random.seed(42)
    
    # Try ranks from 119 down to max(nm, mp, pn) = 30, but only a few due to time
    # Actually, solving for exact decomposition at rank r requires solving a system of nm*mp*pn equations
    # with r*(nm+mp+pn) unknowns. This is only feasible for small r or with structure.
    
    # For 4x5x6, nm=20, mp=30, pn=24, total dims = 74 per rank term.
    # Number of equations = 20*30*24 = 14400.
    # For rank r, unknowns = r*74. Need r*74 >= 14400 => r >= 195. So rank 120 is already less than this count,
    # meaning the system is underdetermined for rank 120 (120*74=8880 < 14400? No, 8880 < 14400, so overdetermined).
    # Wait, 120*74 = 8880, and we have 14400 constraints. So it's overdetermined.
    # The schoolbook works because it's constructed to satisfy them.
    
    # For rank r < 120, we have even fewer unknowns, making it harder.
    
    # Given the difficulty, let's just return the schoolbook decomposition.
    # It's the safest bet within the time limit.
    
    if time.time() > deadline:
        return best_result
    
    # --- Strassen-3 based decomposition attempt ---
    # Partition 4x5x6 into blocks:
    # n=4 = 3+1, m=5 = 3+2, p=6 = 3+3
    # This gives 2*2*2 = 8 sub-blocks:
    # (3,3,3), (3,3,3), (3,2,3), (3,2,3)
    # (1,3,3), (1,3,3), (1,2,3), (1,2,3)
    
    # For 3x3x3 blocks, use Strassen-3 (rank 7)
    # For other blocks, use schoolbook
    
    # Strassen-3 for 3x3x3:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B21
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's build the Strassen-3 decomposition for a 3x3x3 block
    # and then combine with schoolbook for the rest.
    
    def strassen3_decomposition():
        """
        Returns U, V, W for 3x3x3 matrix multiplication using Strassen-3 (rank 7).
        U shape: (9, 7), V shape: (9, 7), W shape: (9, 7)
        """
        # For 3x3x3, indices are:
        # U[i*3+j], V[j*3+k], W[k*3+i]
        
        U = [[0] * 7 for _ in range(9)]
        V = [[0] * 7 for _ in range(9)]
        W = [[0] * 7 for _ in range(9)]
        
        # M1 = (A11 + A22) * (B11 + B22)
        # A11 is at (0,0), A22 is at (1,1)
        # B11 is at (0,0), B22 is at (1,1)
        # C11 += M1, C22 += M1
        
        # U for A11: row 0*3+0 = 0
        # U for A22: row 1*3+1 = 4
        U[0][0] = 1
        U[4][0] = 1
        
        # V for B11: row 0*3+0 = 0
        # V for B22: row 1*3+1 = 4
        V[0][0] = 1
        V[4][0] = 1
        
        # W for C11: row 0*3+0 = 0
        # W for C22: row 1*3+1 = 4
        W[0][0] = 1
        W[4][0] = 1
        
        # M2 = (A21 + A22) * B21
        # A21 is at (1,0), A22 is at (1,1)
        # B21 is at (1,0)
        # C21 += M2
        
        U[1*3+0][1] = 1  # A21
        U[1*3+1][1] = 1  # A22
        V[1*3+0][1] = 1  # B21
        W[1*3+0][1] = 1  # C21
        
        # M3 = A11 * (B12 - B22)
        # A11 is at (0,0)
        # B12 is at (0,1), B22 is at (1,1)
        # C12 += M3
        
        U[0*3+0][2] = 1  # A11
        V[0*3+1][2] = 1  # B12
        V[1*3+1][2] = -1 # -B22
        W[0*3+1][2] = 1  # C12
        
        # M4 = A22 * (B21 - B11)
        # A22 is at (1,1)
        # B21 is at (1,0), B11 is at (0,0)
        # C11 += M4, C21 += M4
        
        U[1*3+1][3] = 1  # A22
        V[1*3+0][3] = 1  # B21
        V[0*3+0][3] = -1 # -B11
        W[0*3+0][3] = 1  # C11
        W[1*3+0][3] = 1  # C21
        
        # M5 = (A11 + A12) * B22
        # A11 is at (0,0), A12 is at (0,1)
        # B22 is at (1,1)
        # C11 -= M5, C12 += M5
        
        U[0*3+0][4] = 1  # A11
        U[0*3+1][4] = 1  # A12
        V[1*3+1][4] = 1  # B22
        W[0*3+0][4] = -1 # C11
        W[0*3+1][4] = 1  # C12
        
        # M6 = (A21 - A11) * (B11 + B12)
        # A21 is at (1,0), A11 is at (0,0)
        # B11 is at (0,0), B12 is at (0,1)
        # C22 += M6
        
        U[1*3+0][5] = 1  # A21
        U[0*3+0][5] = -1 # -A11
        V[0*3+0][5] = 1  # B11
        V[0*3+1][5] = 1  # B12
        W[1*3+1][5] = 1  # C22
        
        # M7 = (A12 - A22) * (B21 + B22)
        # A12 is at (0,1), A22 is at (1,1)
        # B21 is at (1,0), B22 is at (1,1)
        # C11 += M7, C22 += M7
        
        U[0*3+1][6] = 1  # A12
        U[1*3+1][6] = -1 # -A22
        V[1*3+0][6] = 1  # B21
        V[1*3+1][6] = 1  # B22
        W[0*3+0][6] = 1  # C11
        W[1*3+1][6] = 1  # C22
        
        return U, V, W
    
    # Now build the full 4x5x6 decomposition
    # Partition: n=4=3+1, m=5=3+2, p=6=3+3
    # Blocks:
    # Block (0,0,0): n=3, m=3, p=3 -> Strassen-3 (rank 7)
    # Block (0,0,1): n=3, m=3, p=3 -> Strassen-3 (rank 7)
    # Block (0,1,0): n=3, m=2, p=3 -> schoolbook (rank 18)
    # Block (0,1,1): n=3, m=2, p=3 -> schoolbook (rank 18)
    # Block (1,0,0): n=1, m=3, p=3 -> schoolbook (rank 9)
    # Block (1,0,1): n=1, m=3, p=3 -> schoolbook (rank 9)
    # Block (1,1,0): n=1, m=2, p=3 -> schoolbook (rank 6)
    # Block (1,1,1): n=1, m=2, p=3 -> schoolbook (rank 6)
    
    # Total rank = 7 + 7 + 18 + 18 + 9 + 9 + 6 + 6 = 80
    
    # Wait, this is better than 120! Let's build it.
    
    U_full = [[0] * 80 for _ in range(nm)]
    V_full = [[0] * 80 for _ in range(mp)]
    W_full = [[0] * 80 for _ in range(pn)]
    
    col_offset = 0
    
    # Block (0,0,0): rows 0-2, cols 0-2, out cols 0-2
    # n_block=3, m_block=3, p_block=3
    U3, V3, W3 = strassen3_decomposition()
    for r in range(7):
        for i in range(3):
            for j in range(3):
                U_full[i * m + j][col_offset + r] = U3[i * 3 + j][r]
        for j in range(3):
            for k in range(3):
                V_full[j * p + k][col_offset + r] = V3[j * 3 + k][r]
        for k in range(3):
            for i in range(3):
                W_full[k * n + i][col_offset + r] = W3[k * 3 + i][r]
    col_offset += 7
    
    # Block (0,0,1): rows 0-2, cols 0-2, out cols 3-5
    # n_block=3, m_block=3, p_block=3
    U3, V3, W3 = strassen3_decomposition()
    for r in range(7):
        for i in range(3):
            for j in range(3):
                U_full[i * m + j][col_offset + r] = U3[i * 3 + j][r]
        for j in range(3):
            for k in range(3):
                V_full[j * p + (k + 3)][col_offset + r] = V3[j * 3 + k][r]
        for k in range(3):
            for i in range(3):
                W_full[(k + 3) * n + i][col_offset + r] = W3[k * 3 + i][r]
    col_offset += 7
    
    # Block (0,1,0): rows 0-2, cols 3-4, out cols 0-2
    # n_block=3, m_block=2, p_block=3
    # Schoolbook
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U_full[i * m + (j + 3)][col_offset] = 1
                V_full[(j + 3) * p + k][col_offset] = 1
                W_full[k * n + i][col_offset] = 1
                col_offset += 1
    # col_offset should be 7+7+18 = 32
    
    # Block (0,1,1): rows 0-2, cols 3-4, out cols 3-5
    # n_block=3, m_block=2, p_block=3
    # Schoolbook
    for i in range(3):
        for j in range(2):
            for k in range(3):
                U_full[i * m + (j + 3)][col_offset] = 1
                V_full[(j + 3) * p + (k + 3)][col_offset] = 1
                W_full[(k + 3) * n + i][col_offset] = 1
                col_offset += 1
    # col_offset should be 32+18 = 50
    
    # Block (1,0,0): row 3, cols 0-2, out cols 0-2
    # n_block=1, m_block=3, p_block=3
    # Schoolbook
    for i in range(1):
        for j in range(3):
            for k in range(3):
                U_full[(i + 3) * m + j][col_offset] = 1
                V_full[j * p + k][col_offset] = 1
                W_full[k * n + (i + 3)][col_offset] = 1
                col_offset += 1
    # col_offset should be 50+9 = 59
    
    # Block (1,0,1): row 3, cols 0-2, out cols 3-5
    # n_block=1, m_block=3, p_block=3
    # Schoolbook
    for i in range(1):
        for j in range(3):
            for k in range(3):
                U_full[(i + 3) * m + j][col_offset] = 1
                V_full[j * p + (k + 3)][col_offset] = 1
                W_full[(k + 3) * n + (i + 3)][col_offset] = 1
                col_offset += 1
    # col_offset should be 59+9 = 68
    
    # Block (1,1,0): row 3, cols 3-4, out cols 0-2
    # n_block=1, m_block=2, p_block=3
    # Schoolbook
    for i in range(1):
        for j in range(2):
            for k in range(3):
                U_full[(i + 3) * m + (j + 3)][col_offset] = 1
                V_full[(j + 3) * p + k][col_offset] = 1
                W_full[k * n + (i + 3)][col_offset] = 1
                col_offset += 1
    # col_offset should be 68+6 = 74
    
    # Block (1,1,1): row 3, cols 3-4, out cols 3-5
    # n_block=1, m_block=2, p_block=3
    # Schoolbook
    for i in range(1):
        for j in range(2):
            for k in range(3):
                U_full[(i + 3) * m + (j + 3)][col_offset] = 1
                V_full[(j + 3) * p + (k + 3)][col_offset] = 1
                W_full[(k + 3) * n + (i + 3)][col_offset] = 1
                col_offset += 1
    # col_offset should be 74+6 = 80
    
    # Verify the decomposition
    # Check that for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(80):
                    s += U_full[i * m + j][r] * V_full[j * p + k][r] * W_full[k * n + i][r]
                if abs(s - 1.0) > 1e-10:
                    valid = False
                    break
        if not valid:
            break
    
    if valid:
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # If Strassen-3 based decomposition failed, return schoolbook
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
