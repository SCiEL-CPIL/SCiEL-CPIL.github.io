"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np

def _base_strassen_2x2x2():
    """
    Returns U, V, W for 2x2x2 Strassen decomposition.
    U: 4x7, V: 4x7, W: 4x7
    Indices:
    A: 0:00, 1:01, 2:10, 3:11
    B: 0:00, 1:01, 2:10, 3:11
    C: 0:00, 1:01, 2:10, 3:11 (k*n+i, n=2)
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # P0 = (A00 + A11)(B00 + B11)
    U[0, 0] += 1; U[3, 0] += 1
    V[0, 0] += 1; V[3, 0] += 1
    W[0, 0] += 1; W[3, 0] += 1
    
    # P1 = (A10 + A11)B00
    U[2, 1] += 1; U[3, 1] += 1
    V[0, 1] += 1
    W[2, 1] += 1; W[3, 1] -= 1
    
    # P2 = A00(B10 - B00)
    U[0, 2] += 1
    V[2, 2] += 1; V[0, 2] -= 1
    W[1, 2] += 1; W[3, 2] += 1
    
    # P3 = A11(B11 - B01)
    U[3, 3] += 1
    V[3, 3] += 1; V[1, 3] -= 1
    W[0, 3] += 1; W[2, 3] += 1
    
    # P4 = (A00 + A01)B11
    U[0, 4] += 1; U[1, 4] += 1
    V[3, 4] += 1
    W[0, 4] -= 1; W[1, 4] += 1
    
    # P5 = (A10 - A00)(B00 + B01)
    U[2, 5] += 1; U[0, 5] -= 1
    V[0, 5] += 1; V[1, 5] += 1
    W[3, 5] += 1
    
    # P6 = (A01 - A11)(B10 + B11)
    U[1, 6] += 1; U[3, 6] -= 1
    V[2, 6] += 1; V[3, 6] += 1
    W[0, 6] += 1
    
    return U, V, W

def _recursive_strassen(n, m, p):
    """
    Returns U, V, W for nxmxp Strassen decomposition.
    """
    if n == 2 and m == 2 and p == 2:
        return _base_strassen_2x2x2()
    
    n2, m2, p2 = n // 2, m // 2, p // 2
    U_sub, V_sub, W_sub = _recursive_strassen(n2, m2, p2)
    r_sub = U_sub.shape[1]
    r_total = 7 * r_sub
    
    U = np.zeros((n * m, r_total), dtype=int)
    V = np.zeros((m * p, r_total), dtype=int)
    W = np.zeros((p * n, r_total), dtype=int)
    
    # Define the 7 Strassen terms
    # Each term is a tuple of:
    # (A_indices, B_indices, C_indices)
    # A_indices: list of (local_row, local_col, sign) for the sub-A
    # But it's easier to map global to local.
    # For each term, we define how the local sub-matrices map to global.
    
    # Term 0: (A00 + A11)(B00 + B11)
    # A: local (i,j) -> global (i,j) and (i+n2, j+m2)
    # B: local (j,k) -> global (j,k) and (j+m2, k+p2)
    # C: local (k,i) -> global (k,i) and (k+p2, i+n2)
    
    # We iterate through the 7 terms
    for t in range(7):
        col_offset = t * r_sub
        
        # Determine the mapping for U (A)
        # For each local index in U_sub (row i, col j), we add to global U
        # The value is U_sub[i, c] * sign
        
        # We need to define the specific linear combinations for A, B, C
        # for each term t.
        
        # Helper to add contributions
        # A_block: dict mapping local (i,j) to list of (global_i, global_j, sign)
        # This is complex to generalize. Let's hardcode the 7 cases.
        
        if t == 0:
            # A: A00 + A11
            # Local (i, j) corresponds to Global (i, j) + (i+n2, j+m2)
            for i in range(n2):
                for j in range(m2):
                    idx_l = i * m2 + j
                    idx_g1 = i * m + j
                    idx_g2 = (i + n2) * m + (j + m2)
                    U[idx_g1, col_offset:col_offset+r_sub] += U_sub[idx_l]
                    U[idx_g2, col_offset:col_offset+r_sub] += U_sub[idx_l]
            
            # B: B00 + B11
            for j in range(m2):
                for k in range(p2):
                    idx_l = j * p2 + k
                    idx_g1 = j * p + k
                    idx_g2 = (j + m2) * p + (k + p2)
                    V[idx_g1, col_offset:col_offset+r_sub] += V_sub[idx_l]
                    V[idx_g2, col_offset:col_offset+r_sub] += V_sub[idx_l]
            
            # C: C00 + C11 (from formula C00=M0+..., C11=M0-...)
            # Wait, the recursive call returns W for the product M0.
            # M0 is added to C00 with +1 and C11 with -1?
            # No, the recursive call computes the matrix product M0.
            # The W_sub matrix maps the scalar products to the entries of M0.
            # M0 is a n2 x p2 matrix? No, M0 is n2 x p2?
            # A is n x m, B is m x p. Product is n x p.
            # Sub A is n2 x m2, Sub B is m2 x p2. Sub Product is n2 x p2.
            # W_sub indices are k * n2 + i.
            # We need to add M0 to C00 (top-left) and C11 (bottom-right).
            # C00 is at global rows k, cols i.
            # C11 is at global rows k+p2, cols i+n2.
            
            for k in range(p2):
                for i in range(n2):
                    idx_l = k * n2 + i
                    idx_g1 = k * n + i
                    idx_g2 = (k + p2) * n + (i + n2)
                    W[idx_g1, col_offset:col_offset+r_sub] += W_sub[idx_l]
                    W[idx_g2, col_offset:col_offset+r_sub] -= W_sub[idx_l]
                    
        elif t == 1:
            # P1 = (A10 + A11)B00
            # A: A10 + A11
            # Local (i, j) in sub-A corresponds to A10[i, j] + A11[i, j]
            # A10 is global rows n2..n-1, cols 0..m2-1
            # A11 is global rows n2..n-1, cols m2..m-1
            for i in range(n2):
                for j in range(m2):
                    idx_l = i * m2 + j
                    idx_g1 = (i + n2) * m + j
                    idx_g2 = (i + n2) * m + (j + m2)
                    U[idx_g1, col_offset:col_offset+r_sub] += U_sub[idx_l]
                    U[idx_g2, col_offset:col_offset+r_sub] += U_sub[idx_l]
            
            # B: B00
            for j in range(m2):
                for k in range(p2):
                    idx_l = j * p2 + k
                    idx_g1 = j * p + k
                    V[idx_g1, col_offset:col_offset+r_sub] += V_sub[idx_l]
            
            # C: M1 contributes to C10 (+1) and C11 (-1)
            # C10: rows p2..p-1, cols 0..n2-1
            # C11: rows p2..p-1, cols n2..n-1
            for k in range(p2):
                for i in range(n2):
                    idx_l = k * n2 + i
                    idx_g1 = (k + p2) * n + i
                    idx_g2 = (k + p2) * n + (i + n2)
                    W[idx_g1, col_offset:col_offset+r_sub] += W_sub[idx_l]
                    W[idx_g2, col_offset:col_offset+r_sub] -= W_sub[idx_l]

        elif t == 2:
            # P2 = A00(B10 - B00)
            # A: A00
            for i in range(n2):
                for j in range(m2):
                    idx_l = i * m2 + j
                    idx_g1 = i * m + j
                    U[idx_g1, col_offset:col_offset+r_sub] += U_sub[idx_l]
            
            # B: B10 - B00
            for j in range(m2):
                for k in range(p2):
                    idx_l = j * p2 + k
                    idx_g1 = (j + m2) * p + k
                    idx_g2 = j * p + k
                    V[idx_g1, col_offset:col_offset+r_sub] += V_sub[idx_l]
                    V[idx_g2, col_offset:col_offset+r_sub] -= V_sub[idx_l]
            
            # C: M2 contributes to C01 (+1) and C11 (+1)
            # C01: rows 0..p2-1, cols n2..n-1
            # C11: rows p2..p-1, cols n2..n-1
            for k in range(p2):
                for i in range(n2):
                    idx_l = k * n2 + i
                    idx_g1 = k * n + (i + n2)
                    idx_g2 = (k + p2) * n + (i + n2)
                    W[idx_g1, col_offset:col_offset+r_sub] += W_sub[idx_l]
                    W[idx_g2, col_offset:col_offset+r_sub] += W_sub[idx_l]

        elif t == 3:
            # P3 = A11(B11 - B01)
            # A: A11
            for i in range(n2):
                for j in range(m2):
                    idx_l = i * m2 + j
                    idx_g1 = (i + n2) * m + (j + m2)
                    U[idx_g1, col_offset:col_offset+r_sub] += U_sub[idx_l]
            
            # B: B11 - B01
            for j in range(m2):
                for k in range(p2):
                    idx_l = j * p2 + k
                    idx_g1 = (j + m2) * p + (k + p2)
                    idx_g2 = j * p + (k + p2)
                    V[idx_g1, col_offset:col_offset+r_sub] += V_sub[idx_l]
                    V[idx_g2, col_offset:col_offset+r_sub] -= V_sub[idx_l]
            
            # C: M3 contributes to C00 (+1) and C10 (+1)
            # C00: rows 0..p2-1, cols 0..n2-1
            # C10: rows p2..p-1, cols 0..n2-1
            for k in range(p2):
                for i in range(n2):
                    idx_l = k * n2 + i
                    idx_g1 = k * n + i
                    idx_g2 = (k + p2) * n + i
                    W[idx_g1, col_offset:col_offset+r_sub] += W_sub[idx_l]
                    W[idx_g2, col_offset:col_offset+r_sub] += W_sub[idx_l]

        elif t == 4:
            # P4 = (A00 + A01)B11
            # A: A00 + A01
            for i in range(n2):
                for j in range(m2):
                    idx_l = i * m2 + j
                    idx_g1 = i * m + j
                    idx_g2 = i * m + (j + m2)
                    U[idx_g1, col_offset:col_offset+r_sub] += U_sub[idx_l]
                    U[idx_g2, col_offset:col_offset+r_sub] += U_sub[idx_l]
            
            # B: B11
            for j in range(m2):
                for k in range(p2):
                    idx_l = j * p2 + k
                    idx_g1 = (j + m2) * p + (k + p2)
                    V[idx_g1, col_offset:col_offset+r_sub] += V_sub[idx_l]
            
            # C: M4 contributes to C00 (-1) and C01 (+1)
            # C00: rows 0..p2-1, cols 0..n2-1
            # C01: rows 0..p2-1, cols n2..n-1
            for k in range(p2):
                for i in range(n2):
                    idx_l = k * n2 + i
                    idx_g1 = k * n + i
                    idx_g2 = k * n + (i + n2)
                    W[idx_g1, col_offset:col_offset+r_sub] -= W_sub[idx_l]
                    W[idx_g2, col_offset:col_offset+r_sub] += W_sub[idx_l]

        elif t == 5:
            # P5 = (A10 - A00)(B00 + B01)
            # A: A10 - A00
            for i in range(n2):
                for j in range(m2):
                    idx_l = i * m2 + j
                    idx_g1 = (i + n2) * m + j
                    idx_g2 = i * m + j
                    U[idx_g1, col_offset:col_offset+r_sub] += U_sub[idx_l]
                    U[idx_g2, col_offset:col_offset+r_sub] -= U_sub[idx_l]
            
            # B: B00 + B01
            for j in range(m2):
                for k in range(p2):
                    idx_l = j * p2 + k
                    idx_g1 = j * p + k
                    idx_g2 = j * p + (k + p2)
                    V[idx_g1, col_offset:col_offset+r_sub] += V_sub[idx_l]
                    V[idx_g2, col_offset:col_offset+r_sub] += V_sub[idx_l]
            
            # C: M5 contributes to C11 (+1)
            # C11: rows p2..p-1, cols n2..n-1
            for k in range(p2):
                for i in range(n2):
                    idx_l = k * n2 + i
                    idx_g1 = (k + p2) * n + (i + n2)
                    W[idx_g1, col_offset:col_offset+r_sub] += W_sub[idx_l]

        elif t == 6:
            # P6 = (A01 - A11)(B10 + B11)
            # A: A01 - A11
            for i in range(n2):
                for j in range(m2):
                    idx_l = i * m2 + j
                    idx_g1 = i * m + (j + m2)
                    idx_g2 = (i + n2) * m + (j + m2)
                    U[idx_g1, col_offset:col_offset+r_sub] += U_sub[idx_l]
                    U[idx_g2, col_offset:col_offset+r_sub] -= U_sub[idx_l]
            
            # B: B10 + B11
            for j in range(m2):
                for k in range(p2):
                    idx_l = j * p2 + k
                    idx_g1 = (j + m2) * p + k
                    idx_g2 = (j + m2) * p + (k + p2)
                    V[idx_g1, col_offset:col_offset+r_sub] += V_sub[idx_l]
                    V[idx_g2, col_offset:col_offset+r_sub] += V_sub[idx_l]
            
            # C: M6 contributes to C00 (+1)
            # C00: rows 0..p2-1, cols 0..n2-1
            for k in range(p2):
                for i in range(n2):
                    idx_l = k * n2 + i
                    idx_g1 = k * n + i
                    W[idx_g1, col_offset:col_offset+r_sub] += W_sub[idx_l]

    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # p is 8, so we split into two 4x4x4
    U_left, V_left, W_left = _recursive_strassen(4, 4, 4)
    U_right, V_right, W_right = _recursive_strassen(4, 4, 4)
    
    rank = U_left.shape[1] + U_right.shape[1]
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Left part: B cols 0-3, C rows 0-3
    # U is same for both
    U[:, :rank//2] = U_left
    U[:, rank//2:] = U_right
    
    # V mapping
    # Local V index j' * 4 + k'
    # Global V index j' * 8 + k' (for left, k' in 0..3)
    for j in range(4):
        for k in range(4):
            idx_l = j * 4 + k
            idx_g = j * 8 + k
            V[idx_g, :rank//2] = V_left[idx_l]
            
            # Right part: B cols 4-7, C rows 4-7
            idx_g_r = j * 8 + (k + 4)
            V[idx_g_r, rank//2:] = V_right[idx_l]
            
    # W mapping
    # Local W index k' * 4 + i'
    # Global W index (4 + k') * 4 + i' for left? No, left is k in 0..3.
    # Global W index k * 4 + i
    for k in range(4):
        for i in range(4):
            idx_l = k * 4 + i
            idx_g = k * 4 + i
            W[idx_g, :rank//2] = W_left[idx_l]
            
            # Right part: C rows 4-7
            idx_g_r = (k + 4) * 4 + i
            W[idx_g_r, rank//2:] = W_right[idx_l]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
