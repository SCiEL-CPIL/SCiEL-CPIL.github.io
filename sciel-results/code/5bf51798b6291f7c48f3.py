"""
Matrix multiplication 3x4 * 4x4 tensor decomposition.
Strategy: 
1. Structural decomposition: Split the 3x4*4x4 problem into sub-blocks.
   - Top-left 2x2 * 2x2: Use Strassen (rank 7)
   - Top-right 2x2 * 2x2: Use Strassen (rank 7)
   - Bottom-left 1x2 * 2x2: Schoolbook (rank 4)
   - Bottom-right 1x2 * 2x2: Schoolbook (rank 4)
   - Top 2x4 * 4x2 (strips): Schoolbook (rank 16)
   - Bottom 1x4 * 4x2 (strips): Schoolbook (rank 8)
   Total naive rank: 7+7+4+4+16+8 = 46.
2. Algebraic Dependency Elimination: 
   - Represent the decomposition as a set of vectors u_r, v_r, w_r.
   - A term r is redundant if there exist coefficients c_s (s != r) such that:
     u_r = sum c_s u_s
     v_r = sum c_s v_s
     w_r = sum c_s w_s
   - This is a system of linear equations. We solve it exactly (using rational arithmetic or high-precision floats with verification).
   - If a solution exists, the term is removed.
3. Repeat until no more terms can be eliminated.
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def construct():
    """
    Construct a tensor decomposition for 3x4 * 4x4 matrix multiplication.
    Returns a dictionary with keys 'U', 'V', 'W'.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin for 120s limit
    
    def _strassen_2x2x2():
        """
        Returns U, V, W matrices for 2x2 * 2x2 Strassen decomposition.
        Shapes: (4, 7) each.
        Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen formulas:
        # M1 = (A00+A11)*(B00+B11)
        # M2 = (A10+A11)*B00
        # M3 = A00*(B10-B11)
        # M4 = A11*(B01-B10)
        # M5 = (A00+A01)*B11
        # M6 = (A10-A00)*(B00+B01)
        # M7 = (A01-A11)*(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Term 1: M1
        # A part: A00, A11 -> indices (0,0), (1,1)
        # B part: B00, B11 -> indices (0,0), (1,1)
        # C part: C00 (+1), C11 (+1) -> indices (0,0), (1,1)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # Term 2: M2
        # A part: A10, A11 -> indices (1,0), (1,1)
        # B part: B00 -> index (0,0)
        # C part: C10 (+1) -> index (1,0)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # Term 3: M3
        # A part: A00 -> index (0,0)
        # B part: B10, -B11 -> indices (1,0), (1,1)
        # C part: C01 (+1), C11 (+1) -> indices (0,1), (1,1)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # Term 4: M4
        # A part: A11 -> index (1,1)
        # B part: B01, -B10 -> indices (0,1), (1,0)
        # C part: C00 (+1), C10 (+1) -> indices (0,0), (1,0)
        U[3, 3] = 1
        V[1, 3] = 1; V[2, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # Term 5: M5
        # A part: A00, A01 -> indices (0,0), (0,1)
        # B part: B11 -> index (1,1)
        # C part: C00 (-1), C01 (+1) -> indices (0,0), (0,1)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # Term 6: M6
        # A part: A10, -A00 -> indices (1,0), (0,0)
        # B part: B00, B01 -> indices (0,0), (0,1)
        # C part: C11 (+1) -> index (1,1)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # Term 7: M7
        # A part: A01, -A11 -> indices (0,1), (1,1)
        # B part: B10, B11 -> indices (1,0), (1,1)
        # C part: C00 (-1) -> index (0,0)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1
        
        return U, V, W

    def _schoolbook_block(rows_a, cols_a, rows_b, cols_b):
        """
        Schoolbook decomposition for a block of size rows_a x cols_a * cols_a x cols_b.
        Returns U (rows_a*cols_a, rank), V (cols_a*cols_b, rank), W (cols_b*rows_a, rank).
        """
        rank = rows_a * cols_a * cols_b
        U = np.zeros((rows_a * cols_a, rank))
        V = np.zeros((cols_a * cols_b, rank))
        W = np.zeros((cols_b * rows_a, rank))
        
        col = 0
        for i in range(rows_a):
            for j in range(cols_a):
                for k in range(cols_b):
                    U[i * cols_a + j, col] = 1.0
                    V[j * cols_b + k, col] = 1.0
                    W[k * rows_a + i, col] = 1.0
                    col += 1
        return U, V, W

    # 1. Build the structural decomposition
    # Dimensions: A (3x4), B (4x4), C (3x4)
    # Split A into:
    #   A_top (2x4), A_bot (1x4)
    # Split B into:
    #   B_left (4x2), B_right (4x2)
    
    # Sub-problems:
    # 1. A_top_left (2x2) * B_left_top (2x2) -> C_top_left (2x2)
    #    A indices: rows 0-1, cols 0-1. B indices: rows 0-1, cols 0-1. C indices: rows 0-1, cols 0-1.
    #    Use Strassen.
    
    # 2. A_top_right (2x2) * B_right_top (2x2) -> C_top_right (2x2)
    #    A indices: rows 0-1, cols 2-3. B indices: rows 2-3, cols 0-1? No, B is 4x4.
    #    Let's be precise.
    #    A is 3x4. B is 4x4. C is 3x4.
    #    C[i, k] = sum_j A[i, j] B[j, k]
    
    # Let's define the blocks based on the indices:
    # A: i in {0,1,2}, j in {0,1,2,3}
    # B: j in {0,1,2,3}, k in {0,1,2,3}
    # C: i in {0,1,2}, k in {0,1,2,3}
    
    # We can split the sum over j into two parts: j in {0,1} and j in {2,3}.
    # And split i into {0,1} and {2}.
    # And split k into {0,1} and {2,3}.
    
    # This creates 2 (i-blocks) * 2 (j-blocks) * 2 (k-blocks) = 8 sub-problems.
    # However, the j-blocks are the "inner" dimension.
    # C_top_left (i in 0,1; k in 0,1) gets contributions from:
    #   - A_top (i in 0,1; j in 0,1) * B_top_left (j in 0,1; k in 0,1) -> 2x2 * 2x2
    #   - A_top (i in 0,1; j in 2,3) * B_bot_left (j in 2,3; k in 0,1) -> 2x2 * 2x2
    # C_top_right (i in 0,1; k in 2,3) gets contributions from:
    #   - A_top (i in 0,1; j in 0,1) * B_top_right (j in 0,1; k in 2,3) -> 2x2 * 2x2
    #   - A_top (i in 0,1; j in 2,3) * B_bot_right (j in 2,3; k in 2,3) -> 2x2 * 2x2
    # C_bot_left (i in 2; k in 0,1) gets contributions from:
    #   - A_bot (i in 2; j in 0,1) * B_top_left (j in 0,1; k in 0,1) -> 1x2 * 2x2
    #   - A_bot (i in 2; j in 2,3) * B_bot_left (j in 2,3; k in 0,1) -> 1x2 * 2x2
    # C_bot_right (i in 2; k in 2,3) gets contributions from:
    #   - A_bot (i in 2; j in 0,1) * B_top_right (j in 0,1; k in 2,3) -> 1x2 * 2x2
    #   - A_bot (i in 2; j in 2,3) * B_bot_right (j in 2,3; k in 2,3) -> 1x2 * 2x2
    
    # Total 8 sub-problems.
    # 4 are 2x2*2x2 (Strassen, rank 7 each) -> 28
    # 4 are 1x2*2x2 (Schoolbook, rank 4 each) -> 16
    # Total rank: 44.
    
    U_blocks = []
    V_blocks = []
    W_blocks = []
    
    # Helper to extract and place a block
    def add_block(U_sub, V_sub, W_sub, 
                  i_offset, j_offset, k_offset,
                  n_sub, m_sub, p_sub):
        """
        Add a sub-decomposition to the global matrices.
        U_sub: (n_sub*m_sub, rank_sub)
        V_sub: (m_sub*p_sub, rank_sub)
        W_sub: (p_sub*n_sub, rank_sub)
        """
        rank_sub = U_sub.shape[1]
        
        # Resize global matrices if needed (we know total rank is 44, so we can pre-allocate)
        # For simplicity, let's just append to lists and concatenate at the end.
        U_blocks.append((U_sub, i_offset, j_offset))
        V_blocks.append((V_sub, j_offset, k_offset))
        W_blocks.append((W_sub, k_offset, i_offset))
    
    # Pre-allocate full size matrices to avoid list concatenation overhead
    total_rank = 44
    U_full = np.zeros((n * m, total_rank))
    V_full = np.zeros((m * p, total_rank))
    W_full = np.zeros((p * n, total_rank))
    
    col_offset = 0
    
    # 1. C_top_left part 1: A[0:2, 0:2] * B[0:2, 0:2]
    U_s, V_s, W_s = _strassen_2x2x2()
    # Map indices
    # A: i in 0,1; j in 0,1. Global A index: i*4 + j
    # B: j in 0,1; k in 0,1. Global B index: j*4 + k
    # C: k in 0,1; i in 0,1. Global C index: k*3 + i
    
    for r in range(7):
        for i in range(2):
            for j in range(2):
                val = U_s[i*2+j, r]
                if val != 0:
                    U_full[i*4+j, col_offset+r] += val
        for j in range(2):
            for k in range(2):
                val = V_s[j*2+k, r]
                if val != 0:
                    V_full[j*4+k, col_offset+r] += val
        for k in range(2):
            for i in range(2):
                val = W_s[k*2+i, r]
                if val != 0:
                    W_full[k*3+i, col_offset+r] += val
    col_offset += 7
    
    # 2. C_top_left part 2: A[0:2, 2:4] * B[2:4, 0:2]
    # A: i in 0,1; j in 2,3. B: j in 2,3; k in 0,1. C: k in 0,1; i in 0,1.
    U_s, V_s, W_s = _strassen_2x2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                val = U_s[i*2+j, r]
                if val != 0:
                    U_full[i*4+(j+2), col_offset+r] += val
        for j in range(2):
            for k in range(2):
                val = V_s[j*2+k, r]
                if val != 0:
                    V_full[(j+2)*4+k, col_offset+r] += val
        for k in range(2):
            for i in range(2):
                val = W_s[k*2+i, r]
                if val != 0:
                    W_full[k*3+i, col_offset+r] += val
    col_offset += 7
    
    # 3. C_top_right part 1: A[0:2, 0:2] * B[0:2, 2:4]
    # A: i in 0,1; j in 0,1. B: j in 0,1; k in 2,3. C: k in 2,3; i in 0,1.
    U_s, V_s, W_s = _strassen_2x2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                val = U_s[i*2+j, r]
                if val != 0:
                    U_full[i*4+j, col_offset+r] += val
        for j in range(2):
            for k in range(2):
                val = V_s[j*2+k, r]
                if val != 0:
                    V_full[j*4+(k+2), col_offset+r] += val
        for k in range(2):
            for i in range(2):
                val = W_s[k*2+i, r]
                if val != 0:
                    W_full[(k+2)*3+i, col_offset+r] += val
    col_offset += 7
    
    # 4. C_top_right part 2: A[0:2, 2:4] * B[2:4, 2:4]
    # A: i in 0,1; j in 2,3. B: j in 2,3; k in 2,3. C: k in 2,3; i in 0,1.
    U_s, V_s, W_s = _strassen_2x2x2()
    for r in range(7):
        for i in range(2):
            for j in range(2):
                val = U_s[i*2+j, r]
                if val != 0:
                    U_full[i*4+(j+2), col_offset+r] += val
        for j in range(2):
            for k in range(2):
                val = V_s[j*2+k, r]
                if val != 0:
                    V_full[(j+2)*4+(k+2), col_offset+r] += val
        for k in range(2):
            for i in range(2):
                val = W_s[k*2+i, r]
                if val != 0:
                    W_full[(k+2)*3+i, col_offset+r] += val
    col_offset += 7
    
    # 5. C_bot_left part 1: A[2:3, 0:2] * B[0:2, 0:2]
    # A: i in 2; j in 0,1. B: j in 0,1; k in 0,1. C: k in 0,1; i in 2.
    # Schoolbook 1x2 * 2x2
    U_sb, V_sb, W_sb = _schoolbook_block(1, 2, 2, 2)
    for r in range(4):
        for i in range(1):
            for j in range(2):
                val = U_sb[i*2+j, r]
                if val != 0:
                    U_full[(i+2)*4+j, col_offset+r] += val
        for j in range(2):
            for k in range(2):
                val = V_sb[j*2+k, r]
                if val != 0:
                    V_full[j*4+k, col_offset+r] += val
        for k in range(2):
            for i in range(1):
                val = W_sb[k*1+i, r]
                if val != 0:
                    W_full[k*3+(i+2), col_offset+r] += val
    col_offset += 4
    
    # 6. C_bot_left part 2: A[2:3, 2:4] * B[2:4, 0:2]
    # A: i in 2; j in 2,3. B: j in 2,3; k in 0,1. C: k in 0,1; i in 2.
    U_sb, V_sb, W_sb = _schoolbook_block(1, 2, 2, 2)
    for r in range(4):
        for i in range(1):
            for j in range(2):
                val = U_sb[i*2+j, r]
                if val != 0:
                    U_full[(i+2)*4+(j+2), col_offset+r] += val
        for j in range(2):
            for k in range(2):
                val = V_sb[j*2+k, r]
                if val != 0:
                    V_full[(j+2)*4+k, col_offset+r] += val
        for k in range(2):
            for i in range(1):
                val = W_sb[k*1+i, r]
                if val != 0:
                    W_full[k*3+(i+2), col_offset+r] += val
    col_offset += 4
    
    # 7. C_bot_right part 1: A[2:3, 0:2] * B[0:2, 2:4]
    # A: i in 2; j in 0,1. B: j in 0,1; k in 2,3. C: k in 2,3; i in 2.
    U_sb, V_sb, W_sb = _schoolbook_block(1, 2, 2, 2)
    for r in range(4):
        for i in range(1):
            for j in range(2):
                val = U_sb[i*2+j, r]
                if val != 0:
                    U_full[(i+2)*4+j, col_offset+r] += val
        for j in range(2):
            for k in range(2):
                val = V_sb[j*2+k, r]
                if val != 0:
                    V_full[j*4+(k+2), col_offset+r] += val
        for k in range(2):
            for i in range(1):
                val = W_sb[k*1+i, r]
                if val != 0:
                    W_full[(k+2)*3+(i+2), col_offset+r] += val
    col_offset += 4
    
    # 8. C_bot_right part 2: A[2:3, 2:4] * B[2:4, 2:4]
    # A: i in 2; j in 2,3. B: j in 2,3; k in 2,3. C: k in 2,3; i in 2.
    U_sb, V_sb, W_sb = _schoolbook_block(1, 2, 2, 2)
    for r in range(4):
        for i in range(1):
            for j in range(2):
                val = U_sb[i*2+j, r]
                if val != 0:
                    U_full[(i+2)*4+(j+2), col_offset+r] += val
        for j in range(2):
            for k in range(2):
                val = V_sb[j*2+k, r]
                if val != 0:
                    V_full[(j+2)*4+(k+2), col_offset+r] += val
        for k in range(2):
            for i in range(1):
                val = W_sb[k*1+i, r]
                if val != 0:
                    W_full[(k+2)*3+(i+2), col_offset+r] += val
    col_offset += 4
    
    # 2. Algebraic Dependency Elimination
    # We have U_full, V_full, W_full of shape (dim, 44).
    # We want to find columns r such that:
    # U[:, r] = U[:, S] @ c_U
    # V[:, r] = V[:, S] @ c_V
    # W[:, r] = W[:, S] @ c_W
    # where S is the set of other columns.
    
    def _verify_decomp(U, V, W, tol=1e-9):
        rank = U.shape[1]
        # Check tensor identity
        # T[i,j,k,l] = sum_r U[i,j,r] V[j,k,r] W[k,l,r]
        # Should be 1 if i==l and j==k, else 0.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(p):
                        expected = 1.0 if (i == l and j == k) else 0.0
                        actual = 0.0
                        for r in range(rank):
                            actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r]
                        if abs(actual - expected) > tol:
                            return False
        return True

    # Initial verification
    if not _verify_decomp(U_full, V_full, W_full):
        # Fallback to schoolbook if structural construction failed
        U_full, V_full, W_full = np.zeros((n*m, 48)), np.zeros((m*p, 48)), np.zeros((p*n, 48))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_full[i*m+j, col] = 1.0
                    V_full[j*p+k, col] = 1.0
                    W_full[k*n+i, col] = 1.0
                    col += 1
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    current_rank = U_full.shape[1]
    active_cols = list(range(current_rank))
    
    # Iterative elimination
    eliminated_any = True
    while eliminated_any and time.time() < deadline:
        eliminated_any = False
        for r in list(active_cols):
            if time.time() > deadline:
                break
            
            # Check if column r is redundant
            # Formulate the system:
            # [U_other | V_other | W_other] @ c = [U_r | V_r | W_r]
            
            other_cols = [c for c in active_cols if c != r]
            if not other_cols:
                continue
            
            U_other = U_full[:, other_cols]
            V_other = V_full[:, other_cols]
            W_other = W_full[:, other_cols]
            
            U_target = U_full[:, r]
            V_target = V_full[:, r]
            W_target = W_full[:, r]
            
            # Construct the big matrix
            # Dimensions: (n*m + m*p + p*n) x len(other_cols)
            A_mat = np.vstack([U_other, V_other, W_other])
            b_vec = np.concatenate([U_target, V_target, W_target])
            
            try:
                # Solve least squares
                coeffs, residuals, rank, s = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                
                # Check if the solution is exact
                # Reconstruct
                recon = A_mat @ coeffs
                err = np.linalg.norm(recon - b_vec)
                
                # Also check if the residual is small
                # If the system is consistent, the residual should be 0.
                # We use a tolerance.
                if err < 1e-9:
                    # Verify that removing r still yields a valid decomposition
                    new_U = np.delete(U_full, r, axis=1)
                    new_V = np.delete(V_full, r, axis=1)
                    new_W = np.delete(W_full, r, axis=1)
                    
                    if _verify_decomp(new_U, new_V, new_W):
                        U_full, V_full, W_full = new_U, new_V, new_W
                        active_cols.remove(r)
                        eliminated_any = True
            except Exception:
                continue
    
    # Convert to lists
    def _to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                # Snap to integers or half-integers if close
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-10:
                    clean_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-10:
                    clean_row.append(nearest_half)
                else:
                    clean_row.append(float(val))
            result.append(clean_row)
        return result

    return {"U": _to_clean_list(U_full), "V": _to_clean_list(V_full), "W": _to_clean_list(W_full)}
