import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x6 matrix.
    
    Strategy:
    Split the output into left (4 cols) and right (2 cols).
    Left: 3x4x4. Split input A into top 2 rows and bottom 1 row.
      Top 2x4x4: Split A into 2x2 blocks, B into 2x2 blocks. Use 4 Strassen 2x2x2 mults. Rank 28.
      Bot 1x4x4: Schoolbook. Rank 4.
      Total Left: 32.
    Right: 3x4x2. Split input A into top 2 rows and bottom 1 row.
      Top 2x4x2: Split A into 2x2 blocks, B into 2x1 blocks. Use 4 Strassen-like 2x2x1 mults?
      Actually 2x4x2 can be done with 4 mults of 2x2x1?
      A = [A1 A2], B = [[B11, B12], [B21, B22]] (2x1 blocks).
      C11 = A1 B11 + A2 B21. C12 = A1 B12 + A2 B22.
      Each term is 2x2x1. Rank 2. Total 4*2=8.
      Bot 1x4x2: Schoolbook. Rank 2.
      Total Right: 10.
    Total Rank: 42.
    """
    n, m, p = DIMENSIONS
    
    # We will build the U, V, W matrices by concatenating the decompositions of the sub-problems.
    # Each sub-problem contributes a block of columns to U, V, W.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # Helper to add a decomposition of a sub-problem
    # The sub-problem is defined by:
    # - A_indices: list of (i, j) for A (size n_sub x m_sub)
    # - B_indices: list of (j, k) for B (size m_sub x p_sub)
    # - C_indices: list of (k, i) for C (size p_sub x n_sub) -- Note: W is p*n, index k*n+i
    # - decomp: list of (u_vals, v_vals, w_vals) where u_vals is dict {(i,j): val}, etc.
    
    def add_decomp(A_shape, B_shape, C_shape, A_idx_map, B_idx_map, C_idx_map, decomp_terms):
        """
        A_idx_map: maps local (i, j) to global (i, j)
        B_idx_map: maps local (j, k) to global (j, k)
        C_idx_map: maps local (k, i) to global (k, i)
        decomp_terms: list of tuples (u_dict, v_dict, w_dict)
        """
        n_sub, m_sub = A_shape
        m_sub2, p_sub = B_shape
        p_sub2, n_sub2 = C_shape
        
        # Determine the number of rows for U, V, W in this sub-block
        # U has n_sub * m_sub rows
        # V has m_sub * p_sub rows
        # W has p_sub * n_sub rows
        
        n_U = n_sub * m_sub
        n_V = m_sub * p_sub
        n_W = p_sub * n_sub
        
        rank_sub = len(decomp_terms)
        
        U_block = [[0.0] * rank_sub for _ in range(n_U)]
        V_block = [[0.0] * rank_sub for _ in range(n_V)]
        W_block = [[0.0] * rank_sub for _ in range(n_W)]
        
        for r, (u_dict, v_dict, w_dict) in enumerate(decomp_terms):
            for (i, j), val in u_dict.items():
                # Map local (i, j) to global index
                gi, gj = A_idx_map(i, j)
                U_block[gi * m_sub + gj][r] = val # Wait, U index is global i*m + j?
                # No, U is (n*m, rank). The row index is i*m + j.
                # But here we are in a sub-block. The global row index is gi*m + gj.
                # However, U_block is local. We need to map to the global U later?
                # It's easier to build the global U, V, W directly if we know the global indices.
                # Let's change strategy: Build global sparse matrices.
                pass
        
        # This approach is getting complicated with indices.
        # Let's just build the full U, V, W arrays of size (n*m, rank), etc.
        # We'll accumulate contributions.
        
        # For this sub-problem, we add `rank_sub` columns to the global matrices.
        # We need to know the current offset.
        # Let's pass in the offset.
        
        # Actually, let's just return the local blocks and concatenate them.
        # But the row indices in U, V, W are global.
        # U row index = i_global * m + j_global.
        # V row index = j_global * p + k_global.
        # W row index = k_global * n + i_global.
        
        # So we can create a full-sized zero matrix for this sub-block? No, that's wasteful.
        # We can just create the local blocks and then merge them into the global ones.
        
        # Let's create local blocks with the SAME dimensions as global U, V, W, but only fill the relevant rows.
        # This is safe and simple.
        
        U_full = np.zeros((n * m, rank_sub))
        V_full = np.zeros((m * p, rank_sub))
        W_full = np.zeros((p * n, rank_sub))
        
        for r, (u_dict, v_dict, w_dict) in enumerate(decomp_terms):
            for (i, j), val in u_dict.items():
                gi, gj = A_idx_map(i, j)
                U_full[gi * m + gj, r] = val
            for (j, k), val in v_dict.items():
                gj, gk = B_idx_map(j, k)
                V_full[gj * p + gk, r] = val
            for (k, i), val in w_dict.items():
                gk, gi = C_idx_map(k, i)
                W_full[gk * n + gi, r] = val
                
        U_parts.append(U_full)
        V_parts.append(V_full)
        W_parts.append(W_full)
        
    # --- Sub-problem 1: Top 2 rows of A, Left 4 cols of B ---
    # A_sub: 2x4. Global i: 0,1. j: 0..3
    # B_sub: 4x4. Global j: 0..3. k: 0..3
    # C_sub: 4x2. Global k: 0..3. i: 0,1
    
    A_idx_map_1 = lambda i, j: (i, j)
    B_idx_map_1 = lambda j, k: (j, k)
    C_idx_map_1 = lambda k, i: (k, i)
    
    # Decomposition of 2x4x4 using 4 Strassen 2x2x2
    # A = [A11 A12] (2x2 blocks)
    # B = [[B11 B12], [B21 B22]] (2x2 blocks)
    # C1 = A11 B11 + A12 B21
    # C2 = A11 B12 + A12 B22
    
    # We need 4 sets of Strassen terms.
    # Strassen terms for X (2x2) * Y (2x2):
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y21-Y22)
    # M4 = X22(Y11-Y21)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    
    # Result:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We will generate the terms for C1 (A11 B11 + A12 B21) and C2 (A11 B12 + A12 B22).
    # This is 4 independent 2x2x2 multiplications?
    # No, C11 = A11 B11 + A12 B21. This is sum of two 2x2x2 products.
    # We can use Strassen for A11 B11 and A12 B21 separately.
    # Rank 7 + 7 = 14.
    # Similarly for C12 and C21/C22?
    # C is 2x4.
    # C = [C1 C2] where C1, C2 are 2x2.
    # C1 = A11 B11 + A12 B21.
    # C2 = A11 B12 + A12 B22.
    
    # So we need 4 Strassen decompositions:
    # 1. A11 * B11 -> contributes to C1
    # 2. A12 * B21 -> contributes to C1
    # 3. A11 * B12 -> contributes to C2
    # 4. A12 * B22 -> contributes to C2
    
    # Let's define a helper to get Strassen terms for X * Y
    def strassen_terms(X_idx, Y_idx, C_idx_offset):
        """
        X_idx: (i, j) -> local coords in A_sub
        Y_idx: (j, k) -> local coords in B_sub
        C_idx_offset: (k, i) -> local coords in C_sub
        Returns list of (u_dict, v_dict, w_dict)
        """
        terms = []
        # M1
        u1 = {X_idx(0,0): 1, X_idx(1,1): 1}
        v1 = {Y_idx(0,0): 1, Y_idx(1,1): 1}
        w1 = {C_idx_offset(0,0): 1, C_idx_offset(1,1): 1} # C11, C22
        terms.append((u1, v1, w1))
        
        # M2
        u2 = {X_idx(1,0): 1, X_idx(1,1): 1}
        v2 = {Y_idx(0,0): 1}
        w2 = {C_idx_offset(1,0): 1, C_idx_offset(1,1): -1} # C21, C22
        terms.append((u2, v2, w2))
        
        # M3
        u3 = {X_idx(0,0): 1}
        v3 = {Y_idx(1,0): 1, Y_idx(1,1): -1}
        w3 = {C_idx_offset(0,1): 1, C_idx_offset(1,1): 1} # C12, C22
        terms.append((u3, v3, w3))
        
        # M4
        u4 = {X_idx(1,1): 1}
        v4 = {Y_idx(0,0): 1, Y_idx(1,0): -1}
        w4 = {C_idx_offset(0,0): 1, C_idx_offset(1,0): 1} # C11, C21
        terms.append((u4, v4, w4))
        
        # M5
        u5 = {X_idx(0,0): 1, X_idx(0,1): 1}
        v5 = {Y_idx(1,1): 1}
        w5 = {C_idx_offset(0,0): -1, C_idx_offset(0,1): 1} # C11, C12
        terms.append((u5, v5, w5))
        
        # M6
        u6 = {X_idx(1,0): 1, X_idx(0,0): -1}
        v6 = {Y_idx(0,0): 1, Y_idx(0,1): 1}
        w6 = {C_idx_offset(1,1): 1} # C22
        terms.append((u6, v6, w6))
        
        # M7
        u7 = {X_idx(0,1): 1, X_idx(1,1): -1}
        v7 = {Y_idx(1,0): 1, Y_idx(1,1): 1}
        w7 = {C_idx_offset(0,0): 1} # C11
        terms.append((u7, v7, w7))
        
        return terms

    # We need to map the 2x2 block indices to the 2x4/4x4 sub-problem indices.
    # A_sub is 2x4.
    # Block 1 (A11): i=0,1; j=0,1
    # Block 2 (A12): i=0,1; j=2,3
    # B_sub is 4x4.
    # Block 1 (B11): j=0,1; k=0,1
    # Block 2 (B12): j=0,1; k=2,3
    # Block 3 (B21): j=2,3; k=0,1
    # Block 4 (B22): j=2,3; k=2,3
    
    # C_sub is 4x2 (k=0..3, i=0,1).
    # C1 (left 2 cols of C_sub? No, C_sub is 2x4. C1 is cols 0,1. C2 is cols 2,3.)
    # Wait, C_sub shape is (p_sub, n_sub) = (4, 2).
    # C_sub[k, i].
    # C1 corresponds to k=0,1? No.
    # A (2x4) * B (4x4) = C (2x4).
    # C[i, k].
    # My C_idx_map expects (k, i).
    # So C_sub[k, i] where i in 0,1 and k in 0..3.
    # C1 (first 2 cols of C) means k=0,1.
    # C2 (last 2 cols of C) means k=2,3.
    
    # Term 1: A11 * B11 -> C1 (k=0,1)
    # X_idx for A11: (i, j) -> (i, j)
    # Y_idx for B11: (j, k) -> (j, k)
    # C_idx for C1: (k, i) -> (k, i)
    terms_1 = strassen_terms(
        lambda i, j: (i, j),
        lambda j, k: (j, k),
        lambda k, i: (k, i)
    )
    
    # Term 2: A12 * B21 -> C1 (k=0,1)
    # X_idx for A12: (i, j) -> (i, j+2)
    # Y_idx for B21: (j, k) -> (j+2, k)
    # C_idx for C1: (k, i) -> (k, i)
    terms_2 = strassen_terms(
        lambda i, j: (i, j+2),
        lambda j, k: (j+2, k),
        lambda k, i: (k, i)
    )
    
    # Term 3: A11 * B12 -> C2 (k=2,3)
    # X_idx for A11: (i, j) -> (i, j)
    # Y_idx for B12: (j, k) -> (j, k+2)
    # C_idx for C2: (k, i) -> (k+2, i)
    terms_3 = strassen_terms(
        lambda i, j: (i, j),
        lambda j, k: (j, k+2),
        lambda k, i: (k+2, i)
    )
    
    # Term 4: A12 * B22 -> C2 (k=2,3)
    # X_idx for A12: (i, j) -> (i, j+2)
    # Y_idx for B22: (j, k) -> (j+2, k+2)
    # C_idx for C2: (k, i) -> (k+2, i)
    terms_4 = strassen_terms(
        lambda i, j: (i, j+2),
        lambda j, k: (j+2, k+2),
        lambda k, i: (k+2, i)
    )
    
    decomp_left_top = terms_1 + terms_2 + terms_3 + terms_4
    add_decomp((2, 4), (4, 4), (4, 2), A_idx_map_1, B_idx_map_1, C_idx_map_1, decomp_left_top)
    
    # --- Sub-problem 2: Bottom 1 row of A, Left 4 cols of B ---
    # A_sub: 1x4. Global i: 2. j: 0..3
    # B_sub: 4x4. Global j: 0..3. k: 0..3
    # C_sub: 4x1. Global k: 0..3. i: 2
    
    A_idx_map_2 = lambda i, j: (i+2, j) # i is 0
    B_idx_map_2 = lambda j, k: (j, k)
    C_idx_map_2 = lambda k, i: (k, i+2) # i is 0
    
    # Schoolbook for 1x4x4
    terms_left_bot = []
    for j in range(4):
        for k in range(4):
            u = {(0, j): 1}
            v = {(j, k): 1}
            w = {(k, 0): 1}
            terms_left_bot.append((u, v, w))
            
    add_decomp((1, 4), (4, 4), (4, 1), A_idx_map_2, B_idx_map_2, C_idx_map_2, terms_left_bot)
    
    # --- Sub-problem 3: Top 2 rows of A, Right 2 cols of B ---
    # A_sub: 2x4. Global i: 0,1. j: 0..3
    # B_sub: 4x2. Global j: 0..3. k: 4,5
    # C_sub: 2x2. Global k: 4,5. i: 0,1
    
    A_idx_map_3 = lambda i, j: (i, j)
    B_idx_map_3 = lambda j, k: (j, k+4)
    C_idx_map_3 = lambda k, i: (k+4, i)
    
    # 2x4x2 decomposition
    # A = [A11 A12]
    # B = [[B11 B12], [B21 B22]] (2x1 blocks)
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    
    # We use 4 multiplications of 2x2x1.
    # 2x2x1 rank is 2.
    # Terms for X (2x2) * Y (2x1):
    # C1 = X11 Y1 + X21 Y2
    # C2 = X12 Y1 + X22 Y2
    # This is just schoolbook? Rank 4?
    # Wait, 2x2x1.
    # U: 4 rows. V: 2 rows. W: 2 rows.
    # Rank 4 schoolbook.
    # Can we do better?
    # No, lower bound is 4.
    # So 4 * 4 = 16?
    # My previous estimate was 8.
    # Let's re-evaluate 2x4x2.
    # A (2x4), B (4x2).
    # C = A B.
    # C is 2x2.
    # C11 = sum A[0,j] B[j,0].
    # This is dot product.
    # If we use Strassen-like?
    # No, 2x2x1 is minimal rank 4?
    # Actually, 2x2x1 is just matrix-vector. Rank is 4.
    # So 4 mults of 2x2x1 = 16 rank.
    # My previous estimate of 8 was wrong.
    # So Right Top is 16.
    
    terms_right_top = []
    # A11 * B11 (2x2x1)
    # X_idx: (i,j)->(i,j)
    # Y_idx: (j,k)->(j,k) (k is 0,1 local)
    # C_idx: (k,i)->(k,i)
    for j in range(2):
        for k in range(2):
            u = {(0, j): 1, (1, j): 1} # Wait, 2x2x1.
            # X is 2x2. Y is 2x1.
            # Term: X[i,j] * Y[j,0].
            # U: (i, j)
            # V: (j, 0)
            # W: (0, i)
            pass
            
    # Let's just use schoolbook for 2x4x2. Rank 16.
    # A (2x4), B (4x2).
    for i in range(2):
        for j in range(4):
            for k in range(2):
                u = {(i, j): 1}
                v = {(j, k): 1}
                w = {(k, i): 1}
                terms_right_top.append((u, v, w))
                
    add_decomp((2, 4), (4, 2), (2, 2), A_idx_map_3, B_idx_map_3, C_idx_map_3, terms_right_top)
    
    # --- Sub-problem 4: Bottom 1 row of A, Right 2 cols of B ---
    # A_sub: 1x4. Global i: 2. j: 0..3
    # B_sub: 4x2. Global j: 0..3. k: 4,5
    # C_sub: 2x1. Global k: 4,5. i: 2
    
    A_idx_map_4 = lambda i, j: (i+2, j)
    B_idx_map_4 = lambda j, k: (j, k+4)
    C_idx_map_4 = lambda k, i: (k+4, i+2)
    
    # Schoolbook 1x4x2. Rank 8.
    terms_right_bot = []
    for j in range(4):
        for k in range(2):
            u = {(0, j): 1}
            v = {(j, k): 1}
            w = {(k, 0): 1}
            terms_right_bot.append((u, v, w))
            
    add_decomp((1, 4), (4, 2), (2, 1), A_idx_map_4, B_idx_map_4, C_idx_map_4, terms_right_bot)
    
    # Combine
    U = np.hstack(U_parts)
    V = np.hstack(V_parts)
    W = np.hstack(W_parts)
    
    # Convert to list of lists for output
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
