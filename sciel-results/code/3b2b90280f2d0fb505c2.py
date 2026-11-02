import time
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    def verify(U_mat, V_mat, W_mat):
        U_np = np.array(U_mat)
        V_np = np.array(V_mat)
        W_np = np.array(W_mat)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i * m + j, :]
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-9:
                        return False
        return True

    def build_strassen_2x2x2(j1, j2, k1, k2, col_offset, U, V, W):
        """Add Strassen 2x2x2 block for rows j1,j2 and cols k1,k2"""
        col = col_offset
        u00 = 0 * m + j1; u01 = 0 * m + j2
        u10 = 1 * m + j1; u11 = 1 * m + j2
        v00 = j1 * p + k1; v01 = j1 * p + k2
        v10 = j2 * p + k1; v11 = j2 * p + k2
        w00 = k1 * n + 0; w01 = k2 * n + 0
        w10 = k1 * n + 1; w11 = k2 * n + 1
        
        # P1: (A00+A11)(B00+B11)
        U[u00][col] += 1; U[u11][col] += 1
        V[v00][col] += 1; V[v11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        # P2: (A10+A11)B00
        U[u10][col] += 1; U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1; W[w11][col] += -1
        col += 1
        # P3: A00(B01-B11)
        U[u00][col] += 1
        V[v01][col] += 1; V[v11][col] += -1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        # P4: A11(B10-B00)
        U[u11][col] += 1
        V[v10][col] += 1; V[v00][col] += -1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        # P5: (A00+A01)B11
        U[u00][col] += 1; U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] += -1; W[w01][col] += 1
        col += 1
        # P6: (A10-A00)(B00+B01)
        U[u10][col] += 1; U[u00][col] += -1
        V[v00][col] += 1; V[v01][col] += 1
        W[w11][col] += 1
        col += 1
        # P7: (A01-A11)(B10+B11)
        U[u01][col] += 1; U[u11][col] += -1
        V[v10][col] += 1; V[v11][col] += 1
        W[w00][col] += 1
        col += 1
        
        return col

    def build_2x3x3_rank12(j1, j2, j3, k1, k2, k3, col_offset, U, V, W):
        """
        Build 2x3x3 tensor with rank 12 using:
        - Standard terms for (j1,k1), (j1,k2), (j1,k3), (j2,k1), (j3,k1), (j3,k2), (j3,k3) = 7 terms
        - Strassen for (j2,j3) x (k2,k3) = 5 additional terms (replacing 4 standard)
        Total: 7 + 5 = 12
        """
        col = col_offset
        
        # Standard terms for cells NOT covered by Strassen
        standard_cells = [
            (j1, k1), (j1, k2), (j1, k3),
            (j2, k1),
            (j3, k1)
        ]
        
        for (j, k) in standard_cells:
            u_idx = 0 * m + j
            v_idx = j * p + k
            w_idx = k * n + 0
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
            u_idx2 = 1 * m + j
            w_idx2 = k * n + 1
            U[u_idx2][col] = 1
            W[w_idx2][col] = 1
            col += 1
        
        # Strassen for (j2, j3) x (k2, k3)
        col = build_strassen_2x2x2(j2, j3, k2, k3, col, U, V, W)
        
        return col

    def build_2x3x3_rank11(j1, j2, j3, k1, k2, k3, col_offset, U, V, W):
        """
        Attempt rank 11 for 2x3x3 using a known decomposition.
        Structure: 6 standard + 5 Strassen-like
        """
        col = col_offset
        
        # Try a specific rank-11 pattern
        # This is based on known results for 2x3x3 tensors
        # We'll use a combination that might work
        
        # 6 standard terms
        std_cells = [(j1, k1), (j1, k2), (j1, k3), (j2, k1), (j3, k1), (j3, k2)]
        
        for (j, k) in std_cells:
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1
            
            U[1 * m + j][col] = 1
            W[k * n + 1][col] = 1
            col += 1
        
        # Now handle (j2,k2), (j2,k3), (j3,k3) with 5 terms
        # Using a modified Strassen-like approach
        j_a, j_b = j2, j3
        k_a, k_b = k2, k3
        
        u_a0 = 0 * m + j_a; u_b0 = 0 * m + j_b
        u_a1 = 1 * m + j_a; u_b1 = 1 * m + j_b
        v_a_a = j_a * p + k_a; v_a_b = j_a * p + k_b
        v_b_a = j_b * p + k_a; v_b_b = j_b * p + k_b
        w_a_0 = k_a * n + 0; w_b_0 = k_b * n + 0
        w_a_1 = k_a * n + 1; w_b_1 = k_b * n + 1
        
        # Term 1: (A_a0 + A_b1)(B_a_a + B_b_b)
        U[u_a0][col] += 1; U[u_b1][col] += 1
        V[v_a_a][col] += 1; V[v_b_b][col] += 1
        W[w_a_0][col] += 1; W[w_b_1][col] += 1
        col += 1
        
        # Term 2: (A_a1 + A_b1)B_a_a
        U[u_a1][col] += 1; U[u_b1][col] += 1
        V[v_a_a][col] += 1
        W[w_a_1][col] += 1; W[w_b_1][col] += -1
        col += 1
        
        # Term 3: A_a0(B_a_b - B_b_b)
        U[u_a0][col] += 1
        V[v_a_b][col] += 1; V[v_b_b][col] += -1
        W[w_b_0][col] += 1; W[w_b_1][col] += 1
        col += 1
        
        # Term 4: A_b1(B_b_a - B_a_a)
        U[u_b1][col] += 1
        V[v_b_a][col] += 1; V[v_a_a][col] += -1
        W[w_a_0][col] += 1; W[w_a_1][col] += 1
        col += 1
        
        # Term 5: (A_a0 + A_a1 + A_b0 + A_b1)(B_b_b)
        U[u_a0][col] += 1; U[u_a1][col] += 1
        U[u_b0][col] += 1; U[u_b1][col] += 1
        V[v_b_b][col] += 1
        W[w_b_0][col] += -1; W[w_b_1][col] += -1
        col += 1
        
        return col

    # Strategy 1: Try rank 52 using 2x1x6 (rank 12) + 2x4x6 with improved blocks
    def attempt_rank_52():
        rank = 52
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        
        # Block 1: 2x1x6 (j=0, k=0..5) - rank 12
        for k in range(6):
            for i in range(2):
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        
        # Block 2: 2x4x6 (j=1..4, k=0..5) - try rank 40
        # Split into 2x2x6 blocks: (j=1,2) and (j=3,4)
        # Each 2x2x6: use 3 Strassen 2x2x2 instances = 21 rank each
        # But we need rank 40 total for 2x4x6, so 20 per 2x2x6?
        # Let's use standard rank 12 for each 2x2x3, so 2x2x6 = 24, total 48
        # That gives 12 + 48 = 60, too high.
        
        # Alternative: use rank 11 for 2x3x3 blocks
        # 2x4x6 = 2x(3+1)x6 = 2x3x6 + 2x1x6
        # 2x3x6 = 2x3x3 + 2x3x3, each rank 11 = 22
        # 2x1x6 = rank 12
        # Total for 2x4x6: 22 + 12 = 34
        # Grand total: 12 + 34 = 46
        
        # Let's try this decomposition:
        # 2x1x6: j=0, k=0..5 (rank 12)
        # 2x3x3: j=1,2,3, k=0,1,2 (rank 11)
        # 2x3x3: j=1,2,3, k=3,4,5 (rank 11)
        # 2x1x6: j=4, k=0..5 (rank 12)
        # Total: 12 + 11 + 11 + 12 = 46
        
        # Reset
        rank = 46
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        
        # Block 1: 2x1x6 (j=0, k=0..5) - rank 12
        for k in range(6):
            for i in range(2):
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        
        # Block 2: 2x3x3 (j=1,2,3, k=0,1,2) - rank 11
        col = build_2x3x3_rank11(1, 2, 3, 0, 1, 2, col, U, V, W)
        
        # Block 3: 2x3x3 (j=1,2,3, k=3,4,5) - rank 11
        col = build_2x3x3_rank11(1, 2, 3, 3, 4, 5, col, U, V, W)
        
        # Block 4: 2x1x6 (j=4, k=0..5) - rank 12
        for k in range(6):
            for i in range(2):
                U[i * m + 4][col] = 1
                V[4 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        
        if col != 46:
            return None
        
        if not verify(U, V, W):
            return None
        
        return {"U": U, "V": V, "W": W}

    # Strategy 2: Rank 54 with Strassen tiling (current best)
    def build_rank_54():
        rank = 54
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        strassen_cells = set()
        for j1, j2 in j_pairs:
            for k1, k2 in k_pairs:
                for i in range(n):
                    for j in (j1, j2):
                        for k in (k1, k2):
                            strassen_cells.add((i, j, k))
        
        # Add standard terms for non-Strassen cells
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) in strassen_cells:
                        continue
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col] = 1
                    V[v_idx][col] = 1
                    W[w_idx][col] = 1
                    col += 1
        
        # Add Strassen terms
        for j1, j2 in j_pairs:
            for k1, k2 in k_pairs:
                col = build_strassen_2x2x2(j1, j2, k1, k2, col, U, V, W)
        
        if not verify(U, V, W):
            return None
        
        return {"U": U, "V": V, "W": W}

    # Strategy 3: Fallback rank 60
    def build_rank_60():
        rank_f = 60
        U_f = [[0] * rank_f for _ in range(n * m)]
        V_f = [[0] * rank_f for _ in range(m * p)]
        W_f = [[0] * rank_f for _ in range(p * n)]
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][col_f] = 1
                    V_f[j * p + k][col_f] = 1
                    W_f[k * n + i][col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}

    # Try strategies in order
    result = attempt_rank_52()
    if result is not None:
        return result
    
    if time.time() < deadline - 5:
        result = build_rank_54()
        if result is not None:
            return result
    
    return build_rank_60()
