import numpy as np
import time
from scipy.optimize import least_squares
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Target: 3x4 * 4x4 -> 3x4
    # Strategy:
    # Split A (3x4) into A_top (2x4) and A_bot (1x4).
    # Split B (4x4) into B_left (4x2) and B_right (4x2).
    # C_top (2x4) = A_top * B.
    # C_bot (1x4) = A_bot * B.
    
    # C_top is 2x4.
    # C_top_left (2x2) = A_top_left (2x2) * B_top_left (2x2) + A_top_right (2x2) * B_bot_left (2x2).
    # This is a 2x4x2 multiplication: X(2x2), Y(2x2) * U(2x2), V(2x2) -> Z(2x2).
    # C_top_right (2x2) = A_top_left (2x2) * B_top_right (2x2) + A_top_right (2x2) * B_bot_right (2x2).
    # This is also a 2x4x2 multiplication.
    
    # C_bot is 1x4.
    # C_bot_left (1x2) = A_bot_left (1x2) * B_top_left (2x2) + A_bot_right (1x2) * B_bot_left (2x2).
    # This is a 1x4x2 multiplication. Rank 8.
    # C_bot_right (1x2) = A_bot_left (1x2) * B_top_right (2x2) + A_bot_right (1x2) * B_bot_right (2x2).
    # This is also a 1x4x2 multiplication. Rank 8.
    
    # Total rank = 2 * R(2x4x2) + 2 * R(1x4x2).
    # R(1x4x2) = 8.
    # We search for R(2x4x2).
    
    def find_2x4x2_decomposition(max_rank=14):
        """
        Finds a decomposition for 2x4 * 4x2 -> 2x2.
        Returns U, V, W matrices or None.
        """
        # Indices:
        # A: 2x4. a = 4*r + c. r in 0,1. c in 0..3.
        # B: 4x2. b = 2*r + c. r in 0..3. c in 0,1.
        # C: 2x2. c_idx = 2*r + c. r in 0,1. c in 0,1.
        
        non_zero_terms = []
        for r_C in range(2):
            for c_C in range(2):
                c_idx = 2 * r_C + c_C
                for r_B in range(4):
                    a_idx = 4 * r_C + r_B
                    b_idx = 2 * r_B + c_C
                    non_zero_terms.append((a_idx, b_idx, c_idx))
                    
        # non_zero_terms has 16 entries.
        
        best_U, best_V, best_W = None, None, None
        best_rank = max_rank + 1
        
        for rank in range(9, max_rank + 1):
            if time.time() > deadline:
                break
                
            # Random initialization
            np.random.seed(rank) # Reproducibility
            U = np.random.rand(8, rank)
            V = np.random.rand(8, rank)
            W = np.random.rand(4, rank)
            
            # Residual function
            def residuals(params):
                U_flat = params[:8*rank]
                V_flat = params[8*rank:16*rank]
                W_flat = params[16*rank:20*rank]
                
                U_mat = U_flat.reshape(8, rank)
                V_mat = V_flat.reshape(8, rank)
                W_mat = W_flat.reshape(4, rank)
                
                res = []
                # Check non-zero terms
                for a, b, c in non_zero_terms:
                    val = np.sum(U_mat[a, :] * V_mat[b, :] * W_mat[c, :])
                    res.append(val - 1.0)
                    
                # Check zero terms
                # To speed up, we can skip some, but 240 is small.
                all_indices = [(a, b, c) for a in range(8) for b in range(8) for c in range(4)]
                nz_set = set(non_zero_terms)
                for a, b, c in all_indices:
                    if (a, b, c) not in nz_set:
                        val = np.sum(U_mat[a, :] * V_mat[b, :] * W_mat[c, :])
                        res.append(val)
                        
                return res
            
            try:
                sol = least_squares(residuals, params, method='lm', max_nfev=1000)
                if sol.success or sol.cost < 1e-10:
                    U_sol = sol.x[:8*rank].reshape(8, rank)
                    V_sol = sol.x[8*rank:16*rank].reshape(8, rank)
                    W_sol = sol.x[16*rank:20*rank].reshape(4, rank)
                    
                    # Verify exactness
                    err = np.max(np.abs(residuals(sol.x)))
                    if err < 1e-8:
                        best_U, best_V, best_W = U_sol, V_sol, W_sol
                        best_rank = rank
                        break
            except Exception:
                continue
                
        return best_U, best_V, best_W, best_rank

    # Search for 2x4x2 decomposition
    U_242, V_242, W_242, rank_242 = find_2x4x2_decomposition()
    
    if rank_242 > 14:
        # Fallback to Strassen (Rank 7 + 7 = 14)
        rank_242 = 14
        # Construct Strassen decomposition for 2x4x2?
        # No, we can just use two Strassen blocks.
        # But for simplicity, if search fails, we use the original rank 44 logic.
        # Let's just return the original rank 44 if search fails.
        return _construct_original_rank44()
        
    # Construct full decomposition
    # Total rank = 2 * rank_242 + 2 * 8
    total_rank = 2 * rank_242 + 16
    
    U = np.zeros((12, total_rank))
    V = np.zeros((16, total_rank))
    W = np.zeros((12, total_rank))
    
    # Helper to add a block
    def add_block(U_block, V_block, W_block, a_offset, b_offset, c_offset, rank_offset):
        """
        U_block: 2x4 or 1x4 or 2x2? 
        Actually, U_242 is 8xR (flattened 2x4).
        V_242 is 8xR (flattened 4x2).
        W_242 is 4xR (flattened 2x2).
        
        For 1x4x2:
        We need a rank-8 decomposition.
        U_142: 4x8 (flattened 1x4).
        V_142: 8x8 (flattened 4x2).
        W_142: 2x8 (flattened 1x2).
        """
        pass
        
    # We need to map the 2x4x2 decomposition to the specific blocks.
    # Block 1: C_top_left.
    # A_top_left (2x2) is A[0:2, 0:2]. Indices 0,1,4,5.
    # A_top_right (2x2) is A[0:2, 2:4]. Indices 2,3,6,7.
    # B_top_left (2x2) is B[0:2, 0:2]. Indices 0,1,4,5.
    # B_bot_left (2x2) is B[2:4, 0:2]. Indices 8,9,12,13.
    # C_top_left (2x2) is C[0:2, 0:2]. Indices 0,1,3,4.
    
    # The 2x4x2 decomposition assumes:
    # A is 2x4. Indices 0..7.
    # B is 4x2. Indices 0..7.
    # C is 2x2. Indices 0..3.
    
    # For Block 1:
    # A_part 1 (X): A[0:2, 0:2]. In 2x4 A, this is cols 0,1.
    # A_part 2 (Y): A[0:2, 2:4]. In 2x4 A, this is cols 2,3.
    # So the 2x4 A in the decomposition corresponds to A_top (2x4).
    # B_part 1 (U): B[0:2, 0:2]. In 4x2 B, this is rows 0,1.
    # B_part 2 (V): B[2:4, 0:2]. In 4x2 B, this is rows 2,3.
    # So the 4x2 B in the decomposition corresponds to B_left (4x2).
    # C_part (Z): C[0:2, 0:2].
    
    # So we can just use U_242, V_242, W_242 directly for the top-left block.
    # U_242 rows 0..7 correspond to A_top indices 0..7.
    # V_242 rows 0..7 correspond to B_left indices 0..7.
    # W_242 rows 0..3 correspond to C_top_left indices 0..3.
    
    # Offset for Block 1:
    a_off = 0
    b_off = 0
    c_off = 0
    r_off = 0
    
    for i in range(8):
        U[a_off+i, r_off:r_off+rank_242] = U_242[i, :]
    for i in range(8):
        V[b_off+i, r_off:r_off+rank_242] = V_242[i, :]
    for i in range(4):
        W[c_off+i, r_off:r_off+rank_242] = W_242[i, :]
        
    # Block 2: C_top_right.
    # A_top (2x4) is the same A.
    # B_right (4x2) is B[0:4, 2:4]. Indices 2,3,6,7,10,11,14,15.
    # In the 4x2 B decomposition, rows 0..3 correspond to B_right rows 0..3.
    # C_top_right (2x2) is C[0:2, 2:4]. Indices 6,7,9,10.
    
    a_off = 0
    b_off = 2 # B_right starts at col 2 of B, so indices 2,3... 
    # Wait, B is 4x4. B_right is B[:, 2:4].
    # Flattened B indices:
    # Row 0: 0,1,2,3. B_right Row 0: 2,3.
    # Row 1: 4,5,6,7. B_right Row 1: 6,7.
    # Row 2: 8,9,10,11. B_right Row 2: 10,11.
    # Row 3: 12,13,14,15. B_right Row 3: 14,15.
    # The 4x2 B in the decomposition has indices 0..7.
    # We need to map decomposition B indices to global B indices.
    # Decomp B index 0 -> Global 2
    # Decomp B index 1 -> Global 3
    # Decomp B index 2 -> Global 6
    # Decomp B index 3 -> Global 7
    # Decomp B index 4 -> Global 10
    # Decomp B index 5 -> Global 11
    # Decomp B index 6 -> Global 14
    # Decomp B index 7 -> Global 15
    
    b_map = [2, 3, 6, 7, 10, 11, 14, 15]
    
    c_off = 6 # C_top_right starts at col 2. C indices:
    # C is 3x4.
    # Row 0: 0,1,2,3. C_top_right Row 0: 2,3.
    # Row 1: 4,5,6,7. C_top_right Row 1: 6,7.
    # So C indices are 2,3,6,7.
    # Decomp C indices 0..3 map to 2,3,6,7.
    c_map = [2, 3, 6, 7]
    
    r_off = rank_242
    
    for i in range(8):
        U[a_off+i, r_off:r_off+rank_242] = U_242[i, :]
    for i in range(8):
        V[b_map[i], r_off:r_off+rank_242] = V_242[i, :]
    for i in range(4):
        W[c_map[i], r_off:r_off+rank_242] = W_242[i, :]
        
    # Block 3: C_bot_left.
    # 1x4x2 multiplication. Rank 8.
    # A_bot_left (1x2) is A[2, 0:2]. Indices 8,9.
    # A_bot_right (1x2) is A[2, 2:4]. Indices 10,11.
    # B_top_left (2x2) is B[0:2, 0:2]. Indices 0,1,4,5.
    # B_bot_left (2x2) is B[2:4, 0:2]. Indices 8,9,12,13.
    # C_bot_left (1x2) is C[2, 0:2]. Indices 8,9.
    
    # Rank 8 decomposition for 1x4x2.
    # U: 4x8. V: 8x8. W: 2x8.
    # We can just use schoolbook.
    # U[i, j] = 1 if i == j? No.
    # Term r corresponds to A_idx, B_idx.
    # A indices: 0,1,2,3 (local).
    # B indices: 0,1,2,3 (local 4x2 flattened? No, 4x2 has 8 indices).
    # Let's use the explicit schoolbook terms.
    
    r_off = 2 * rank_242
    
    # Schoolbook 1x4 * 4x2 -> 1x2
    # C0 = A0 B00 + A1 B10 + A2 B20 + A3 B30
    # C1 = A0 B01 + A1 B11 + A2 B21 + A3 B31
    
    # Local A indices 0..3 map to Global A: 8,9,10,11
    a_map_bot = [8, 9, 10, 11]
    
    # Local B indices 0..7 map to Global B:
    # B_top_left (0..1, 0..1): 0,1,4,5
    # B_bot_left (2..3, 0..1): 8,9,12,13
    # Local B 0 -> Global 0
    # Local B 1 -> Global 1
    # Local B 2 -> Global 4
    # Local B 3 -> Global 5
    # Local B 4 -> Global 8
    # Local B 5 -> Global 9
    # Local B 6 -> Global 12
    # Local B 7 -> Global 13
    b_map_bot = [0, 1, 4, 5, 8, 9, 12, 13]
    
    # Local C indices 0..1 map to Global C: 8,9
    c_map_bot = [8, 9]
    
    for k in range(4):
        # C0 term
        U[a_map_bot[k], r_off + k] = 1
        V[b_map_bot[2*k], r_off + k] = 1
        W[c_map_bot[0], r_off + k] = 1
        
        # C1 term
        U[a_map_bot[k], r_off + 4 + k] = 1
        V[b_map_bot[2*k + 1], r_off + 4 + k] = 1
        W[c_map_bot[1], r_off + 4 + k] = 1
        
    # Block 4: C_bot_right.
    r_off = 2 * rank_242 + 8
    
    a_map_bot = [8, 9, 10, 11]
    
    # B_top_right (0..1, 2..3): 2,3,6,7
    # B_bot_right (2..3, 2..3): 10,11,14,15
    b_map_bot = [2, 3, 6, 7, 10, 11, 14, 15]
    
    c_map_bot = [10, 11]
    
    for k in range(4):
        U[a_map_bot[k], r_off + k] = 1
        V[b_map_bot[2*k], r_off + k] = 1
        W[c_map_bot[0], r_off + k] = 1
        
        U[a_map_bot[k], r_off + 4 + k] = 1
        V[b_map_bot[2*k + 1], r_off + 4 + k] = 1
        W[c_map_bot[1], r_off + 4 + k] = 1

    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _construct_original_rank44():
    # Fallback
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    cols = []
    def add_term(u_entries, v_entries, w_entries):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for idx, val in u_entries: u_vec[idx] = val
        for idx, val in v_entries: v_vec[idx] = val
        for idx, val in w_entries: w_vec[idx] = val
        cols.append((u_vec, v_vec, w_vec))
        
    def add_strassen(a,b,c,d, e,f,g,h, w11,w12,w21,w22):
        add_term([(a,1),(d,1)], [(e,1),(h,1)], [(w11,1),(w22,1)])
        add_term([(c,1),(d,1)], [(e,1)], [(w21,1),(w22,-1)])
        add_term([(a,1)], [(f,1),(h,-1)], [(w12,1),(w22,1)])
        add_term([(d,1)], [(g,1),(e,-1)], [(w11,1),(w21,1)])
        add_term([(a,1),(b,1)], [(h,1)], [(w12,1),(w11,-1)])
        add_term([(c,1),(a,-1)], [(e,1),(f,1)], [(w22,1)])
        add_term([(b,1),(d,-1)], [(g,1),(h,1)], [(w11,1)])
        
    def add_rank4(a,b, c,d,e,f, w1,w2):
        add_term([(a,1)], [(c,1)], [(w1,1)])
        add_term([(b,1)], [(e,1)], [(w1,1)])
        add_term([(a,1)], [(d,1)], [(w2,1)])
        add_term([(b,1)], [(f,1)], [(w2,1)])
        
    add_strassen(0, 1, 4, 5, 0, 1, 4, 5, 0, 3, 1, 4)
    add_strassen(0, 1, 4, 5, 2, 3, 6, 7, 6, 9, 7, 10)
    add_rank4(8, 9, 0, 1, 4, 5, 2, 5)
    add_rank4(8, 9, 2, 3, 6, 7, 8, 11)
    add_strassen(2, 3, 6, 7, 8, 9, 12, 13, 0, 3, 1, 4)
    add_strassen(2, 3, 6, 7, 10, 11, 14, 15, 6, 9, 7, 10)
    add_rank4(10, 11, 8, 9, 12, 13, 2, 5)
    add_rank4(10, 11, 10, 11, 14, 15, 8, 11)
    
    for r in range(rank):
        u_vec, v_vec, w_vec = cols[r]
        for i in range(n * m): U[i][r] = u_vec[i]
        for j in range(m * p): V[j][r] = v_vec[j]
        for k in range(p * n): W[k][r] = w_vec[k]
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
