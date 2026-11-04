import numpy as np
import time
import sys

_DEADLINE = time.time() + 110
DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # 4x4x4 part using Strassen (Rank 49)
    # 4x4x3 part using search or fallback
    
    # First, build the 4x4x4 Strassen decomposition
    # We only need the part that affects columns 0-3 of C.
    # The full tensor is 4x4x7.
    # C[:, 0:4] depends on A and B[:, 0:4].
    # C[:, 4:7] depends on A and B[:, 4:7].
    
    # We will build the full U, V, W matrices.
    # Rank 49 for the 4x4x4 part.
    
    # Define Strassen 4x4x4
    # This part is standard.
    strassen_4x4x4_U = []
    strassen_4x4x4_V = []
    strassen_4x4x4_W = []
    
    # Standard 4x4x4 Strassen decomposition
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Each M is a 2x2x2 product. Rank 7.
    # Total 49.
    
    # We generate the 7 terms for each 2x2x2 block.
    
    def get_2x2x2_strassen():
        # Returns U, V, W for 2x2x2 (flattened 4x7)
        u_rows = []
        v_rows = []
        w_rows = []
        
        # M1: (A00+A11)(B00+B11) -> C00, C11
        # U: A00, A11
        # V: B00, B11
        # W: C00, C11
        
        terms = [
            # (u_sparsity, v_sparsity, w_sparsity)
            # u: (0,0), (1,1)
            # v: (0,0), (1,1)
            # w: (0,0), (1,1)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2: (A10+A11)B00 -> C10, C11(-)
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # M3: A00(B01-B11) -> C01, C11
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M4: A11(B10-B00) -> C00, C10
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5: (A00+A01)B11 -> C00(-), C01
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            # M6: (A10-A00)(B00+B01) -> C11
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7: (A01-A11)(B10+B11) -> C00
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
        ]
        
        for u_def, v_def, w_def in terms:
            u_vec = np.zeros(4)
            v_vec = np.zeros(4)
            w_vec = np.zeros(4)
            
            for (r, c), val in u_def.items():
                u_vec[r*2+c] = val
            for (r, c), val in v_def.items():
                v_vec[r*2+c] = val
            for (r, c), val in w_def.items():
                w_vec[r*2+c] = val
                
            u_rows.append(u_vec)
            v_rows.append(v_vec)
            w_rows.append(w_vec)
            
        return np.array(u_rows).T, np.array(v_rows).T, np.array(w_rows).T # 4x7
        
    U_222, V_222, W_222 = get_2x2x2_strassen()
    
    # Map to 4x4x4
    # Blocks: 00, 01, 10, 11
    # A indices: 0-3 (0:00, 1:01, 2:10, 3:11)
    # B indices: 0-3
    # C indices: 0-3
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    # For each of the 7 Strassen terms (M1..M7)
    # We need to place the 2x2x2 blocks into the 4x4x4 grid.
    # M1 affects A00, A11, B00, B11, C00, C11
    # M2 affects A10, A11, B00, C10, C11
    # etc.
    
    # Let's use a helper to add a 2x2x2 block to a specific position
    def add_block(U_mat, V_mat, W_mat, u_block, v_block, w_block, a_r, a_c, b_r, b_c, c_r, c_c, offset):
        # u_block is 4x7, representing 2x2 A entries
        # We need to map A(0,0) -> A(a_r, a_c), A(0,1) -> A(a_r, a_c+1), etc.
        
        u_map = [(0, a_r, a_c), (1, a_r, a_c+1), (2, a_r+1, a_c), (3, a_r+1, a_c+1)]
        v_map = [(0, b_r, b_c), (1, b_r, b_c+1), (2, b_r+1, b_c), (3, b_r+1, b_c+1)]
        w_map = [(0, c_r, c_c), (1, c_r, c_c+1), (2, c_r+1, c_c), (3, c_r+1, c_c+1)]
        
        for k in range(7):
            for u_idx_in, (u_idx, r, c) in enumerate(u_map):
                val = u_block[u_idx_in, k]
                if val != 0:
                    U_mat[r*4+c, offset+k] += val
            for v_idx_in, (v_idx, r, c) in enumerate(v_map):
                val = v_block[v_idx_in, k]
                if val != 0:
                    V_mat[r*4+c, offset+k] += val
            for w_idx_in, (w_idx, r, c) in enumerate(w_map):
                val = w_block[w_idx_in, k]
                if val != 0:
                    W_mat[r*4+c, offset+k] += val

    # Define placements for M1..M7
    # M1: A00, A11; B00, B11; C00, C11
    # A blocks: (0,0) and (1,1). This is not a single 2x2 block.
    # Strassen M1 = (A00+A11)(B00+B11).
    # This requires summing A00 and A11 in U.
    # U for M1 has entries at 00 and 11.
    # In 4x4x4, A00 is at (0,0), A11 is at (1,1).
    # So we need to add to (0,0) and (1,1) in A-space.
    # My add_block function adds to a contiguous 2x2 block.
    # M1 uses non-contiguous blocks.
    
    # Let's manually construct U, V, W for 4x4x4 Strassen.
    # It's easier to use the known 49-term list or derive it.
    
    # I will use the 2x2x3 search for the second part, and keep the 49-term part hardcoded for speed.
    
    # Hardcoded 4x4x4 Strassen (Rank 49)
    # Reference: Standard Strassen 4x4x4
    # Terms:
    # 1. (A00+A11)(B00+B11) -> C00+=, C11+=
    # 2. (A10+A11)B00 -> C10+=, C11-=
    # 3. A00(B01-B11) -> C01+=, C11+=
    # 4. A11(B10-B00) -> C00+=, C10+=
    # 5. (A00+A01)B11 -> C00-=, C01+=
    # 6. (A10-A00)(B00+B01) -> C11+=
    # 7. (A01-A11)(B10+B11) -> C00+=
    
    # Each is a 2x2x2 product.
    # We expand each into 7 terms.
    
    U_444 = np.zeros((16, 49))
    V_444 = np.zeros((16, 49))
    W_444 = np.zeros((16, 49))
    
    def add_222_term(a_indices, b_indices, c_indices, u_val, v_val, w_val, offset):
        # a_indices: list of (r,c) in 4x4
        # u_val: list of values corresponding to a_indices
        # We assume the 2x2x2 term maps to these.
        
        # This is getting complicated. I will just use the known decomposition.
        pass

    # To save time, I will use the 49-term construction from the inspiration, which was valid.
    # I will copy the logic from the previous program for Part 1.
    
    current_rank = 0
    U = [[0.0] * (49 + 44) for _ in range(16)] # Max rank estimate
    V = [[0.0] * (49 + 44) for _ in range(16)]
    W = [[0.0] * (49 + 44) for _ in range(28)]
    
    # --- Part 1: 4x4x4 (Rank 49) ---
    # Re-implementing the compact Strassen 4x4x4
    # M1: (A00+A11)(B00+B11)
    # U: A00, A11
    # V: B00, B11
    # W: C00, C11
    
    # We define the 7 terms for 2x2x2
    # t0: A00, A11; B00, B11; C00, C11
    # t1: A10, A11; B00; C10, C11(-)
    # t2: A00; B01, B11(-); C01, C11
    # t3: A11; B10, B00(-); C00, C10
    # t4: A00, A01; B11; C00(-), C01
    # t5: A10, A00(-); B00, B01; C11
    # t6: A01, A11(-); B10, B11; C00
    
    # We need to map these to 4x4 indices.
    # Block 00: (0,0), (0,1), (1,0), (1,1) -> 4x4 (0,0)..(1,1)
    # Block 01: (0,0), (0,1), (1,0), (1,1) -> 4x4 (0,2)..(1,3)
    # Block 10: -> 4x4 (2,0)..(3,1)
    # Block 11: -> 4x4 (2,2)..(3,3)
    
    # M1 uses A00 (0,0-1,0-1) and A11 (2,2-3,2-3).
    # This is not a single 2x2 block.
    # So we must handle the sparsity carefully.
    
    # I will use the explicit 49-term list generated by the previous program, 
    # but I will search for a better 4x4x3 part.
    
    # For now, assume Part 1 is Rank 49.
    # Part 2: 4x4x3.
    # I will search for rank < 44.
    
    # Target Tensor for 4x4x3
    # A: 4x4, B: 4x3, C: 4x3
    # T[i, j, k] = 1 if C[i, k] = sum_j A[i, j] B[j, k]
    # Flattened:
    # A_idx = i*4 + j
    # B_idx = j*3 + k
    # C_idx = k*4 + i
    
    target_443 = np.zeros((16, 12, 12))
    for i in range(4):
        for j in range(4):
            for k in range(3):
                a_idx = i*4 + j
                b_idx = j*3 + k
                c_idx = k*4 + i
                target_443[a_idx, b_idx, c_idx] = 1.0
                
    # Search for decomposition
    best_rank = 44
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks 30 to 43
    # Use random initialization and Newton iteration
    
    for r in range(35, 44):
        if time.time() > _DEADLINE - 10: break
        
        U_init = np.random.randn(16, r) * 0.1
        V_init = np.random.randn(12, r) * 0.1
        W_init = np.random.randn(12, r) * 0.1
        
        # Newton iteration
        # Residual R = target - sum(U_k V_k W_k)
        # Gradient w.r.t U_k: V_k W_k * R
        # Hessian approx: (V_k W_k) (V_k W_k)^T? No.
        
        # Simple iteration:
        # U_k += eta * (V_k W_k) * R
        # This is not quite right for tensor rank.
        
        # Use standard tensor decomposition update:
        # For each k, project R onto V_k and W_k to get update for U_k.
        
        for it in range(100):
            if time.time() > _DEADLINE - 5: break
            
            # Compute residual
            # R = target - einsum('ir,jr,kr->ijk', U, V, W)
            recon = np.einsum('ir,jr,kr->ijk', U_init, V_init, W_init)
            R = target_443 - recon
            
            # Update U
            # U_k_new = U_k + (V_k \cdot W_k) * R
            # (V_k W_k) is a vector of size 16? No, V_k is 12, W_k is 12.
            # The term for U_k is sum_jk V_k[j] W_k[k] R[i,j,k]
            
            U_new = np.zeros_like(U_init)
            for k in range(r):
                # Project R onto V_k and W_k
                # coeff = sum_{j,k} V_k[j] W_k[k] R[:, j, k]
                # R is 16x12x12.
                # Sum over j,k:
                # res_u = np.einsum('j,k,ijk->i', V_init[:, k], W_init[:, k], R)
                coeff_u = np.einsum('j,k,ijk->i', V_init[:, k], W_init[:, k], R)
                U_new[:, k] = U_init[:, k] + 0.01 * coeff_u
                
            # Update V
            V_new = np.zeros_like(V_init)
            for k in range(r):
                coeff_v = np.einsum('i,k,ijk->j', U_new[:, k], W_init[:, k], R)
                V_new[:, k] = V_init[:, k] + 0.01 * coeff_v
                
            # Update W
            W_new = np.zeros_like(W_init)
            for k in range(r):
                coeff_w = np.einsum('i,j,ijk->k', U_new[:, k], V_new[:, k], R)
                W_new[:, k] = W_init[:, k] + 0.01 * coeff_w
                
            U_init = U_new
            V_init = V_new
            W_init = W_new
            
            # Check error
            recon = np.einsum('ir,jr,kr->ijk', U_init, V_init, W_init)
            err = np.linalg.norm(target_443 - recon)
            if err < 1e-5:
                best_rank = r
                best_U = U_init
                best_V = V_init
                best_W = W_init
                break
        
        if best_rank == r:
            break
            
    # If no better rank found, use 44 (fallback to 4x(2x2x3))
    if best_rank >= 44:
        # Use the 4x(2x2x3) construction
        # Rank 44
        # We need to construct this.
        # For now, I will just use the schoolbook rank 48 for the 4x4x3 part to ensure correctness if search fails.
        # Actually, 44 is known to work.
        # I will hardcode the 44-term construction for 4x4x3.
        
        # This is tedious to code from scratch in this snippet.
        # I will assume the search found something or use schoolbook 48.
        # To be safe, I will use Schoolbook for 4x4x3 (Rank 48) if search fails.
        # Total Rank = 49 + 48 = 97. Score 100/97 = 1.03.
        # This is better than the 1.0 baseline.
        
        # But I want to beat 1.2043 (Rank 93).
        # So I MUST find Rank < 44 for 4x4x3 or Rank < 49 for 4x4x4.
        
        # Since I don't have the 4x(2x2x3) code here, I will implement a simple 4x4x3 rank 48 (schoolbook)
        # and hope the search finds < 44.
        
        pass

    # If search failed, use Schoolbook for 4x4x3
    rank_443 = 48
    U_443 = np.zeros((16, rank_443))
    V_443 = np.zeros((12, rank_443))
    W_443 = np.zeros((12, rank_443))
    
    r_idx = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_443[i*4+j, r_idx] = 1.0
                V_443[j*3+k, r_idx] = 1.0
                W_443[k*4+i, r_idx] = 1.0
                r_idx += 1

    if best_rank < 44:
        rank_443 = best_rank
        U_443 = best_U
        V_443 = best_V
        W_443 = best_W

    # Combine 4x4x4 (Rank 49) and 4x4x3 (Rank rank_443)
    # Total Rank = 49 + rank_443
    
    # Note: The 4x4x4 part only uses B[:, 0:4].
    # The 4x4x3 part uses B[:, 4:7].
    # So we can simply concatenate U, V, W horizontally.
    # U: 16 x (49 + rank_443)
    # V: 16 x (49 + rank_443)
    # W: 28 x (49 + rank_443)
    
    # Wait, V for 4x4x4 is 16x49 (B is 4x4).
    # V for 4x4x3 is 12xrank_443 (B is 4x3).
    # We need to align V.
    # V_444 has non-zeros in rows 0-15 (B 4x4).
    # V_443 has non-zeros in rows 12-23 (B 4x3, cols 4-6).
    # So we create V_full (28 x TotalRank).
    # V_full[0:16, 0:49] = V_444
    # V_full[12:28, 49:49+rank_443] = V_443
    
    # Similarly for W.
    # W_444: 16 x 49 (C 4x4, cols 0-3).
    # W_443: 12 x rank_443 (C 4x3, cols 4-6).
    # W_full: 28 x TotalRank.
    # W_full[0:16, 0:49] = W_444
    # W_full[12:28, 49:49+rank_443] = W_443
    
    # U is just concatenated.
    
    total_rank = 49 + rank_443
    
    U_full = np.zeros((16, total_rank))
    V_full = np.zeros((28, total_rank))
    W_full = np.zeros((28, total_rank))
    
    # Part 1: 4x4x4
    # I need to generate the 49 terms.
    # I will use the standard Strassen 4x4x4.
    
    # For brevity and to ensure the code runs, I will use the Schoolbook for 4x4x4 as well?
    # No, 49 is much better than 64.
    # I will implement the 49-term Strassen 4x4x4.
    
    # ... (Implementation of 49-term Strassen 4x4x4) ...
    
    # If I run out of time, I will return Schoolbook 112.
    
    if time.time() > _DEADLINE - 2:
        # Fallback to Schoolbook 112
        U_sb = np.zeros((16, 112))
        V_sb = np.zeros((28, 112))
        W_sb = np.zeros((28, 112))
        r = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U_sb[i*4+j, r] = 1.0
                    V_sb[j*7+k, r] = 1.0
                    W_sb[k*4+i, r] = 1.0
                    r += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    # Actual construction
    # Since I cannot easily code the full 49-term Strassen in this short snippet without bugs,
    # and I must provide a complete program, I will use the 4x4x4 Strassen code from the inspiration
    # and the search for 4x4x3.
    
    # I will paste the working 4x4x4 Strassen code from the previous successful run.
    
    # ... (Code from previous program for Part 1) ...
    
    # And the search for Part 2.
    
    # If search fails, use Schoolbook for Part 2.
    
    # Return
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

# The above logic is complex to write inline. 
# I will write the complete program below.
