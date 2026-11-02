"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 135.0

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Use a few random tests
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for c in range(rank):
            u_val = np.sum(U[:, c].reshape(n, m) * A)
            v_val = np.sum(V[:, c].reshape(m, p) * B)
            W_c = W[:, c].reshape(p, n).T
            C_computed += u_val * v_val * W_c
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _construct_rank_97(n, m, p):
    # n=4, m=4, p=7
    # Split B into B1 (4x4) and B2 (4x3)
    # Use rank 49 for A*B1 (4x4x4) via Strassen (2x2 blocks of 2x2)
    # Use rank 48 for A*B2 (4x4x3) via Schoolbook
    # Total rank 97
    
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u_vec, v_vec, w_vec):
        U_list.append(u_vec)
        V_list.append(v_vec)
        W_list.append(w_vec)
        
    # Base 2x2x2 Strassen decomposition
    # Returns list of 7 tuples (u, v, w) of length 4
    # u, v, w are vectors for A, B, C (flattened 2x2)
    # A index: i*2 + j
    # B index: i*2 + j
    # C index: k*2 + i  (Note: W index is k*n + i, so for n=2, k*2+i)
    
    base_decomp = [
        # M1: (A00+A11)(B00+B11) -> C00, C11
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (A10+A11)B00 -> C10, C11
        ([0, 1, 0, 1], [1, 0, 0, 0], [0, 1, 0, 1]),
        # M3: A00(B01-B11) -> C01, C11
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        # M4: A11(B10-B00) -> C00, C10
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        # M5: (A00+A01)B11 -> C00, C01
        ([1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]),
        # M6: (A10-A00)(B00+B01) -> C11
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: (A01-A11)(B10+B11) -> C00
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Block Strassen definitions for 4x4x4
    # Each entry is (X_def, Y_def, C_contrib)
    # X_def: list of (block_idx, coeff) for A
    # Y_def: list of (block_idx, coeff) for B
    # C_contrib: list of (block_idx, coeff) for C
    # Block indices: 00, 01, 10, 11
    
    block_strassen = [
        # M1 = (A00+A11)(B00+B11)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (A10+A11)B00
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]), # C10, C11(-1)? 
        # Wait, standard Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Let's re-verify the C_contrib for each M.
        # M1 contributes to C00 (+1) and C11 (+1).
        # M2 contributes to C10 (+1) and C11 (-1).
        # M3 contributes to C01 (+1) and C11 (+1).
        # M4 contributes to C00 (+1) and C10 (+1).
        # M5 contributes to C00 (-1) and C01 (+1).
        # M6 contributes to C11 (+1).
        # M7 contributes to C00 (+1).
        
        # So:
        # M1: C00(+1), C11(+1)
        # M2: C10(+1), C11(-1)
        # M3: C01(+1), C11(+1)
        # M4: C00(+1), C10(+1)
        # M5: C00(-1), C01(+1)
        # M6: C11(+1)
        # M7: C00(+1)
        
        # My previous base_decomp assumed specific C contributions.
        # I must ensure the base_decomp matches the C_contrib of the block term.
        # The base_decomp I listed above has FIXED C contributions.
        # This means I can ONLY use it if the block term's C contribution matches the base term's C contribution?
        # NO. The base_decomp computes the FULL 2x2 matrix product.
        # So M1 (block) is a 2x2 matrix.
        # The base_decomp computes that 2x2 matrix exactly.
        # So I don't need to specify C_contrib for the block level?
        # Wait.
        # Block Strassen:
        # M1 = (A00+A11)(B00+B11). This is a 2x2 matrix.
        # This 2x2 matrix is added to C00 and C11.
        # So C00 += M1, C11 += M1.
        # The base_decomp computes the entries of M1.
        # Let M1 = [[m00, m01], [m10, m11]].
        # Then C00[0,0] += m00, C00[0,1] += m01, etc.
        # And C11[0,0] += m00, etc.
        
        # So for each block term r:
        # 1. Determine X_r and Y_r (linear combos of A/B blocks).
        # 2. Use base_decomp to generate 7 terms for the product X_r Y_r.
        # 3. For each of these 7 terms, the local W vector w_local has 4 entries (m00, m10, m01, m11? No, index k*2+i).
        #    w_local[0] is C[0,0] of the product.
        #    w_local[1] is C[1,0] of the product.
        #    w_local[2] is C[0,1] of the product.
        #    w_local[3] is C[1,1] of the product.
        # 4. Map these to global C blocks.
        
        # So I don't need C_contrib in the loop, I just need to know WHICH C blocks get the contribution.
        # M1 -> C00, C11.
        # M2 -> C10, C11.
        # M3 -> C01, C11.
        # M4 -> C00, C10.
        # M5 -> C00, C01.
        # M6 -> C11.
        # M7 -> C00.
        
        # Let's redefine the loop.
    ]
    
    # Correct Block Strassen Definitions
    # (X_blocks, Y_blocks, C_targets)
    # X_blocks: dict {block_idx: coeff}
    # Y_blocks: dict {block_idx: coeff}
    # C_targets: dict {block_idx: coeff}
    
    block_defs = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),       # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),       # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),       # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),       # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),       # M7
    ]
    
    for X_blocks, Y_blocks, C_targets in block_defs:
        # For each of the 7 base terms in the 2x2x2 decomposition
        for u_base, v_base, w_base in base_decomp:
            U_vec = [0] * 16
            V_vec = [0] * 28
            W_vec = [0] * 28
            
            # Process U (A)
            # u_base has 4 entries: indices 0,1,2,3 correspond to local (0,0), (0,1), (1,0), (1,1)
            # Local (i,j) -> Global (r_off+i, c_off+j)
            for l in range(4):
                coeff = u_base[l]
                if coeff == 0: continue
                i_loc = l // 2
                j_loc = l % 2
                # Sum contributions from all A blocks
                for b_idx, a_coeff in X_blocks.items():
                    r_off = (b_idx // 2) * 2
                    c_off = (b_idx % 2) * 2
                    global_i = r_off + i_loc
                    global_j = c_off + j_loc
                    U_vec[global_i * 4 + global_j] += coeff * a_coeff
            
            # Process V (B)
            for l in range(4):
                coeff = v_base[l]
                if coeff == 0: continue
                i_loc = l // 2
                j_loc = l % 2
                for b_idx, b_coeff in Y_blocks.items():
                    r_off = (b_idx // 2) * 2
                    c_off = (b_idx % 2) * 2
                    global_i = r_off + i_loc
                    global_j = c_off + j_loc
                    V_vec[global_i * 7 + global_j] += coeff * b_coeff
            
            # Process W (C)
            # w_base has 4 entries: indices 0,1,2,3 correspond to local C (0,0), (1,0), (0,1), (1,1)
            # Note: w_base index k*2+i.
            # l=0: k=0, i=0 -> C[0,0]
            # l=1: k=0, i=1 -> C[1,0]
            # l=2: k=1, i=0 -> C[0,1]
            # l=3: k=1, i=1 -> C[1,1]
            for l in range(4):
                coeff = w_base[l]
                if coeff == 0: continue
                k_loc = l // 2
                i_loc = l % 2
                for c_idx, c_coeff in C_targets.items():
                    r_off = (c_idx // 2) * 2
                    c_off = (c_idx % 2) * 2
                    global_i = r_off + i_loc
                    global_k = c_off + k_loc
                    # W index is k*n + i
                    W_vec[global_k * 4 + global_i] += coeff * c_coeff
            
            add_term(U_vec, V_vec, W_vec)
            
    # Part 2: 4x4x3 using Schoolbook (Rank 48)
    for i in range(4):
        for j in range(4):
            for k_local in range(3):
                k_global = 4 + k_local
                U_vec = [0] * 16
                V_vec = [0] * 28
                W_vec = [0] * 28
                
                U_vec[i*4 + j] = 1
                V_vec[j*7 + k_global] = 1
                W_vec[k_global*4 + i] = 1
                
                add_term(U_vec, V_vec, W_vec)
                
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    n, m, p = DIMENSIONS
    
    best = _build_schoolbook(n, m, p)
    
    # Try the rank 97 construction
    try:
        candidate = _construct_rank_97(n, m, p)
        # Verify the candidate
        if _verify_decomposition(candidate['U'], candidate['V'], candidate['W'], n, m, p):
            best = candidate
    except Exception as e:
        print(f"Construction failed: {e}")
        
    return best

# EVOLVE-BLOCK-END
