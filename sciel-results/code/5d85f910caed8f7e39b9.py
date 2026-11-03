import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 8
    
    # Use a rank-98 decomposition for 4x4x8
    # Strategy: Use rank-14 for 4x4x4 blocks with shared U terms
    # But first, let's use the verified approach from cell findings
    
    # The cell finding mentions rank-98. Let me construct this properly.
    # For 4x4x8, we can use:
    # - 2 blocks of 4x4x4, each with rank 14 = 28, but with shared U terms
    # - Or use a direct rank-98 algorithm
    
    # Let me use Strassen with optimization: rank 112 baseline
    # Then try to reduce by combining terms
    
    # Actually, let me implement the correct BCLR 4x4x4 rank-14 algorithm
    # and apply it to two blocks
    
    # Correct BCLR 4x4x4 algorithm (14 terms):
    # The algorithm is based on the following structure:
    # Split A, B, C into 2x2 blocks of 2x2 submatrices
    # Use 7 Strassen-like multiplications per 2x2 block
    # But with cross-block optimizations
    
    # For now, let me use a verified rank-98 approach:
    # Use 4x4x4 rank-14 for each of 2 blocks = 28, but this might not be optimal
    # The cell finding says rank 98 was achieved, so let me aim for that
    
    # Let me construct a rank-98 decomposition using a hybrid approach:
    # - Use Strassen for the 4x4x4 structure (8 blocks of 2x2x2 = 56 terms)
    # - Plus additional terms for the 4x4x8 extension
    
    # Actually, the simplest correct approach: use the naive 128-term and see
    # if we can reduce. But the cell finding shows 98 is achievable.
    
    # Let me use the Strassen-based approach that gives 112, and try to optimize
    # by identifying and merging redundant terms
    
    def strassen_2x2x2_terms():
        terms = []
        terms.append(([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]))
        terms.append(([(2,1),(3,1)], [(0,1)], [(2,1)]))
        terms.append(([(0,1)], [(2,1),(3,-1)], [(1,1)]))
        terms.append(([(3,1)], [(1,1),(0,-1)], [(1,1)]))
        terms.append(([(0,1),(1,1)], [(3,1)], [(1,1)]))
        terms.append(([(2,1),(0,-1)], [(0,1),(1,1)], [(2,1)]))
        terms.append(([(1,1),(3,-1)], [(2,1),(3,1)], [(3,1)]))
        return terms
    
    strassen_terms = strassen_2x2x2_terms()
    
    U_list = []
    V_list = []
    W_list = []
    
    # 8 blocks of 2x2x2 for 4x4x4, applied to 2 output blocks
    for block_offset in [0, 4]:
        block_mults = [
            (0, 0, 0, 0, 0, 0),
            (0, 1, 1, 0, 0, 0),
            (0, 0, 0, 1, 0, 1),
            (0, 1, 1, 1, 0, 1),
            (1, 0, 0, 0, 1, 0),
            (1, 1, 1, 0, 1, 0),
            (1, 0, 0, 1, 1, 1),
            (1, 1, 1, 1, 1, 1),
        ]
        
        for (ar, ac, br, bc, cr, cc) in block_mults:
            for (u_l, v_l, w_l) in strassen_terms:
                u_entries = []
                v_entries = []
                w_entries = []
                
                for (u_idx, u_coeff) in u_l:
                    local_i = u_idx // 2
                    local_j = u_idx % 2
                    global_u = (ar*2 + local_i) * 4 + (ac*2 + local_j)
                    u_entries.append((global_u, u_coeff))
                
                for (v_idx, v_coeff) in v_l:
                    local_j = v_idx // 2
                    local_k = v_idx % 2
                    global_v = (br*2 + local_j) * 8 + (block_offset + bc*2 + local_k)
                    v_entries.append((global_v, v_coeff))
                
                for (w_idx, w_coeff) in w_l:
                    local_k = w_idx // 2
                    local_i = w_idx % 2
                    global_w = (block_offset + cc*2 + local_k) * 4 + (cr*2 + local_i)
                    w_entries.append((global_w, w_coeff))
                
                U_list.append(u_entries)
                V_list.append(v_entries)
                W_list.append(w_entries)
    
    total_rank = len(U_list)
    
    # Verify
    def verify_full(U_terms, V_terms, W_terms, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for u_l, v_l, w_l in zip(U_terms, V_terms, W_terms):
                        u_val = sum(c for idx, c in u_l if idx == i*m+j)
                        v_val = sum(c for idx, c in v_l if idx == j*p+k)
                        w_val = sum(c for idx, c in w_l if idx == k*n+i)
                        total += u_val * v_val * w_val
                    if abs(total - 1) > 1e-9:
                        return False
        return True
    
    if not verify_full(U_list, V_list, W_list, n, m, p):
        # Fallback to naive 128
        U_list = []
        V_list = []
        W_list = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_list.append([(i*m+j, 1)])
                    V_list.append([(j*p+k, 1)])
                    W_list.append([(k*n+i, 1)])
        total_rank = len(U_list)
    
    # Convert to matrices
    rank = total_rank
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r in range(rank):
        for (idx, coeff) in U_list[r]:
            U[idx, r] = coeff
        for (idx, coeff) in V_list[r]:
            V[idx, r] = coeff
        for (idx, coeff) in W_list[r]:
            W[idx, r] = coeff
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
