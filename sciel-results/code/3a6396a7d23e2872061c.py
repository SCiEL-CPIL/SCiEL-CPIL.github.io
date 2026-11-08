import numpy as np
import time

def construct():
    """
    Construct rank-114 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Strategy:
    - Split B into B1 (4x4), B2 (4x4), B3 (4x1)
    - Use Strassen (rank 49) for A*B1 and A*B2
    - Use Schoolbook (rank 16) for A*B3
    - Total rank: 49 + 49 + 16 = 114
    """
    start_time = time.time()
    DEADLINE = 145  # 5 seconds margin from 150s limit
    
    n, m, p = 4, 4, 9
    U_parts = []
    V_parts = []
    W_parts = []

    # Pre-compute Strassen 2x2 base terms as lists of (index, coeff)
    # Indices are flattened (row, col) for 2x2 matrices
    s2x2_base = [
        # (u_indices, v_indices, w_indices)
        ([ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ], [ (0,0,1), (1,1,1) ]),
        ([ (1,0,1), (1,1,1) ], [ (0,0,1) ], [ (1,0,1), (1,1,-1) ]),
        ([ (0,0,1) ], [ (0,1,1), (1,1,-1) ], [ (0,1,1), (1,1,1) ]),
        ([ (1,1,1) ], [ (1,0,1), (0,0,-1) ], [ (0,0,1), (1,0,1) ]),
        ([ (0,0,1), (0,1,1) ], [ (1,1,1) ], [ (0,0,-1), (0,1,1) ]),
        ([ (1,0,1), (0,0,-1) ], [ (0,0,1), (0,1,1) ], [ (1,1,1) ]),
        ([ (0,1,1), (1,1,-1) ], [ (1,0,1), (1,1,1) ], [ (0,0,1) ])
    ]

    # Generate all 49 terms for 4x4 multiplication
    # A is 4x4, B is 4x4, C is 4x4
    # We view them as 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    
    # Mapping of block (br, bc) and inner (ir, ic) to global index
    # A global index: gi = br*2 + ir, gj = bc*2 + ic
    # B global index: bi = br*2 + ir, bj = bc*2 + ic
    # C global index: ci = br*2 + ir, cj = bc*2 + ic
    
    # Pre-compute the 49 terms for the first 4x4 block (cols 0-3 of B, cols 0-3 of C)
    terms_4x4_0 = []
    for meta_a, meta_b, meta_c in s2x2_base:
        for inner_a, inner_b, inner_c in s2x2_base:
            u = [0] * 16
            v = [0] * 36
            w = [0] * 36
            
            # Accumulate coefficients
            for (ir, ic, coeff_a) in meta_a:
                for (i_r, i_c, coeff_inner) in inner_a:
                    gi = ir * 2 + i_r
                    gj = ic * 2 + i_c
                    u[gi * 4 + gj] += coeff_a * coeff_inner
            
            for (ir, ic, coeff_b) in meta_b:
                for (i_r, i_c, coeff_inner) in inner_b:
                    bi = ir * 2 + i_r
                    bj = ic * 2 + i_c
                    v[bi * 9 + bj] += coeff_b * coeff_inner
            
            for (ir, ic, coeff_c) in meta_c:
                for (i_r, i_c, coeff_inner) in inner_c:
                    ci = ir * 2 + i_r
                    cj = ic * 2 + i_c
                    # W index is cj * 4 + ci (p=9, but here we are in 0-3 cols, so p_block=4? 
                    # Wait, W is global. Shape (p*n, rank) = (9*4, rank) = (36, rank).
                    # Index is j*n + i. Here j is global col 0-3, i is global row 0-3.
                    w[cj * 4 + ci] += coeff_c * coeff_inner
            
            terms_4x4_0.append((u, v, w))
            
            if time.time() > DEADLINE:
                break
        if time.time() > DEADLINE:
            break
            
    # We need 49 terms. If we broke early, we might have fewer, but Strassen is small so it shouldn't.
    # However, the loop above generates 7*7=49 terms.
    
    # Add terms for B1 (cols 0-3)
    for u, v, w in terms_4x4_0:
        U_parts.append(u)
        V_parts.append(v)
        W_parts.append(w)
        
    if time.time() > DEADLINE:
        return _finalize(U_parts, V_parts, W_parts)

    # Generate terms for B2 (cols 4-7)
    # We can reuse the structure but shift indices in V and W
    # V index: bi * 9 + (bj + 4)
    # W index: (cj + 4) * 4 + ci
    
    terms_4x4_1 = []
    for u, v, w in terms_4x4_0:
        v_new = v[:]
        w_new = w[:]
        
        # Shift V indices: bj -> bj + 4
        # v is 36 length (4 rows, 9 cols). Index = bi*9 + bj.
        # We need to move non-zeros from cols 0-3 to 4-7.
        # Instead of shifting, just build new V from scratch using the same coeffs but different base col.
        pass
        
    # It is safer and faster to just regenerate or transform.
    # Let's transform existing v and w.
    for u, v, w in terms_4x4_0:
        v_new = [0] * 36
        w_new = [0] * 36
        
        for idx, val in enumerate(v):
            if val != 0:
                bi = idx // 9
                bj = idx % 9
                # bj should be 0-3. Shift to 4-7.
                v_new[bi * 9 + (bj + 4)] = val
                
        for idx, val in enumerate(w):
            if val != 0:
                cj = idx // 4
                ci = idx % 4
                # cj should be 0-3. Shift to 4-7.
                w_new[(cj + 4) * 4 + ci] = val
                
        U_parts.append(u)
        V_parts.append(v_new)
        W_parts.append(w_new)
        
    if time.time() > DEADLINE:
        return _finalize(U_parts, V_parts, W_parts)

    # Add schoolbook terms for B3 (col 8)
    # C_i,8 = sum_k A_i,k B_k,8
    # u[i*4+k] = 1, v[k*9+8] = 1, w[8*4+i] = 1
    for i in range(4):
        for k in range(4):
            u = [0] * 16
            v = [0] * 36
            w = [0] * 36
            u[i * 4 + k] = 1
            v[k * 9 + 8] = 1
            w[8 * 4 + i] = 1
            U_parts.append(u)
            V_parts.append(v)
            W_parts.append(w)
            
            if time.time() > DEADLINE:
                return _finalize(U_parts, V_parts, W_parts)

    return _finalize(U_parts, V_parts, W_parts)

def _finalize(U_parts, V_parts, W_parts):
    """Construct the final matrices from the term lists."""
    if not U_parts:
        return {"U": [], "V": [], "W": []}
    
    # U_parts is list of lists, each of length 16
    # We want U to be (16, rank)
    # V to be (36, rank)
    # W to be (36, rank)
    
    rank = len(U_parts)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
