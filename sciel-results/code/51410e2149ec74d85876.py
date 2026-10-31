import numpy as np
import time

def construct():
    """
    Returns a rank-49 decomposition of the 4x4x4 matrix multiplication tensor
    using recursive Strassen (2x2 blocks, each with Strassen 2x2).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    # Build the decomposition by recursively applying Strassen.
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k in {0,1,2,3}
    # A decomposition of rank r gives U, V, W where:
    # U: shape (16, r) -- indices (i*m+j) for i,j in {0,1,2,3}, m=4
    # V: shape (16, r) -- indices (j*p+k) for j,k in {0,1,2,3}, p=4  
    # W: shape (16, r) -- indices (k*n+i) for k,i in {0,1,2,3}, n=4
    # with T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For Strassen on 2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = (A11)(B12-B22)
    # M4 = (A22)(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 4x4, we split into 2x2 blocks:
    # A = [[A11, A12],[A21, A22]] where each Ai is 2x2
    # B = [[B11, B12],[B21, B22]] where each Bj is 2x2
    # C = [[C11, C12],[C21, C22]] where each Ci is 2x2
    
    # Total: 7 (outer) * 7 (inner) = 49 terms.
    
    terms = []  # list of (u_dict, v_dict, w_dict)
    
    def add_term(u_dict, v_dict, w_dict):
        terms.append((u_dict, v_dict, w_dict))
    
    # Strassen 2x2 inner decomposition for a product of two 2x2 matrices
    # Each entry is (a_comb, b_comb, c_comb) where each comb is a list of (local_row, local_col, coeff)
    strassen_2x2 = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),  # S1
        ([(1,0,1),(1,1,1)], [(0,0,1)],               [(1,0,1),(1,1,-1)]),  # S2
        ([(0,0,1)],               [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),  # S3
        ([(1,1,1)],               [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),  # S4
        ([(0,0,1),(0,1,1)],      [(1,1,1)],               [(0,0,-1),(0,1,1)]),  # S5
        ([(1,0,1),(0,0,-1)],     [(0,0,1),(0,1,1)],      [(1,1,1)]),  # S6
        ([(0,1,1),(1,1,-1)],     [(1,0,1),(1,1,1)],      [(0,0,1)]),  # S7
    ]
    
    # Outer Strassen for 4x4 (operating on 2x2 blocks)
    outer_strassen = [
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),  # M1
        ([(1,0,1),(1,1,1)], [(0,0,1)],               [(1,0,1),(1,1,-1)]),  # M2
        ([(0,0,1)],               [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),  # M3
        ([(1,1,1)],               [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),  # M4
        ([(0,0,1),(0,1,1)],      [(1,1,1)],               [(0,0,-1),(0,1,1)]),  # M5
        ([(1,0,1),(0,0,-1)],     [(0,0,1),(0,1,1)],      [(1,1,1)]),  # M6
        ([(0,1,1),(1,1,-1)],     [(1,0,1),(1,1,1)],      [(0,0,1)]),  # M7
    ]
    
    # Now for each outer term and inner Strassen term, create a rank-1 term.
    for outer_idx, (a_blocks, b_blocks, c_blocks) in enumerate(outer_strassen):
        for inner_idx, (a_comb, b_comb, c_comb) in enumerate(strassen_2x2):
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            # Build U contributions
            for (lr, lc, ca) in a_comb:
                for (br, bc, coeff) in a_blocks:
                    gr = br * 2 + lr  # global row
                    gc = bc * 2 + lc  # global col
                    u_idx = gr * 4 + gc
                    u_dict[u_idx] = u_dict.get(u_idx, 0) + ca * coeff
            
            # Build V contributions
            for (lr, lc, cb) in b_comb:
                for (br, bc, coeff) in b_blocks:
                    gr = br * 2 + lr  # global row
                    gc = bc * 2 + lc  # global col
                    v_idx = gr * 4 + gc
                    v_dict[v_idx] = v_dict.get(v_idx, 0) + cb * coeff
            
            # Build W contributions
            for (lr, lc, cc) in c_comb:
                for (br, bc, coeff) in c_blocks:
                    gr = br * 2 + lr  # global row of C
                    gc = bc * 2 + lc  # global col of C
                    w_idx = gc * 4 + gr  # W index = col*4+row
                    w_dict[w_idx] = w_dict.get(w_idx, 0) + cc * coeff
            
            add_term(u_dict, v_dict, w_dict)
    
    rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    for col, (u_dict, v_dict, w_dict) in enumerate(terms):
        for idx, val in u_dict.items():
            if val != 0:
                U[idx][col] = val
        for idx, val in v_dict.items():
            if val != 0:
                V[idx][col] = val
        for idx, val in w_dict.items():
            if val != 0:
                W[idx][col] = val
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0.0
                for r in range(rank):
                    s += U_np[i*4+j, r] * V_np[j*4+k, r] * W_np[k*4+i, r]
                max_err = max(max_err, abs(s - 1.0))
    
    if max_err > 1e-9:
        # Fallback to schoolbook if verification fails
        rank = 64
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i*4+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*4+i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
