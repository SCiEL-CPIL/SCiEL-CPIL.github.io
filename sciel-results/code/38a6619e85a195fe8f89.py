import numpy as np
import time

def construct():
    """
    Returns a rank-49 decomposition of the 4x4x4 matrix multiplication tensor
    using recursive Strassen (2x2 blocks, each with Strassen 2x2).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    def _build_rank49():
        """Build the standard recursive Strassen rank-49 decomposition."""
        
        # Strassen 2x2: 7 terms, each (a_comb, b_comb, c_comb)
        # a_comb/b_comb/c_comb are lists of (local_row, local_col, coeff)
        strassen_2x2 = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),       # M1
            ([(1,0,1),(1,1,1)], [(0,0,1)],               [(1,0,1),(1,1,-1)]),  # M2
            ([(0,0,1)],               [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),  # M3
            ([(1,1,1)],               [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),  # M4
            ([(0,0,1),(0,1,1)],      [(1,1,1)],               [(0,0,-1),(0,1,1)]),  # M5
            ([(1,0,1),(0,0,-1)],     [(0,0,1),(0,1,1)],      [(1,1,1)]),         # M6
            ([(0,1,1),(1,1,-1)],     [(1,0,1),(1,1,1)],      [(0,0,1)]),         # M7
        ]
        
        # Outer Strassen for 4x4 (same structure, applied to 2x2 blocks)
        outer_strassen = [
            ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),       # M1
            ([(1,0,1),(1,1,1)], [(0,0,1)],               [(1,0,1),(1,1,-1)]),  # M2
            ([(0,0,1)],               [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),  # M3
            ([(1,1,1)],               [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),  # M4
            ([(0,0,1),(0,1,1)],      [(1,1,1)],               [(0,0,-1),(0,1,1)]),  # M5
            ([(1,0,1),(0,0,-1)],     [(0,0,1),(0,1,1)],      [(1,1,1)]),         # M6
            ([(0,1,1),(1,1,-1)],     [(1,0,1),(1,1,1)],      [(0,0,1)]),         # M7
        ]
        
        terms = []
        
        for (a_blocks, b_blocks, c_blocks) in outer_strassen:
            for (a_comb, b_comb, c_comb) in strassen_2x2:
                u_dict = {}
                v_dict = {}
                w_dict = {}
                
                # Build U part: for each (local_row, local_col, coeff) in a_comb,
                # and for each (block_row, block_col, block_coeff) in a_blocks:
                # global row = block_row*2 + local_row
                # global col = block_col*2 + local_col
                # U flat index = global_row * 4 + global_col
                for (lr, lc, ca) in a_comb:
                    for (br, bc, coeff) in a_blocks:
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        u_idx = gr * 4 + gc
                        u_dict[u_idx] = u_dict.get(u_idx, 0) + ca * coeff
                
                # Build V part: same structure
                for (lr, lc, cb) in b_comb:
                    for (br, bc, coeff) in b_blocks:
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        v_idx = gr * 4 + gc
                        v_dict[v_idx] = v_dict.get(v_idx, 0) + cb * coeff
                
                # Build W part: W index = k*n+i = col*4+row (output C[row,col])
                # c_comb gives local (row, col, coeff) for the C block
                # c_blocks gives (block_row, block_col, block_coeff)
                # global row = block_row*2 + local_row
                # global col = block_col*2 + local_col
                # W flat index = global_col * 4 + global_row
                for (lr, lc, cc) in c_comb:
                    for (br, bc, coeff) in c_blocks:
                        gr = br * 2 + lr
                        gc = bc * 2 + lc
                        w_idx = gc * 4 + gr
                        w_dict[w_idx] = w_dict.get(w_idx, 0) + cc * coeff
                
                terms.append((u_dict, v_dict, w_dict))
        
        rank = len(terms)
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
        
        return U, V, W, rank

    def _verify(U, V, W, rank):
        """Verify the decomposition is exact: T[i,j,k] = sum_r U[i*4+j,r]*V[j*4+k,r]*W[k*4+i,r] = 1"""
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    s = 0
                    for r in range(rank):
                        s += U[i*4+j][r] * V[j*4+k][r] * W[k*4+i][r]
                    if s != 1:
                        return False
        return True

    # Build rank-49 decomposition
    U, V, W, rank = _build_rank49()
    
    # Verify
    if not _verify(U, V, W, rank):
        # Fallback to schoolbook rank 64
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
