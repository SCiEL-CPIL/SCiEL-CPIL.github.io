import time
import numpy as np

def construct():
    """
    Rank-72 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    
    Split B into [B1 | B2 | B3] where B1, B2 are 4x4 and B3 is 4x1.
    - C1 = A @ B1: Strassen 4x4x4 (rank 28)
    - C2 = A @ B2: Strassen 4x4x4 (rank 28)
    - C3 = A @ B3: schoolbook 4x4x1 (rank 16)
    Total: 72
    
    Indexing:
    U: (16, rank) - A[i*4+j]
    V: (36, rank) - B[j*9+k]
    W: (36, rank) - C[k*4+i]
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 9
    total_rank = 72
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    col = 0
    
    # Strassen 4x4x4 with schoolbook 2x2x2 (rank 4 per sub-multiplication)
    # 7 sub-multiplications * 4 = 28 rank-1 terms
    
    # For each of the 7 Strassen products M_c = S_c * T_c:
    # S_c is a 2x2 matrix (linear combo of A blocks)
    # T_c is a 2x2 matrix (linear combo of B blocks)
    # M_c is 2x2, computed by schoolbook: M_c[r,c] = sum_l S_c[r,l] * T_c[l,c]
    
    # Strassen components:
    # c=0: S=A11+A22, T=B11+B22, C: C11+=M, C22+=M
    # c=1: S=A21+A22, T=B11, C: C21+=M
    # c=2: S=A11, T=B12-B22, C: C12+=M, C22-=M
    # c=3: S=A22, T=B11-B12, C: C11-=M, C22-=M
    # c=4: S=A11+A12, T=B22, C: C12+=M
    # c=5: S=A21-A11, T=B11+B12, C: C21-=M
    # c=6: S=A12-A22, T=B21+B22, C: C11+=M, C22+=M
    
    def add_strassen_4x4x4(b_offset):
        nonlocal col
        
        # A block flat indices (for 4x4 A, flat index = row*4+col)
        # A11: [0,1,4,5], A12: [2,3,6,7], A21: [8,9,12,13], A22: [10,11,14,15]
        # Local 2x2 index: (r,c) -> r*2+c, so indices [0,1,2,3] = [(0,0),(0,1),(1,0),(1,1)]
        
        A_blocks = {
            0: [0, 1, 4, 5],    # A11
            1: [2, 3, 6, 7],    # A12
            2: [8, 9, 12, 13],  # A21
            3: [10, 11, 14, 15] # A22
        }
        
        # B blocks for 4x9 B, with column offset
        def B_block(bid, boff):
            # bid 0: rows 0-1, cols boff+0,1
            # bid 1: rows 0-1, cols boff+2,3
            # bid 2: rows 2-3, cols boff+0,1
            # bid 3: rows 2-3, cols boff+2,3
            row_base = 0 if bid < 2 else 2
            col_base = boff + (0 if bid % 2 == 0 else 2)
            return [(row_base + r) * 9 + (col_base + c) for r in range(2) for c in range(2)]
        
        # C blocks for 4x9 C, with column offset (same as B offset)
        # C[i,k] at W[k*4+i]
        def C_block(bid, coff):
            row_base = 0 if bid < 2 else 2
            col_base = coff + (0 if bid % 2 == 0 else 2)
            return [(col_base + c) * 4 + (row_base + r) for r in range(2) for c in range(2)]
        
        # Strassen components: (S_blocks, T_blocks, C_contribs)
        # S_blocks: [(block_id, sign), ...]
        # T_blocks: [(block_id, sign), ...]
        # C_contribs: [(block_id, sign), ...]
        
        comps = [
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),   # c=0
            ([(2,1),(3,1)], [(0,1)], [(2,1)]),               # c=1
            ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,-1)]),       # c=2
            ([(3,1)], [(0,1),(1,-1)], [(0,-1),(3,-1)]),      # c=3
            ([(0,1),(1,1)], [(3,1)], [(1,1)]),               # c=4
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(2,-1)]),       # c=5
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1),(3,1)]),  # c=6
        ]
        
        for S_blocks, T_blocks, C_contribs in comps:
            # Schoolbook 2x2*2x2: for r,l,c in 0..1
            for r in range(2):
                for l in range(2):
                    for c in range(2):
                        # S[r,l] coefficient
                        s_coeff = {}
                        for bid, sign in S_blocks:
                            idxs = A_blocks[bid]
                            local = r * 2 + l
                            gidx = idxs[local]
                            s_coeff[gidx] = s_coeff.get(gidx, 0) + sign
                        
                        # T[l,c] coefficient
                        t_coeff = {}
                        for bid, sign in T_blocks:
                            idxs = B_block(bid, b_offset)
                            local = l * 2 + c
                            gidx = idxs[local]
                            t_coeff[gidx] = t_coeff.get(gidx, 0) + sign
                        
                        # C contribution: M[r,c] goes to C_contribs at local (r,c)
                        c_coeff = {}
                        for bid, sign in C_contribs:
                            idxs = C_block(bid, b_offset)
                            local = r * 2 + c
                            gidx = idxs[local]
                            c_coeff[gidx] = c_coeff.get(gidx, 0) + sign
                        
                        # Add rank-1 terms
                        for a_idx, a_val in s_coeff.items():
                            if a_val == 0: continue
                            for b_idx, b_val in t_coeff.items():
                                if b_val == 0: continue
                                for c_idx, c_val in c_coeff.items():
                                    if c_val == 0: continue
                                    if col < total_rank:
                                        coeff = a_val * b_val * c_val
                                        U[a_idx][col] = coeff
                                        V[b_idx][col] = 1
                                        W[c_idx][col] = 1
                                        col += 1
    
    add_strassen_4x4x4(0)  # C1 = A @ B1
    add_strassen_4x4x4(4)  # C2 = A @ B2
    
    # Schoolbook for C3 = A @ B3 (B3 is 4x1, column 8)
    for i in range(4):
        for j in range(4):
            if col < total_rank:
                U[i*4 + j][col] = 1
                V[j*9 + 8][col] = 1
                W[8*4 + i][col] = 1
                col += 1
    
    if col < total_rank:
        U = [row[:col] for row in U]
        V = [row[:col] for row in V]
        W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
