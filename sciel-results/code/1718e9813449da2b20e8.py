def construct():
    n, m, p = 4, 4, 8
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def idx_a(i, j): return i * m + j
    def idx_b(j, k): return j * p + k
    def idx_c(k, i): return k * n + i
    
    strassen_terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [0, 0, 0, 1]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]),
        ([0, 0, 1, -1], [1, 1, 0, 0], [1, 0, 0, 0]),
        ([0, 1, 0, -1], [0, 0, 1, 1], [0, 1, 0, 0]),
    ]
    
    strassen_blocks = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (0,1,1), (1,0,1), (1,1,1)]),
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1)]),
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(1,1,1)]),
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(1,1,1)]),
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(0,0,1)]),
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,1,1)]),
    ]
    
    def generate_444(rank_start, b_col_offset, c_row_offset):
        rank_idx = rank_start
        for a_blocks, b_blocks, c_blocks in strassen_blocks:
            for a_c, b_c, c_c in strassen_terms:
                for local_idx in range(4):
                    coeff = a_c[local_idx]
                    if coeff != 0:
                        for br, bc, sign in a_blocks:
                            r0, c0 = 2*br, 2*bc
                            if local_idx == 0: gr, gc = r0, c0
                            elif local_idx == 1: gr, gc = r0, c0+1
                            elif local_idx == 2: gr, gc = r0+1, c0
                            else: gr, gc = r0+1, c0+1
                            U[idx_a(gr, gc)][rank_idx] += coeff * sign
                
                for local_idx in range(4):
                    coeff = b_c[local_idx]
                    if coeff != 0:
                        for br, bc, sign in b_blocks:
                            r0, c0 = 2*br, 2*bc
                            if local_idx == 0: gr, gc = r0, c0
                            elif local_idx == 1: gr, gc = r0, c0+1
                            elif local_idx == 2: gr, gc = r0+1, c0
                            else: gr, gc = r0+1, c0+1
                            V[idx_b(gr, gc + b_col_offset)][rank_idx] += coeff * sign
                
                for local_idx in range(4):
                    coeff = c_c[local_idx]
                    if coeff != 0:
                        for br, bc, sign in c_blocks:
                            r0, c0 = 2*br, 2*bc
                            if local_idx == 0: gr, gc = r0, c0
                            elif local_idx == 1: gr, gc = r0, c0+1
                            elif local_idx == 2: gr, gc = r0+1, c0
                            else: gr, gc = r0+1, c0+1
                            W[idx_c(gr + c_row_offset, gc)][rank_idx] += coeff * sign
                            
                rank_idx += 1
        return rank_idx
    
    # First 4x4x4
    next_rank = generate_444(0, 0, 0)
    # Second 4x4x4
    next_rank = generate_444(next_rank, 4, 4)
    
    return {"U": U, "V": V, "W": W}
