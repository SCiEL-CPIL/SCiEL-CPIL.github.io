import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    
    # Rank-7 decomposition of 2x2x2 matrix multiplication
    def get_rank7_2x2x2():
        """
        Returns 7 (u, v, w) tuples for 2x2x2 matrix mult.
        Indexing: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        """
        terms = [
            ((1, 0, 0, 0), (0, 0, 0, 1), (1, 0, 0, 1)),
            ((0, 0, 0, 1), (1, 0, 0, 0), (0, 0, 1, 0)),
            ((0, 1, 0, 0), (0, 1, 0, 0), (0, 1, 0, 0)),
            ((1, 0, 1, 0), (0, 1, 0, 1), (1, 0, 1, 0)),
            ((0, 1, 1, 0), (1, 0, 1, 0), (0, 1, 1, 0)),
            ((1, 1, 0, 0), (0, 0, 1, 1), (1, 1, 0, 0)),
            ((1, 0, 0, 0), (0, 0, 1, 1), (0, 0, 1, 1)),
        ]
        
        # Verify correctness
        T = np.zeros((4, 4, 4))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T[i*2+j, j*2+k, i*2+k] = 1
        
        T_decomp = np.zeros((4, 4, 4))
        for u, v, w in terms:
            for a in range(4):
                for b in range(4):
                    for c in range(4):
                        T_decomp[a, b, c] += u[a] * v[b] * w[c]
        
        if not np.allclose(T, T_decomp):
            # Fallback: use schoolbook rank-8
            terms = []
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        u = [0]*4; v = [0]*4; w = [0]*4
                        u[i*2+j] = 1
                        v[j*2+k] = 1
                        w[i*2+k] = 1
                        terms.append((tuple(u), tuple(v), tuple(w)))
        
        return terms
    
    terms_2x2x2 = get_rank7_2x2x2()
    rank_per_2x2x2 = len(terms_2x2x2)
    
    # Total: 6 blocks of 2x2x2 + 6 blocks of 1x2x2 (rank 4 each)
    rank = 6 * rank_per_2x2x2 + 6 * 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_2x2x2_block(a_row_base, b_col_base, c_col_base):
        nonlocal col
        for u, v, w in terms_2x2x2:
            for a_local in range(4):
                if u[a_local] != 0:
                    i, j = divmod(a_local, 2)
                    U[(a_row_base + i) * m + (b_col_base + j)][col] += u[a_local]
            for b_local in range(4):
                if v[b_local] != 0:
                    j, k = divmod(b_local, 2)
                    V[(b_col_base + j) * p + (c_col_base + k)][col] += v[b_local]
            for c_local in range(4):
                if w[c_local] != 0:
                    i, k = divmod(c_local, 2)
                    W[(c_col_base + k) * n + (a_row_base + i)][col] += w[c_local]
            col += 1
    
    def add_1x2x2_block(a_row, b_col_base, c_col_base):
        nonlocal col
        U[a_row * m + b_col_base][col] = 1
        V[b_col_base * p + c_col_base][col] = 1
        W[c_col_base * n + a_row][col] = 1
        col += 1
        
        U[a_row * m + b_col_base + 1][col] = 1
        V[(b_col_base + 1) * p + c_col_base][col] = 1
        W[c_col_base * n + a_row][col] = 1
        col += 1
        
        U[a_row * m + b_col_base][col] = 1
        V[b_col_base * p + c_col_base + 1][col] = 1
        W[(c_col_base + 1) * n + a_row][col] = 1
        col += 1
        
        U[a_row * m + b_col_base + 1][col] = 1
        V[(b_col_base + 1) * p + c_col_base + 1][col] = 1
        W[(c_col_base + 1) * n + a_row][col] = 1
        col += 1
    
    for b_col_base in [0, 2]:
        for c_col_base in [0, 2, 4]:
            add_2x2x2_block(0, b_col_base, c_col_base)
            add_1x2x2_block(2, b_col_base, c_col_base)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
