"""Program for rank-minimal tensor decomposition of matrix multiplication 3x4x7."""
DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact using NumPy."""
    import numpy as np
    n, m, p = n, m, p
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    max_diff = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > max_diff:
                    max_diff = abs(s - 1.0)
    return max_diff < 1e-9

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    DEADLINE = start_time + 115  # 120s limit, keep 5s margin

    n, m, p = DIMENSIONS  # 3, 4, 7

    def schoolbook():
        """Standard schoolbook decomposition of rank n*m*p."""
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

    best = schoolbook()

    def verify(U, V, W, n, m, p):
        """Verify exactness: for all (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if s != 1:
                        return False
        return True

    def strassen_222():
        """
        Strassen decomposition for 2x2x2 matrix multiplication.
        Returns (U, V, W) where:
          U: shape (4, 7) - coefficients for A entries
          V: shape (4, 7) - coefficients for B entries
          W: shape (4, 7) - coefficients for C entries
        Indexing: U[i*2+j, c] for A_{i,j}, V[j*2+k, c] for B_{j,k}, W[k*2+i, c] for C_{i,k}
        """
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]

        # Strassen's 7 products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B01-B11)
        # M4 = (A11)(B10-B00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # M1: (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1   # A00, A11
        V[0][0] = 1; V[3][0] = 1   # B00, B11
        W[0][0] = 1                # C00 += M1

        # M2: (A10+A11)(B00)
        U[2][1] = 1; U[3][1] = 1   # A10, A11
        V[0][1] = 1                # B00
        W[1][1] = 1                # C10 += M2

        # M3: A00(B01-B11)
        U[0][2] = 1                # A00
        V[1][2] = 1; V[3][2] = -1  # B01 - B11
        W[2][2] = 1                # C01 += M3

        # M4: A11(B10-B00)
        U[3][3] = 1                # A11
        V[2][3] = 1; V[0][3] = -1  # B10 - B00
        W[0][3] = 1                # C00 += M4
        W[1][3] = 1                # C10 += M4

        # M5: (A00+A01)(B11)
        U[0][4] = 1; U[1][4] = 1   # A00, A01
        V[3][4] = 1                # B11
        W[2][4] = 1                # C01 += M5
        W[0][4] = -1               # C00 -= M5

        # M6: (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1  # A10 - A00
        V[0][5] = 1; V[1][5] = 1   # B00 + B01
        W[3][5] = 1                # C11 += M6

        # M7: (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1  # A01 - A11
        V[2][6] = 1; V[3][6] = 1   # B10 + B11
        W[0][6] = 1                # C00 += M7
        W[3][6] = 1                # C11 += M7

        return U, V, W

    def block_decompose():
        """
        Block decomposition for 3x4x7 using Strassen on 2x2x2 sub-blocks.
        
        Strategy:
        - Split n=3 into (2,1), m=4 into (2,2), p=7 into (2,2,2,1)
        - For each combination, use appropriate decomposition
        - 2x2x2 blocks use Strassen (rank 7)
        - 2x2x1 blocks use schoolbook (rank 4)
        - 1x2x2 blocks use schoolbook (rank 4)
        - 1x2x1 blocks use schoolbook (rank 2)
        
        Total rank:
        - 2x2x2 blocks: 3 (from p split) * 2 (from m split) = 6 blocks * 7 = 42
        - 2x2x1 blocks: 1 (last p) * 2 (from m split) = 2 blocks * 4 = 8
        - 1x2x2 blocks: 3 (from p split) * 2 (from m split) = 6 blocks * 4 = 24
        - 1x2x1 blocks: 1 (last p) * 2 (from m split) = 2 blocks * 2 = 4
        Total: 42+8+24+4 = 78
        """
        # Full dimensions
        n_full, m_full, p_full = 3, 4, 7
        
        # We'll build the full U, V, W matrices
        # U: (n_full*m_full, rank) = (12, rank)
        # V: (m_full*p_full, rank) = (28, rank)
        # W: (p_full*n_full, rank) = (21, rank)
        
        # Use dictionaries to accumulate entries, then convert to lists
        from collections import defaultdict
        U_dict = defaultdict(lambda: defaultdict(float))
        V_dict = defaultdict(lambda: defaultdict(float))
        W_dict = defaultdict(lambda: defaultdict(float))
        
        col = 0
        
        def add_decomp(U_blk, V_blk, W_blk, n_blk, m_blk, p_blk, 
                       i_off, j_off, k_off):
            """
            Add a block decomposition to the full matrices.
            U_blk: (n_blk*m_blk, rank_blk)
            V_blk: (m_blk*p_blk, rank_blk)
            W_blk: (p_blk*n_blk, rank_blk)
            i_off: offset for A rows (n dimension)
            j_off: offset for A cols / B rows (m dimension)
            k_off: offset for B cols / C cols (p dimension)
            """
            nonlocal col
            rank_blk = len(U_blk[0])
            
            for i in range(n_blk):
                for j in range(m_blk):
                    for c in range(rank_blk):
                        val = U_blk[i*m_blk+j][c]
                        if val != 0:
                            # Full U index: (i_off + i) * m_full + (j_off + j)
                            U_dict[(i_off + i) * m_full + (j_off + j)][col + c] += val
            
            for j in range(m_blk):
                for k in range(p_blk):
                    for c in range(rank_blk):
                        val = V_blk[j*p_blk+k][c]
                        if val != 0:
                            # Full V index: (j_off + j) * p_full + (k_off + k)
                            V_dict[(j_off + j) * p_full + (k_off + k)][col + c] += val
            
            for k in range(p_blk):
                for i in range(n_blk):
                    for c in range(rank_blk):
                        val = W_blk[k*n_blk+i][c]
                        if val != 0:
                            # Full W index: (k_off + k) * n_full + (i_off + i)
                            W_dict[(k_off + k) * n_full + (i_off + i)][col + c] += val
            
            col += rank_blk
        
        # Split dimensions:
        # n: [0, 2) and [2, 3) -> sizes 2, 1
        # m: [0, 2) and [2, 4) -> sizes 2, 2
        # p: [0, 2), [2, 4), [4, 6), [6, 7) -> sizes 2, 2, 2, 1
        
        n_splits = [(0, 2), (2, 1)]  # (offset, size)
        m_splits = [(0, 2), (2, 2)]
        p_splits = [(0, 2), (2, 2), (4, 2), (6, 1)]
        
        for i_off, n_size in n_splits:
            for j_off, m_size in m_splits:
                for k_off, p_size in p_splits:
                    if n_size == 2 and m_size == 2 and p_size == 2:
                        # Use Strassen
                        U_blk, V_blk, W_blk = strassen_222()
                        add_decomp(U_blk, V_blk, W_blk, 2, 2, 2, i_off, j_off, k_off)
                    elif n_size == 2 and m_size == 2 and p_size == 1:
                        # Schoolbook 2x2x1
                        rank_blk = 4
                        U_blk = [[0]*rank_blk for _ in range(4)]
                        V_blk = [[0]*rank_blk for _ in range(2)]
                        W_blk = [[0]*rank_blk for _ in range(2)]
                        c = 0
                        for i in range(2):
                            for j in range(2):
                                U_blk[i*2+j][c] = 1
                                V_blk[j*1+0][c] = 1
                                W_blk[0*2+i][c] = 1
                                c += 1
                        add_decomp(U_blk, V_blk, W_blk, 2, 2, 1, i_off, j_off, k_off)
                    elif n_size == 1 and m_size == 2 and p_size == 2:
                        # Schoolbook 1x2x2
                        rank_blk = 4
                        U_blk = [[0]*rank_blk for _ in range(2)]
                        V_blk = [[0]*rank_blk for _ in range(4)]
                        W_blk = [[0]*rank_blk for _ in range(2)]
                        c = 0
                        for i in range(1):
                            for j in range(2):
                                for k in range(2):
                                    U_blk[i*2+j][c] = 1
                                    V_blk[j*2+k][c] = 1
                                    W_blk[k*1+i][c] = 1
                                    c += 1
                        add_decomp(U_blk, V_blk, W_blk, 1, 2, 2, i_off, j_off, k_off)
                    elif n_size == 1 and m_size == 2 and p_size == 1:
                        # Schoolbook 1x2x1
                        rank_blk = 2
                        U_blk = [[0]*rank_blk for _ in range(2)]
                        V_blk = [[0]*rank_blk for _ in range(2)]
                        W_blk = [[0]*rank_blk for _ in range(1)]
                        c = 0
                        for i in range(1):
                            for j in range(2):
                                U_blk[i*2+j][c] = 1
                                V_blk[j*1+0][c] = 1
                                W_blk[0*1+i][c] = 1
                                c += 1
                        add_decomp(U_blk, V_blk, W_blk, 1, 2, 1, i_off, j_off, k_off)
        
        # Convert dictionaries to lists
        total_rank = col
        U = [[0] * total_rank for _ in range(n_full * m_full)]
        V = [[0] * total_rank for _ in range(m_full * p_full)]
        W = [[0] * total_rank for _ in range(p_full * n_full)]
        
        for i in range(n_full * m_full):
            for c in range(total_rank):
                if i in U_dict and c in U_dict[i]:
                    U[i][c] = U_dict[i][c]
        
        for j in range(m_full * p_full):
            for c in range(total_rank):
                if j in V_dict and c in V_dict[j]:
                    V[j][c] = V_dict[j][c]
        
        for k in range(p_full * n_full):
            for c in range(total_rank):
                if k in W_dict and c in W_dict[k]:
                    W[k][c] = W_dict[k][c]
        
        return {"U": U, "V": V, "W": W}

    # Try block decomposition
    try:
        block_result = block_decompose()
        if verify(block_result["U"], block_result["V"], block_result["W"], n, m, p):
            best = block_result
    except Exception:
        pass

    # Return the best result
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
