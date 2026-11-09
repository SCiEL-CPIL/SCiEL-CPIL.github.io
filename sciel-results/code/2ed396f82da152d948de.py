"""Program for rank-minimal tensor decomposition of matrix multiplication 3x4x7."""
DIMENSIONS = [3, 4, 7]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact using NumPy."""
    import numpy as np
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
    DEADLINE = start_time + 115

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
        """Verify exactness."""
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
          U: shape (4, 7)
          V: shape (4, 7)
          W: shape (4, 7)
        Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        rank = 7
        U = [[0] * rank for _ in range(4)]
        V = [[0] * rank for _ in range(4)]
        W = [[0] * rank for _ in range(4)]

        # M1: (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1

        # M2: (A10+A11)(B00)
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1

        # M3: A00(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1

        # M4: A11(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1
        W[1][3] = 1

        # M5: (A00+A01)(B11)
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[2][4] = 1
        W[0][4] = -1

        # M6: (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1

        # M7: (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        W[3][6] = 1

        return U, V, W

    def merge_duplicate_columns(U, V, W, n, m, p):
        """
        Merge duplicate rank columns to reduce total rank.
        Two columns c1, c2 can be merged if for all (i,j,k):
        U[i,j,c1]*V[j,k,c1]*W[k,i,c1] == U[i,j,c2]*V[j,k,c2]*W[k,i,c2]
        """
        import numpy as np
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Compute the tensor contribution of each column
        # T[i,j,k,c] = U[i,j,c] * V[j,k,c] * W[k,i,c]
        T = np.zeros((n, m, p, rank))
        for c in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k, c] = U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
        
        # Try to merge columns
        merged = list(range(rank))
        for c1 in range(rank):
            if c1 not in merged:
                continue
            for c2 in range(c1 + 1, rank):
                if c2 not in merged:
                    continue
                # Check if columns are identical
                if np.allclose(T[:, :, :, c1], T[:, :, :, c2], atol=1e-10):
                    merged.remove(c2)
        
        # Build new matrices
        new_rank = len(merged)
        new_U = [[0] * new_rank for _ in range(n * m)]
        new_V = [[0] * new_rank for _ in range(m * p)]
        new_W = [[0] * new_rank for _ in range(p * n)]
        
        for new_c, old_c in enumerate(merged):
            for i in range(n * m):
                new_U[i][new_c] = U[i][old_c]
            for j in range(m * p):
                new_V[j][new_c] = V[j][old_c]
            for k in range(p * n):
                new_W[k][new_c] = W[k][old_c]
        
        return {"U": new_U, "V": new_V, "W": new_W}

    def block_decompose():
        """
        Block decomposition for 3x4x7 using Strassen on 2x2x2 sub-blocks.
        Total rank: 78
        """
        n_full, m_full, p_full = 3, 4, 7
        
        from collections import defaultdict
        U_dict = defaultdict(lambda: defaultdict(float))
        V_dict = defaultdict(lambda: defaultdict(float))
        W_dict = defaultdict(lambda: defaultdict(float))
        
        col = 0
        
        def add_decomp(U_blk, V_blk, W_blk, n_blk, m_blk, p_blk, 
                       i_off, j_off, k_off):
            nonlocal col
            rank_blk = len(U_blk[0])
            
            for i in range(n_blk):
                for j in range(m_blk):
                    for c in range(rank_blk):
                        val = U_blk[i*m_blk+j][c]
                        if val != 0:
                            U_dict[(i_off + i) * m_full + (j_off + j)][col + c] += val
            
            for j in range(m_blk):
                for k in range(p_blk):
                    for c in range(rank_blk):
                        val = V_blk[j*p_blk+k][c]
                        if val != 0:
                            V_dict[(j_off + j) * p_full + (k_off + k)][col + c] += val
            
            for k in range(p_blk):
                for i in range(n_blk):
                    for c in range(rank_blk):
                        val = W_blk[k*n_blk+i][c]
                        if val != 0:
                            W_dict[(k_off + k) * n_full + (i_off + i)][col + c] += val
            
            col += rank_blk
        
        n_splits = [(0, 2), (2, 1)]
        m_splits = [(0, 2), (2, 2)]
        p_splits = [(0, 2), (2, 2), (4, 2), (6, 1)]
        
        for i_off, n_size in n_splits:
            for j_off, m_size in m_splits:
                for k_off, p_size in p_splits:
                    if n_size == 2 and m_size == 2 and p_size == 2:
                        U_blk, V_blk, W_blk = strassen_222()
                        add_decomp(U_blk, V_blk, W_blk, 2, 2, 2, i_off, j_off, k_off)
                    elif n_size == 2 and m_size == 2 and p_size == 1:
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

    try:
        block_result = block_decompose()
        if verify(block_result["U"], block_result["V"], block_result["W"], n, m, p):
            best = block_result
    except Exception:
        pass
    
    # Try to reduce rank by merging duplicate columns
    try:
        if time.time() < DEADLINE - 5:
            merged_result = merge_duplicate_columns(best["U"], best["V"], best["W"], n, m, p)
            if verify(merged_result["U"], merged_result["V"], merged_result["W"], n, m, p):
                if len(merged_result["U"][0]) < len(best["U"][0]):
                    best = merged_result
    except Exception:
        pass

    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
