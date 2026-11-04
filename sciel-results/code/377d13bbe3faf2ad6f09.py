"""Matrix multiplication 5x5x5 using block decomposition with Strassen for 2x2x2 blocks."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _strassen_2x2x2():
    """Return the 7-term Strassen decomposition for 2x2x2 matrix multiplication."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1],   # A11
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 1, 0, 0, 1, 0],    # B01
        [0, 0, 0, 1, 1, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1],   # B11
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C00 (W[0])
        [0, 1, 0, 1, 0, 0, 0],    # C10 (W[1])
        [0, 0, 1, 0, 1, 0, 0],    # C01 (W[2])
        [1, -1, 1, 0, 0, 1, 0],   # C11 (W[3])
    ]
    return U, V, W

def _direct_sxtt(s, t, u):
    """Direct (schoolbook) decomposition for s×t×u matrix multiplication."""
    rank = s * t * u
    U = [[0] * rank for _ in range(s * t)]
    V = [[0] * rank for _ in range(t * u)]
    W = [[0] * rank for _ in range(u * s)]
    col = 0
    for i in range(s):
        for j in range(t):
            for k in range(u):
                U[i * t + j][col] = 1
                V[j * u + k][col] = 1
                W[k * s + i][col] = 1
                col += 1
    return U, V, W

def _combine_blocks(block_specs):
    """Combine block decompositions into a full 5x5x5 decomposition."""
    n = m = p = 5
    total_rank = sum(len(Ub[0]) for _, _, _, s, t, u, Ub, Vb, Wb in block_specs)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for (i_start, j_start, k_start, s, t, u, Ub, Vb, Wb) in block_specs:
        r = len(Ub[0])
        for row in range(s * t):
            i_local, j_local = divmod(row, t)
            i_global = i_start + i_local
            j_global = j_start + j_local
            for c in range(r):
                U[i_global * m + j_global][col_offset + c] = Ub[row][c]
        for row in range(t * u):
            j_local, k_local = divmod(row, u)
            j_global = j_start + j_local
            k_global = k_start + k_local
            for c in range(r):
                V[j_global * p + k_global][col_offset + c] = Vb[row][c]
        for row in range(u * s):
            k_local, i_local = divmod(row, s)
            k_global = k_start + k_local
            i_global = i_start + i_local
            for c in range(r):
                W[k_global * n + i_global][col_offset + c] = Wb[row][c]
        col_offset += r
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    
    # Block sizes: 5 = 2 + 2 + 1
    block_starts = [0, 2, 4]
    block_sizes = [2, 2, 1]
    
    strassen_U, strassen_V, strassen_W = _strassen_2x2x2()
    
    block_specs = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                i_start = block_starts[a]
                j_start = block_starts[b]
                k_start = block_starts[c]
                s = block_sizes[a]
                t = block_sizes[b]
                u = block_sizes[c]
                
                if s == 2 and t == 2 and u == 2:
                    Ub, Vb, Wb = strassen_U, strassen_V, strassen_W
                else:
                    Ub, Vb, Wb = _direct_sxtt(s, t, u)
                
                block_specs.append((i_start, j_start, k_start, s, t, u, Ub, Vb, Wb))
    
    result = _combine_blocks(block_specs)
    
    elapsed = time.time() - start_time
    if elapsed > 100:
        pass
    
    return result
# EVOLVE-BLOCK-END
