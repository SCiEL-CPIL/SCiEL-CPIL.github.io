<<<<<<< SEARCH
def _scalar_mult_block(rows_a, cols_b, rows_c):
    """
    Schoolbook decomposition for A[rows_a x cols_b] * B[cols_b x rows_c] = C[rows_a x rows_c].
    Returns list of (u, v, w) tuples.
    u has rows_a*cols_b entries indexed by i*cols_b+j
    v has cols_b*rows_c entries indexed by j*rows_c+k
    w has rows_c*rows_a entries indexed by i*rows_c+k (C[i,k])
    """
    terms = []
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(rows_c):
                u = [0]*(rows_a*cols_b)
                v = [0]*(cols_b*rows_c)
                w = [0]*(rows_c*rows_a)
                u[i*cols_b + j] = 1
                v[j*rows_c + k] = 1
                w[i*rows_c + k] = 1  # C[i,k] indexed as i*rows_c+k
                terms.append((u, v, w))
    return terms
=======
def _scalar_mult_block(rows_a, cols_b, rows_c):
    """
    Schoolbook decomposition for A[rows_a x cols_b] * B[cols_b x rows_c] = C[rows_a x rows_c].
    Returns list of (u, v, w) tuples.
    u has rows_a*cols_b entries indexed by i*cols_b+j
    v has cols_b*rows_c entries indexed by j*rows_c+k
    w has rows_c*rows_a entries indexed by k*rows_a+i (matching W[k*n+i] convention)
    """
    terms = []
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(rows_c):
                u = [0]*(rows_a*cols_b)
                v = [0]*(cols_b*rows_c)
                w = [0]*(rows_c*rows_a)
                u[i*cols_b + j] = 1
                v[j*rows_c + k] = 1
                w[k*rows_a + i] = 1  # W convention: k*rows_a + i
                terms.append((u, v, w))
    return terms
>>>>>>> REPLACE

Now I need to also fix the Laderman and Strassen w-indexing to match. In `_laderman_3x3`, w should be indexed as k*3+i. Let me check my Laderman terms... I had `w_vec[k*3+i]` in the docstring but let me make sure the actual data matches.

Actually, this is getting complicated. Let me just rewrite the whole thing more carefully with a verification step. Let me also reconsider - maybe I should just use a simpler, verified approach.

Let me rewrite the entire EVOLVE-BLOCK with a clean, verified implementation:

