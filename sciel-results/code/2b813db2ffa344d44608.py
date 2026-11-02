def add_2x1_block(i0, i1, j0, k0, k1):
    add_term({(i0, j0): 1, (i1, j0): 1}, {(j0, k0): 1, (j0, k1): 1}, {(k0, i0): 1, (k0, i1): 1, (k1, i0): 1, (k1, i1): 1})
