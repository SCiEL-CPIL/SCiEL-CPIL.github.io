def _apply_single(M, q, g, n):
    # M: (2^n, 2^n), g: (2,2)
    # reshape to (2,)*2n, first n axes = row (q0..), next n = col
    T = M.reshape([2]*(2*n))
    # contract g with row axis q and col axis n+q
    # new_T = einsum('ab,...b...d...->...a...c...')? messy.
    # Do two tensordots:
    # Move row-axis q to position 0
    T = np.moveaxis(T, q, 0)
    T = np.tensordot(g, T, axes=([1],[0]))  # g[a,b] T[b, rest] -> T'[a, rest], rest = 2n-1 axes
    # now T' shape (2, 2n-1). The col axis for q was originally at n+q; after moving row q to front and contracting, indices shifted.
    ...
