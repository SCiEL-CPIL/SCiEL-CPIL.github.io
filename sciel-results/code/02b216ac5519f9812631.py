    # Precompute overlap matrix
    # Use a list of lists or a flat array
    # N=5040. N*N = 25,401,600.
    # Python lists of lists are slow.
    # Use array.array('B') or numpy?
    # NumPy is available.
    import numpy as np
    
    # But building the matrix takes time.
    # Let's see if it's worth it.
    # If we do 1M iterations in 100s, that's 10k/s.
    # With matrix lookup, we can do 1M/s?
    # Yes, huge speedup.
    
    # Let's build the matrix.
    # overlap_mat[i][j] = overlap(perms[i], perms[j])
    
    # This might take too long to build in Python loops.
    # Vectorized?
    # We can compare suffixes and prefixes.
    # Suffixes of all perms: shape (N, n-1)
    # Prefixes of all perms: shape (N, n-1)
    # For each k, check if suffix_k == prefix_k.
    
    # Let's try to do it efficiently.
    # Actually, for n=7, the number of non-zero overlaps is small.
    # Each perm has exactly 1 neighbor with overlap 6 (the rotation).
    # How many have overlap 5?
    # P = (a,b,c,d,e,f,g).
    # Q must start with (c,d,e,f,g).
    # Q = (c,d,e,f,g,h). h must be a or b.
    # So 2 neighbors with overlap 5?
    # Generally, number of neighbors with overlap k is (n-k).
    # So the matrix is very sparse.
    # We can store it as a dictionary of (i,j) -> ov.
    # Or just compute on the fly.
    # Given the speedup from SA, on-the-fly might be enough if `fast_overlap` is optimized.
    
    # I will stick to on-the-fly `fast_overlap` to avoid startup time risk.
