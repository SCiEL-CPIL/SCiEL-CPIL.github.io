import numpy as np

def _fast_verify(U, V, W, n, m, p):
    # U: (nm, R), V: (mp, R), W: (pn, R)
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # We need T to match the multiplication tensor.
    # Instead of checking all a,b,c, we can check the reconstruction.
    # But direct check is safer.
    # Using NumPy:
    # T = np.einsum('ar,br,cr->abc', U, V, W)
    # This creates a (nm, mp, pn) array.
    # nm=6, mp=18, pn=12. Size 1296.
    # R is small.
    # This is very fast.
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Construct expected tensor
    # T_expected[i*m+j, j'*p+k, k'*n+i'] = 1 if j==j', k==k', i==i' else 0
    # We can construct this efficiently.
    # Or just check the non-zero entries.
    # The non-zero entries are at (i*m+j, j*p+k, k*n+i) for i,j,k.
    # There are n*m*p = 36 such entries.
    # We can check these 36 entries.
    # And we must ensure all OTHER entries are 0.
    # Total entries 1296.
    # Checking 1296 entries is fast.
    
    # Let's just check all.
    # Create indices
    i_idx, j_idx, k_idx = np.indices((n, m, p))
    # Flatten to a, b, c
    a = i_idx * m + j_idx
    b = j_idx * p + k_idx
    c = k_idx * n + i_idx
    
    # The expected tensor is 1 at these (a,b,c) and 0 elsewhere.
    # We can check if T matches this.
    # T is (nm, mp, pn).
    # We need to check T[a, b, c] == 1 for all valid (i,j,k).
    # And T[other] == 0.
    
    # Check valid ones
    valid_vals = T[a, b, c]
    if not np.allclose(valid_vals, 1.0, atol=1e-9):
        return False
        
    # Check invalid ones
    # Create a mask of valid (a,b,c)
    mask = np.zeros((n*m, m*p, p*n), dtype=bool)
    mask[a, b, c] = True
    invalid_vals = T[~mask]
    if not np.allclose(invalid_vals, 0.0, atol=1e-9):
        return False
        
    return True
