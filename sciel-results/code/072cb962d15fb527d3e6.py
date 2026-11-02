<<<<<<< SEARCH
def _autoconv(h):
    """Compute full autoconvolution of h."""
    n = len(h)
    c = np.zeros(2*n - 1)
    for k in range(n):
        c[k] = np.dot(h[:n-k], h[k:])
        c[2*n-1-k] = c[k]
    return c
=======
def _autoconv(h):
    """Compute full autoconvolution of h."""
    n = len(h)
    c = np.zeros(2*n - 1)
    for k in range(n):
        c[k] = np.dot(h[:n-k], h[k:])
        c[2*n-2-k] = c[k]
    return c
>>>>>>> REPLACE
