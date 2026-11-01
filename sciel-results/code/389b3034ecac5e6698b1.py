<<<<<<< SEARCH
def _precompute(q, n, radius):
=======
def _int_to_digits(w, q, n):
    """Convert integer index w to a list of n digits in base q."""
    digs = []
    for _ in range(n):
        digs.append(w % q)
        w //= q
    return digs


def _precompute(q, n, radius):
>>>>>>> REPLACE

