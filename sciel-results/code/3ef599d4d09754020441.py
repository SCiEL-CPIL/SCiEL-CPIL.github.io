<<<<<<< SEARCH
def _als_decomposition(rank, rng, iters=200, deadline=None):
    """
    Use Alternating Least Squares to find a rank-`rank` decomposition.
    Returns (U, V, W) or None if time exceeded.
    """
    # Initialize
    U = rng.normal(size=(n * m, rank))
    V = rng.normal(size=(m * p, rank))
    W = rng.normal(size=(p * n, rank))
=======
def _als_decomposition(rank, rng, iters=200, deadline=None):
    """
    Use Alternating Least Squares to find a rank-`rank` decomposition.
    Returns (U, V, W) or None if time exceeded.
    """
    rank = int(rank)
    # Initialize
    U = rng.normal(size=(n * m, rank))
    V = rng.normal(size=(m * p, rank))
    W = rng.normal(size=(p * n, rank))
>>>>>>> REPLACE
