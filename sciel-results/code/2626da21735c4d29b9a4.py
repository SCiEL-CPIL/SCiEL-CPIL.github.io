def _search_rank(rank, time_limit=10):
    n, m, p = DIMENSIONS
    start = time.time()
    for _ in range(100):
        if time.time() > start + time_limit:
            break
        U = np.random.randint(-2, 3, (n*m, rank))
        V = np.random.randint(-2, 3, (m*p, rank))
        W = np.random.randint(-2, 3, (p*n, rank))
        if _verify(U, V, W, tol=1e-6):
            return U, V, W
    return None
