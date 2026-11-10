<<<<<<< SEARCH
def construct():
    start = time.time()
    dl = start + DEADLINE * SAFETY

    def _fibonacci_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1 - z ** 2))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)

    best = None
    best_E = 1e18

    # Try multiple initializations
    np.random.seed(42)
    initializations = [_fibonacci_spiral()]
    for _ in range(5):
        X = np.random.randn(N, 3)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        initializations.append(X)

    for X0 in initializations:
        if time.time() > dl:
            break
        Xopt = _opt_cartesian(X0, maxiter=800)
        Xopt /= np.linalg.norm(Xopt, axis=1, keepdims=True)
        E = _energy(Xopt)
        if E < best_E:
            best_E = E
            best = Xopt.copy()

    if best is None:
        best = _fibonacci_spiral()

    # Final refinement with tighter tolerances if time permits
    if time.time() < dl - 5.0:
        Xref = _opt_cartesian(best, maxiter=500)
        Xref /= np.linalg.norm(Xref, axis=1, keepdims=True)
        Eref = _energy(Xref)
        if Eref < best_E:
            best = Xref

    return best
=======
def construct():
    start = time.time()
    dl = start + DEADLINE * SAFETY

    def _fibonacci_spiral():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1 - z ** 2))
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)

    best = None
    best_E = 1e18

    # Try multiple initializations
    np.random.seed(42)
    initializations = [_fibonacci_spiral()]
    for _ in range(5):
        X = np.random.randn(N, 3)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        initializations.append(X)

    for X0 in initializations:
        if time.time() > dl:
            break
        Xopt = _opt_cartesian(X0, maxiter=800)
        Xopt /= np.linalg.norm(Xopt, axis=1, keepdims=True)
        E = _energy(Xopt)
        if E < best_E:
            best_E = E
            best = Xopt.copy()

    if best is None:
        best = _fibonacci_spiral()

    # Final refinement with Riemannian CG if time permits
    if time.time() < dl - 10.0:
        Xref = _riemannian_cg(best, max_iter=1000, tol=1e-13, time_limit=dl - 5.0)
        Xref /= np.linalg.norm(Xref, axis=1, keepdims=True)
        Eref = _energy(Xref)
        if Eref < best_E:
            best = Xref
            best_E = Eref

    # Additional L-BFGS polish if time remains
    if time.time() < dl - 3.0:
        Xref2 = _opt_cartesian(best, maxiter=300)
        Xref2 /= np.linalg.norm(Xref2, axis=1, keepdims=True)
        Eref2 = _energy(Xref2)
        if Eref2 < best_E:
            best = Xref2

    return best
>>>>>>> REPLACE
