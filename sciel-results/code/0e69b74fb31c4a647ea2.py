        Cv = C[viol]
        base = best_v.copy()
        beta = 8.0
        for p in range(4000):
            r = np.mod(Cv @ base, m).astype(np.float64)
            r = np.minimum(r, m - r)
            w = np.exp(-beta * r * r)
            w = w / (w.sum() + 1e-12)
            gvec = -(Cv.T @ w) * (2.0 * r)
        