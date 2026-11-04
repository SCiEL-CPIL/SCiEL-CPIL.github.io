<<<<<<< SEARCH
            if min_val < 0:
                # Find best in lower triangle only
                flat_idx = np.argmin(tri_deltas)
                # Convert flat index in lower triangle to (i,j)
                # Number of elements in lower triangle rows:
                # Row 0: 0, Row 1: 1, Row 2: 2, ..., Row k: k
                # Cumulative: 0, 1, 3, 6, ...
                # Find which row
                row = int(np.ceil((-1 + np.sqrt(1 + 8 * (flat_idx + 1))) / 2)) - 1
                # Actually let's just use a simpler approach
                # Set diagonal and upper triangle to inf, then argmin
                d_masked = deltas.copy()
                d_masked[np.diag_indices(n)] = np.inf
                d_masked[np.triu_indices(n, k=1)] = np.inf
                i, j = np.unravel_index(np.argmin(d_masked), (n, n))
                p[i], p[j] = p[j], p[i]
                improved = True
=======
            if min_val < 0:
                flat_idx = np.argmin(tri_deltas)
                i, j = idx[0][flat_idx], idx[1][flat_idx]
                p[i], p[j] = p[j], p[i]
                improved = True
>>>>>>> REPLACE

Now fix the final polish similarly:

