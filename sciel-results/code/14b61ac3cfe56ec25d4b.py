<<<<<<< SEARCH
    def full_verify(U, V, W):
        """Full verification of the decomposition."""
        r = U.shape[1]
        result = np.zeros((nm, mp, pn), dtype=np.float64)
        for c in range(r):
            result += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
        for (i, j, k) in T_indices:
            if abs(result[i, j, k] - 1) > 1e-10:
                return False
        for (i, j, k) in all_zero_indices:
            if abs(result[i, j, k]) > 1e-10:
                return False
        return True
=======
    def full_verify(U, V, W):
        """Full verification of the decomposition."""
        r = U.shape[1]
        result = np.zeros((nm, mp, pn), dtype=np.float64)
        for c in range(r):
            result += np.einsum('ir,jr,kr->ijk', U[:, c], V[:, c], W[:, c])
        for (i, j, k) in T_indices:
            if abs(result[i, j, k] - 1) > 1e-10:
                return False
        for (i, j, k) in all_zero_indices:
            if abs(result[i, j, k]) > 1e-10:
                return False
        return True
>>>>>>> REPLACE

The `compute_error_fast` function is also slow because it loops over indices in Python. We can vectorize the non-zero check. The zero check is harder to vectorize easily without building the full tensor, but we can keep the loop for zeros if the sample size is small (300). Let's optimize the non-zero part which is 42 entries.

