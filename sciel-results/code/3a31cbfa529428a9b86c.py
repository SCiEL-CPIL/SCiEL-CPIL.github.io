    # Phase 1: Spectral Initialization
    def spectral_init():
        # Construct matrix M = A o B (Hadamard product)
        # This captures the interaction between flow and distance
        M = a * b
        
        # Compute leading eigenvector of M
        # Using eigh for symmetric matrix (assuming A, B are symmetric)
        # If not symmetric, we might need to symmetrize or use eigs
        # For QAP, A and B are typically symmetric.
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(M)
            # eigh returns eigenvalues in ascending order
            # The leading eigenvector corresponds to the largest eigenvalue
            v = eigenvectors[:, -1]
        except Exception:
            # Fallback to random if eigh fails
            rng = np.random.default_rng(123)
            v = rng.normal(0, 1, n)
            v /= np.linalg.norm(v)
            
        # Use the leading eigenvector to define a cost matrix for Hungarian
        # We want to align facilities and locations with similar spectral components
        # C[i, j] = -v[i] * v[j]
        # We add small noise to break ties and avoid degenerate solutions
        C = -np.outer(v, v)
        rng = np.random.default_rng(123)
        C += rng.normal(0, 0.01, (n, n))
        
        # Solve Linear Assignment Problem
        row_idx, col_idx = linear_sum_assignment(C)
        pos = col_idx.copy()
        
        cost = compute_cost(pos)
        return pos, cost
