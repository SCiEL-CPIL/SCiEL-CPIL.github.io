def try_reduce_rank(U, V, W, T):
    """Try to reduce rank by checking if any term is redundant."""
    rank = U.shape[1]
    dim = T.size  # 6*18*12 = 1296
    
    # Flatten T
    T_flat = T.flatten()
    
    # Compute all terms
    terms = []
    for r in range(rank):
        term = np.outer(np.outer(U[:,r], V[:,r]), W[:,r]).flatten()
        terms.append(term)
    
    # For each r, check if T is in span of remaining terms
    for r in range(rank):
        remaining = [terms[i] for i in range(rank) if i != r]
        remaining = np.array(remaining).T  # shape (1296, rank-1)
        # Check if T_flat is in the column space of remaining
        # Solve remaining @ x = T_flat
        try:
            x, residuals, rank_rem, sv = np.linalg.lstsq(remaining, T_flat, rcond=None)
            if len(residuals) > 0 and residuals[0] < 1e-9:
                return r, remaining, x
        except:
            pass
    
    return None
