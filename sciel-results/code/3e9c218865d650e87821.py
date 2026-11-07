    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Expected shape is (5, 100) based on task description
    # But let's be flexible.
    if prizes.shape != weights.shape:
        # Try to align them.
        # If one is (100, 5) and other is (5, 100), transpose the (100, 5) one.
        if prizes.T.shape == weights.shape:
            prizes = prizes.T
        elif weights.T.shape == prizes.shape:
            weights = weights.T
        else:
            raise ValueError(f"prizes and weights must have the same shape or be transposes. Got {prizes.shape} and {weights.shape}")
            
    # Now they have the same shape.
    # Ensure it's (n_constraints, n_items)
    # Task says 5 constraints, 100 items.
    # If shape is (100, 5), we should transpose to (5, 100) for the logic below?
    # The logic uses `prizes.sum(axis=1)` which sums over items if shape is (5, 100).
    # If shape is (100, 5), `sum(axis=1)` sums over constraints.
    # The task says "prizes and weights, of shape (5, 100)".
    # So we assume (5, 100).
    # If they come in as (100, 5), we should transpose them to (5, 100).
    
    if prizes.shape == (100, 5):
        prizes = prizes.T
        weights = weights.T
    elif prizes.shape != (5, 100):
        # Fallback: assume first dim is constraints if it's smaller?
        # But task is fixed.
        pass
