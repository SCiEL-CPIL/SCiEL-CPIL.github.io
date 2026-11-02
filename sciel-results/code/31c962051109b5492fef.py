def _adapt_to_4x8(U4, V4, W4):
    """
    Adapt 4x4 Strassen factors to 4x8 multiplication.
    Input: U4: 16x49, V4: 16x49, W4: 16x49 (for 4x4 * 4x4 -> 4x4)
    Output: U: 16x98, V: 32x98, W: 32x98 (for 4x4 * 4x8 -> 4x8)
    """
    rank4 = U4.shape[1]
    rank8 = rank4 * 2
    
    # U: 16 x 98
    # Both halves use the same U coefficients from 4x4
    U = np.zeros((16, rank8))
    U[:, :rank4] = U4
    U[:, rank4:] = U4
    
    # V: 32 x 98
    # First 16 rows (B_left) use V4 in first 49 cols
    # Last 16 rows (B_right) use V4 in last 49 cols
    V = np.zeros((32, rank8))
    V[:16, :rank4] = V4
    V[16:, rank4:] = V4
    
    # W: 32 x 98
    # First 16 rows (C_left) use W4 in first 49 cols
    # Last 16 rows (C_right) use W4 in last 49 cols
    W = np.zeros((32, rank8))
    W[:16, :rank4] = W4
    W[16:, rank4:] = W4
    
    return U.tolist(), V.tolist(), W.tolist()
