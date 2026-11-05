def construct():
    """Construct a tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U': ..., 'V': ..., 'W': ...} with integer/half-integer entries.
    """
    U, V, W = _schoolbook()
    
    # Ensure we return lists of floats
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist(),
    }
