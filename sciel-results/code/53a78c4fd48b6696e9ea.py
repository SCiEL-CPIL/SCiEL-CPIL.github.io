def construct():
    """Entry point: return the best decomposition found."""
    # The tensor T_{2,2,8} is the direct sum of four T_{2,2,2} tensors.
    # Rank(T_{2,2,8}) = 4 * Rank(T_{2,2,2}) = 4 * 7 = 28.
    # This is the theoretical minimum. No search can improve this.
    U, V, W = strassen_decomposition_228()
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
