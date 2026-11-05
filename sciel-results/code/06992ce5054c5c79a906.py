def _iswap_gate():
    """Return iSWAP gate matrix."""
    return np.array([[1, 0, 0, 0],
                     [0, 0, 1j, 0],
                     [0, 1j, 0, 0],
                     [0, 0, 0, 1]], dtype=complex)
