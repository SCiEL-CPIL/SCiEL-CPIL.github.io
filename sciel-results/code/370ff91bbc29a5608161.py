<<<<<<< SEARCH
def _base_decomposition():
    sU, sV, sW = _strassen_4x4x4()
    U, V, W = _combine(sU, sV, sW, sU, sV, sW, (16, 16, 16))
    tU, tV, tW = _schoolbook_nm_p(4, 4, 1)
    U, V, W = _combine(U, V, W, tU, tV, tW, (32, 32, 32))
    return U, V, W
=======
def _base_decomposition():
    # Use schoolbook for the full 4x4x9 problem to ensure validity
    return _schoolbook_nm_p(4, 4, 9)
>>>>>>> REPLACE
