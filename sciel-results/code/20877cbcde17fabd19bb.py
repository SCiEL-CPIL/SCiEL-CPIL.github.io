_FULL_INPUTS = None

def _get_full_inputs():
    global _FULL_INPUTS
    if _FULL_INPUTS is None:
        _FULL_INPUTS = np.array([(i >> j) & 1 for i in range(2**N) for j in range(N)]).reshape(-1, N)
    return _FULL_INPUTS
