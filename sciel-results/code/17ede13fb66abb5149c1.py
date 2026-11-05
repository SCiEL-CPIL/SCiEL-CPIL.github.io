def _verify(network):
    inputs = _get_initial_inputs().copy()
    for (i, j) in network:
        a = inputs[:, i]
        b = inputs[:, j]
        inputs[:, i] = np.minimum(a, b)
        inputs[:, j] = np.maximum(a, b)
    return not np.any(inputs[:, :-1] > inputs[:, 1:])
