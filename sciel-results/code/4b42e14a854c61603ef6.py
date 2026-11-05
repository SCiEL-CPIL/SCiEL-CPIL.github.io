def _simulate_network(network, inputs):
    state = inputs.copy()
    for i, j in network:
        a = state[:, i]
        b = state[:, j]
        # min to i, max to j
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    return state
