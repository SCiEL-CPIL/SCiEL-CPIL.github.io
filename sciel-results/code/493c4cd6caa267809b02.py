def validate_network(net):
    # Input: list of (i, j)
    # Create array of all 2^N inputs
    # Shape: (2^N, N)
    # For each comp (i, j):
    #   a = state[:, i]
    #   b = state[:, j]
    #   min_ab = np.minimum(a, b)
    #   max_ab = np.maximum(a, b)
    #   state[:, i] = min_ab
    #   state[:, j] = max_ab
    # Check if final state is sorted: state[:, k] <= state[:, k+1] for all k
    pass
