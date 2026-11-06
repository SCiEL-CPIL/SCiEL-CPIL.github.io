# Quick check with subset
def _is_valid_fast(network, subset_inputs):
    state = _simulate(network, subset_inputs)
    diffs = np.diff(state, axis=1)
    return np.all(diffs >= 0)

# Pre-generate subset
SUBSET_SIZE = 2000
subset_indices = np.random.choice(2**N, SUBSET_SIZE, replace=False)
subset_inputs = (subset_indices[:, None] >> np.arange(N)[::-1]) & 1
