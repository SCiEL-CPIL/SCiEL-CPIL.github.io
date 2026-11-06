# EVOLVE-BLOCK-START
def construct():
    import numpy as np
    import time

    N = 16
    start_time = time.time()
    deadline = start_time + 85.0

    # 1. Generate Batcher's Odd-Even Merge Sort Network
    def gen_merge(n):
        if n == 1:
            return []
        half = n // 2
        # Recursively merge odd indices (1, 3, ..., n-1)
        # Map sub-index i to 2*i + 1
        odd_merge = [(2*i + 1, 2*j + 1) for i, j in gen_merge(half)]
        # Recursively merge even indices (0, 2, ..., n-2)
        # Map sub-index i to 2*i
        even_merge = [(2*i, 2*j) for i, j in gen_merge(half)]
        # Final layer: (0,1), (2,3), ...
        final_layer = [(2*k, 2*k + 1) for k in range(half)]
        return odd_merge + even_merge + final_layer

    def gen_sort(n):
        if n == 1:
            return []
        half = n // 2
        # Sort evens
        even_sort = [(2*i, 2*j) for i, j in gen_sort(half)]
        # Sort odds
        odd_sort = [(2*i + 1, 2*j + 1) for i, j in gen_sort(half)]
        # Merge
        merge_comps = gen_merge(n)
        return even_sort + odd_sort + merge_comps

    network = gen_sort(N)
    
    # 2. Verify the network
    # Pre-compute inputs for verification
    inputs = np.array([list(map(int, format(i, f'0{N}b'))) for i in range(2**N)])
    
    def verify_fast(comparators):
        state = inputs.copy()
        for i, j in comparators:
            mask = state[:, i] > state[:, j]
            if np.any(mask):
                state[mask, i] = 1 - state[mask, i]
                state[mask, j] = 1 - state[mask, j]
        return np.array_equal(state, np.sort(state, axis=1))

    if verify_fast(network):
        return network
    else:
        # Fallback to Odd-Even Transposition (120 comparators)
        # This is guaranteed to be correct
        fallback = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        return fallback
# EVOLVE-BLOCK-END
