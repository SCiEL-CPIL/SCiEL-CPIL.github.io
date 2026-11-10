<<<<<<< SEARCH
def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Build index and reverse map
    idx, total, reverse = _build_index(v, t)

    # Fast greedy construction
    blocks = _greedy_construct(v, k, t, idx, total, reverse, deadline)

    # Deterministic improvement pass
    if time.time() < deadline - 10:
        blocks = _improve_blocks(blocks, v, k, t, idx, deadline)

    # Randomized removal pass
    if time.time() < deadline - 8:
        blocks = _random_removal_pass(blocks, v, k, t, idx, deadline)

    # Block swap optimization
    if time.time() < deadline - 6:
        blocks = _block_swap(blocks, v, k, t, idx, deadline)

    # Simulated annealing
    if time.time() < deadline - 3:
        blocks = _simulated_annealing(blocks, v, k, t, idx, deadline)

    # Final cleanup pass
    if time.time() < deadline - 1:
        blocks = _improve_blocks(blocks, v, k, t, idx, deadline)
        blocks = _random_removal_pass(blocks, v, k, t, idx, deadline, n_trials=5000)

    # High-precision polish: repeated removal passes with increasing trial counts
    if time.time() < deadline - 0.5:
        for _ in range(3):
            if time.time() > deadline - 0.5:
                break
            blocks = _improve_blocks(blocks, v, k, t, idx, deadline)
            blocks = _random_removal_pass(blocks, v, k, t, idx, deadline, n_trials=10000)

    return blocks
=======
def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # Build index and reverse map
    idx, total, reverse = _build_index(v, t)

    # Fast greedy construction (returns precomputed structures for reuse)
    blocks, all_ksubs, all_bitsets, tsub_to_ksubs = _greedy_construct(v, k, t, idx, total, reverse, deadline)

    # Deterministic improvement pass
    if time.time() < deadline - 10:
        blocks = _improve_blocks(blocks, v, k, t, idx, deadline)

    # Randomized removal pass
    if time.time() < deadline - 8:
        blocks = _random_removal_pass(blocks, v, k, t, idx, deadline)

    # Block swap optimization (reuse precomputed structures)
    if time.time() < deadline - 6:
        blocks = _block_swap(blocks, v, k, t, idx, deadline, all_ksubs, all_bitsets, tsub_to_ksubs)

    # Simulated annealing (reuse precomputed structures)
    if time.time() < deadline - 3:
        blocks = _simulated_annealing(blocks, v, k, t, idx, deadline, all_ksubs, all_bitsets, tsub_to_ksubs)

    # Final cleanup pass
    if time.time() < deadline - 1:
        blocks = _improve_blocks(blocks, v, k, t, idx, deadline)
        blocks = _random_removal_pass(blocks, v, k, t, idx, deadline, n_trials=5000)

    # High-precision polish: repeated removal passes with increasing trial counts
    if time.time() < deadline - 0.5:
        for _ in range(3):
            if time.time() > deadline - 0.5:
                break
            blocks = _improve_blocks(blocks, v, k, t, idx, deadline)
            blocks = _random_removal_pass(blocks, v, k, t, idx, deadline, n_trials=10000)

    return blocks
>>>>>>> REPLACE
