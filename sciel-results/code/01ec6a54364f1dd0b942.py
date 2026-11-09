import time

N = 16

# EVOLVE-BLOCK-START
def _apply_comparator(state, i, j):
    bi = (state >> i) & 1
    bj = (state >> j) & 1
    if bi < bj:
        state |= (1 << i)
        state &= ~(1 << j)
    return state

def _apply_network(state, network):
    for i, j in network:
        state = _apply_comparator(state, i, j)
    return state

def _is_sorted(state):
    if state == 0 or state == (1 << N) - 1:
        return True
    low = (state & -state).bit_length() - 1
    mask = ((1 << N) - 1) ^ ((1 << low) - 1)
    return state == mask

def _verify(network):
    for mask in range(1 << N):
        if not _is_sorted(_apply_network(mask, network)):
            return False
    return True

def _initial_network():
    return [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    ]

def construct():
    start_time = time.time()
    deadline = 85.0
    
    network = _initial_network()
    
    # Greedy removal
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > deadline:
                break
            candidate = network[:idx] + network[idx+1:]
            if _verify(candidate):
                network = candidate
                improved = True
                break
                
    return network
# EVOLVE-BLOCK-END
