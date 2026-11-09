import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin

    def verify_network(comparators):
        masks = np.arange(1 << N, dtype=np.uint8)
        wires = np.zeros((1 << N, N), dtype=np.uint8)
        for i in range(N):
            wires[:, i] = (masks >> i) & 1
            
        for i, j in comparators:
            old_i = wires[:, i]
            old_j = wires[:, j]
            wires[:, i] = old_i & old_j
            wires[:, j] = old_i | old_j
            
        diffs = wires[:, 1:] - wires[:, :-1]
        return not np.any(diffs < 0)

    # Candidate: The network from the prompt
    candidate = [
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (1, 2), (5, 6), (9, 10), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (2, 4), (3, 5), (6, 8), (7, 9), (10, 12), (11, 13),
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)
    ]

    if verify_network(candidate):
        network = candidate
    else:
        # Fallback to Odd-Even Transposition Sort (120 comparators)
        network = []
        for i in range(N):
            if i % 2 == 0:
                for j in range(0, N-1, 2):
                    network.append((j, j+1))
            else:
                for j in range(1, N-1, 2):
                    network.append((j, j+1))

    # Greedy Pruning
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(network)):
            new_net = network[:i] + network[i+1:]
            if verify_network(new_net):
                network = new_net
                improved = True
                break

    return network
# EVOLVE-BLOCK-END
