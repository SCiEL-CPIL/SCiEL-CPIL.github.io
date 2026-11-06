import time
import numpy as np

N = 18
TIME_LIMIT = 50  # Safety margin for 90s limit

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    # Pre-compute bit-parallel states for fast verification
    # Number of 64-bit blocks needed for 2^18 inputs
    num_blocks = 2**N // 64
    
    # states[w] is an array of num_blocks uint64s
    # Bit b of states[w][k] is the value of wire w for input (k*64 + b)
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    # Fill states efficiently
    # We can generate the bit patterns using bit manipulation on the index
    # k is the input index (0 to 2^18 - 1)
    # Wire w is 1 if (k >> w) & 1 is 1
    # We fill blocks of 64 inputs at a time
    for k in range(2**N):
        block_idx = k >> 6
        bit_idx = k & 63
        mask = 1 << bit_idx
        for w in range(N):
            if (k >> w) & 1:
                states[w][block_idx] |= mask

    def _fast_verify(comparators):
        """
        Verify network using bit-parallel simulation.
        Returns True if network sorts all inputs.
        """
        # Copy states
        current = [s.copy() for s in states]
        
        for i, j in comparators:
            a = current[i]
            b = current[j]
            # min goes to i, max goes to j
            current[i] = a & b
            current[j] = a | b
            
        # Check if sorted
        for w in range(N - 1):
            if np.any(current[w] & ~current[w + 1]):
                return False
        return True

    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    # Verify seed
    if not _fast_verify(best_network):
        return best_network

    current_network = list(best_network)
    
    def greedy_reduce(net):
        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            idx = 0
            while idx < len(net):
                if time.time() > deadline - 5:
                    return net
                trial = net[:idx] + net[idx+1:]
                if _fast_verify(trial):
                    net = trial
                    changed = True
                    # Do not increment idx, check the new element at idx
                else:
                    idx += 1
        return net

    current_network = greedy_reduce(current_network)
    if len(current_network) < best_len:
        best_network = current_network
        best_len = len(current_network)

    # Phase 2: Random Restart Local Search
    np.random.seed(42)
    
    while time.time() < deadline - 10:
        if len(current_network) > 1:
            # Random Swap
            idx = np.random.randint(len(current_network) - 1)
            trial = current_network.copy()
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _fast_verify(trial):
                current_network = trial
                current_network = greedy_reduce(current_network)
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
        else:
            break

        if len(current_network) > 0 and time.time() < deadline - 10:
            # Wire Mutation
            idx = np.random.randint(len(current_network))
            i, j = current_network[idx]
            a, b = np.random.choice(N, 2, replace=False)
            if a > b:
                a, b = b, a
            if a == i and b == j:
                continue
            trial = current_network.copy()
            trial[idx] = (a, b)
            if _fast_verify(trial):
                current_network = trial
                current_network = greedy_reduce(current_network)
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)

    return best_network
# EVOLVE-BLOCK-END
