import numpy as np
import time
import random

N = 17
NUM_INPUTS = 1 << N
NUM_WORDS = NUM_INPUTS // 64

PACKED_INPUTS = None

def construct():
    global PACKED_INPUTS
    if PACKED_INPUTS is None:
        indices = np.arange(NUM_INPUTS, dtype=np.uint32)
        # Generate bits
        bits = (indices[:, None] >> np.arange(N, dtype=np.uint64)[None, :]) & 1
        # Reshape to (2048, 64, 17)
        bits_reshaped = bits.reshape(NUM_WORDS, 64, N).transpose(0, 2, 1)
        # Pack bits into uint64
        powers = 1 << np.arange(64, dtype=np.uint64)
        PACKED_INPUTS = (bits_reshaped * powers[None, None, :]).sum(axis=2).astype(np.uint64)

    start_time = time.time()
    deadline = start_time + 50.0 # 10s safety margin

    # Seed network: Odd-Even Transposition
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))

    def is_valid(network):
        if not network:
            return N == 1
        
        state = PACKED_INPUTS.copy()
        
        for i, j in network:
            min_val = state[:, i] & state[:, j]
            max_val = state[:, i] | state[:, j]
            state[:, i] = min_val
            state[:, j] = max_val

        for k in range(N - 1):
            if np.any(state[:, k] & ~state[:, k+1]):
                return False
        return True

    def reduce_network(network):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for idx in range(len(network)):
                if time.time() > deadline:
                    break
                candidate = network[:idx] + network[idx+1:]
                if is_valid(candidate):
                    network = candidate
                    changed = True
                    break
        return network

    # Initial reduction
    best_network = reduce_network(seed_network[:])
    best_cost = len(best_network)
    
    current_network = best_network[:]
    current_cost = best_cost
    
    random.seed(42)
    
    # Simple Local Search
    while time.time() < deadline:
        # Try to remove a random comparator
        if current_cost > 0 and random.random() < 0.5:
            idx = random.randint(0, len(current_network) - 1)
            candidate = current_network[:idx] + current_network[idx+1:]
            if is_valid(candidate):
                current_network = candidate
                current_cost -= 1
                if current_cost < best_cost:
                    best_network = current_network[:]
                    best_cost = current_cost
        else:
            # Try to swap or move
            if len(current_network) > 1:
                idx1 = random.randint(0, len(current_network) - 1)
                idx2 = random.randint(0, len(current_network) - 1)
                if idx1 != idx2:
                    candidate = current_network[:]
                    candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]
                    if is_valid(candidate):
                        # Optional: reduce
                        # candidate = reduce_network(candidate)
                        current_network = candidate
                        # current_cost remains same
            elif len(current_network) == 1:
                 # Try to move the single comparator
                 pass # Can't move a single item

    return best_network
