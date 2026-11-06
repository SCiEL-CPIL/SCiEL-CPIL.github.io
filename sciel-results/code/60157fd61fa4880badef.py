import time
import numpy as np

N = 18
TIME_LIMIT = 55  # Safety margin

def _odd_even_transposition():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    num_blocks = 2**N // 64
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    # Vectorized fill of states
    # For each wire w, bit b of block k is set if ((k*64+b) >> w) & 1
    # We can generate this more efficiently by noting the pattern
    # For wire w, within a block of 64 inputs, the pattern repeats with period 2^(w+1)
    # Let's use a simpler approach: generate all 2^N inputs and extract bits
    
    # Actually, let's use a more efficient method:
    # For wire w, the bit pattern for inputs 0..2^N-1 is a repeating pattern
    # of 2^w zeros followed by 2^w ones, repeated
    # We can generate this using numpy
    
    for w in range(N):
        # Pattern for wire w: 2^w zeros, 2^w ones, repeated
        period = 1 << (w + 1)
        half = 1 << w
        # Number of full periods
        num_full = (2**N) // period
        rem = (2**N) % period
        
        # Generate one period
        period_pattern = np.zeros(period, dtype=np.uint8)
        period_pattern[half:] = 1
        
        # Tile for full periods
        if num_full > 0:
            pattern = np.tile(period_pattern, num_full)
        else:
            pattern = np.array([], dtype=np.uint8)
        
        # Add remainder
        if rem > 0:
            pattern = np.concatenate([pattern, period_pattern[:rem]])
        
        # Now pack into uint64 blocks
        # pattern has 2^N elements (0s and 1s)
        # Reshape to (num_blocks, 64) and pack bits
        pattern_reshaped = pattern.reshape(num_blocks, 64)
        # Pack bits: bit 0 is LSB
        # We want bit b of states[w][k] to be pattern_reshaped[k, b]
        # So we need to reverse the bit order within each 64-bit word
        # Actually, let's just compute it directly
        
        # For each block, compute the uint64 value
        # bit b corresponds to 2^b
        bit_values = (1 << np.arange(64, dtype=np.uint64))
        states[w] = (pattern_reshaped.astype(np.uint64) * bit_values).sum(axis=1, dtype=np.uint64)

    def _fast_verify(comparators):
        current = [s.copy() for s in states]
        for i, j in comparators:
            a = current[i]
            b = current[j]
            current[i] = a & b
            current[j] = a | b
        
        # Check sorted: current[w] & ~current[w+1] should be all zeros
        for w in range(N - 1):
            if np.any(current[w] & ~current[w + 1]):
                return False
        return True

    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    if not _fast_verify(best_network):
        return best_network

    current_network = list(best_network)
    
    def greedy_reduce(net, max_time=None):
        if max_time is None:
            max_time = deadline - 5
        changed = True
        while changed and time.time() < max_time:
            changed = False
            idx = 0
            while idx < len(net):
                if time.time() > max_time:
                    return net
                trial = net[:idx] + net[idx+1:]
                if _fast_verify(trial):
                    net = trial
                    changed = True
                else:
                    idx += 1
        return net

    current_network = greedy_reduce(current_network)
    if len(current_network) < best_len:
        best_network = current_network
        best_len = len(current_network)

    np.random.seed(42)
    
    # Local Search with improved time management
    while time.time() < deadline - 10:
        elapsed = time.time() - start_time
        remaining = deadline - 10 - time.time()
        if remaining <= 0:
            break
        
        # Choose move type based on remaining time
        move_type = np.random.choice(3, p=[0.4, 0.3, 0.3])
        
        # Move 1: Swap adjacent
        if move_type == 0 and len(current_network) > 1:
            idx = np.random.randint(len(current_network) - 1)
            trial = current_network.copy()
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _fast_verify(trial):
                current_network = trial
                # Try to reduce after swap with limited time
                reduce_time = min(remaining * 0.3, 2.0)
                reduced = greedy_reduce(current_network, max_time=time.time() + reduce_time)
                if len(reduced) < len(current_network):
                    current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)

        # Move 2: Mutation (change wire pair)
        elif move_type == 1 and len(current_network) > 0:
            idx = np.random.randint(len(current_network))
            i, j = current_network[idx]
            # Pick new pair - bias towards nearby wires for locality
            if np.random.random() < 0.5:
                # Nearby wire
                new_i = i
                new_j = j
                if np.random.random() < 0.5:
                    # Change i
                    new_i = max(0, min(N-2, i + np.random.choice([-1, 1])))
                    if new_i == j:
                        new_i = (new_i + 1) % N
                    if new_i > new_j:
                        new_i, new_j = new_j, new_i
                else:
                    # Change j
                    new_j = min(N-1, max(0, j + np.random.choice([-1, 1])))
                    if new_j == i:
                        new_j = (new_j + 1) % N
                    if new_i > new_j:
                        new_i, new_j = new_j, new_i
            else:
                # Random pair
                a, b = np.random.choice(N, 2, replace=False)
                if a > b: a, b = b, a
                new_i, new_j = a, b
            
            if new_i == i and new_j == j:
                continue
            trial = current_network.copy()
            trial[idx] = (new_i, new_j)
            if _fast_verify(trial):
                current_network = trial
                reduce_time = min(remaining * 0.3, 2.0)
                reduced = greedy_reduce(current_network, max_time=time.time() + reduce_time)
                if len(reduced) < len(current_network):
                    current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
                    
        # Move 3: Insert-Delete (2-opt style)
        elif move_type == 2 and len(current_network) < 200:
            idx = np.random.randint(len(current_network) + 1)
            a, b = np.random.choice(N, 2, replace=False)
            if a > b: a, b = b, a
            trial = current_network[:idx] + [(a,b)] + current_network[idx:]
            # Try to reduce with limited time
            reduce_time = min(remaining * 0.4, 3.0)
            reduced = greedy_reduce(trial, max_time=time.time() + reduce_time)
            if len(reduced) < len(current_network):
                current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)

    return best_network
