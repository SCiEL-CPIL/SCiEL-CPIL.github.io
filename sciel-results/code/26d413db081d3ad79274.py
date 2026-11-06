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
    
    # Fill states
    # k is input index 0..2^N-1
    # We can generate this faster using broadcasting if we reshape, 
    # but the loop is 262k * 18 ops, which is ~4.7M ops. In Python this might take ~0.5s.
    # Let's try to vectorize the fill.
    # states[w][k//64] has bit (k%64) set if (k>>w)&1.
    # This is tricky to vectorize fully without creating a huge (2^N, N) array first.
    # Let's stick to the loop but optimize it.
    
    # Actually, we can generate the patterns for 64 consecutive inputs at once.
    # For a block of 64 inputs, the bits for wire w are the bits of the indices 0..63 shifted by w?
    # No, the inputs are 0..2^N-1.
    # Let's just use the loop, it's likely under 1 second.
    
    for k in range(2**N):
        block_idx = k >> 6
        bit_idx = k & 63
        mask = 1 << bit_idx
        for w in range(N):
            if (k >> w) & 1:
                states[w][block_idx] |= mask

    def _fast_verify(comparators):
        current = [s.copy() for s in states]
        for i, j in comparators:
            a = current[i]
            b = current[j]
            current[i] = a & b
            current[j] = a | b
        
        # Check sorted
        # A network is valid if for all inputs, the output is sorted.
        # For Boolean inputs, sorted means 0s then 1s.
        # This implies current[w] is a subset of current[w+1] in terms of bits?
        # No. If input is 0011, output is 0011.
        # Bit w is 0 for first k inputs, 1 for rest.
        # So current[w] should be "less than or equal" to current[w+1] in the sense that
        # if current[w] has a 1, current[w+1] must have a 1?
        # Yes. If wire w is 1, then wire w+1 must be 1 (since 0s come first).
        # So we check: current[w] & ~current[w+1] == 0 for all w.
        for w in range(N - 1):
            if np.any(current[w] & ~current[w + 1]):
                return False
        return True

    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    if not _fast_verify(best_network):
        return best_network

    current_network = list(best_network)
    
    def greedy_reduce(net):
        # One pass removal
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
                else:
                    idx += 1
        return net

    current_network = greedy_reduce(current_network)
    if len(current_network) < best_len:
        best_network = current_network
        best_len = len(current_network)

    np.random.seed(42)
    
    # Local Search
    while time.time() < deadline - 10:
        # Move 1: Swap adjacent
        if len(current_network) > 1:
            idx = np.random.randint(len(current_network) - 1)
            trial = current_network.copy()
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _fast_verify(trial):
                current_network = trial
                # Try to reduce after swap
                reduced = greedy_reduce(current_network)
                if len(reduced) < len(current_network):
                    current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)

        # Move 2: Mutation (change wire pair)
        if len(current_network) > 0 and time.time() < deadline - 10:
            idx = np.random.randint(len(current_network))
            i, j = current_network[idx]
            # Pick new pair
            a, b = np.random.choice(N, 2, replace=False)
            if a > b: a, b = b, a
            if a == i and b == j: continue
            trial = current_network.copy()
            trial[idx] = (a, b)
            if _fast_verify(trial):
                current_network = trial
                reduced = greedy_reduce(current_network)
                if len(reduced) < len(current_network):
                    current_network = reduced
                if len(current_network) < best_len:
                    best_network = list(current_network)
                    best_len = len(current_network)
                    
        # Move 3: Insert-Delete (2-opt style)
        # Insert a random comparator at a random position, then try to remove
        if len(current_network) < 200 and time.time() < deadline - 10:
             idx = np.random.randint(len(current_network) + 1)
             a, b = np.random.choice(N, 2, replace=False)
             if a > b: a, b = b, a
             trial = current_network[:idx] + [(a,b)] + current_network[idx:]
             # Don't verify trial directly, just try to reduce it
             # If it reduces to something shorter than current, it's a good move
             reduced = greedy_reduce(trial)
             if len(reduced) < len(current_network):
                 current_network = reduced
                 if len(current_network) < best_len:
                     best_network = list(current_network)
                     best_len = len(current_network)
             # If it didn't reduce enough, we might still want to keep the current network
             # The greedy_reduce on trial might have removed the inserted one and some others,
             # but if the result is not shorter, we discard.

    return best_network
