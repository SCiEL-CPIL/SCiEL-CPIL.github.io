import time
import numpy as np

N = 18
TIME_LIMIT = 60  # Increased from 55 to allow more search time

def _odd_even_transposition():
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    num_blocks = 2**N // 64
    states = [np.zeros(num_blocks, dtype=np.uint64) for _ in range(N)]
    
    # Vectorized initialization of states
    # For each wire w, we need to set bit (k%64) in block (k//64) 
    # if (k>>w)&1 is 1.
    # This is equivalent to: for each block b, the bits set are those 
    # where the global index k = b*64 + bit has the w-th bit set.
    # k = b*64 + bit.
    # (b*64 + bit) >> w & 1.
    # If w < 6, the bit depends only on 'bit' and potentially b if w >= 6? 
    # No, 64 is 2^6. So bits 0-5 depend only on 'bit'.
    # Bits 6+ depend on 'b'.
    
    # Let's generate the pattern for one block of 64 inputs for each wire w.
    # Actually, it's easier to just loop over blocks and compute the mask for that block.
    # For a given block index b, the inputs are k in [b*64, (b+1)*64).
    # For wire w, we want a 64-bit mask where bit i is 1 if ((b*64+i)>>w)&1 == 1.
    
    # Precompute masks for each block? That's 4096 blocks * 18 wires = 73k masks.
    # Or just compute on the fly.
    
    # Let's try a hybrid: generate the base pattern for bits 0-5, and handle bits 6-17.
    # Actually, the loop in the current program is:
    # for k in range(2**N): ...
    # This is 262k iterations. Each iteration does 18 checks.
    # 4.7M operations. Python does ~10-50M ops/sec. So ~0.1-0.5s.
    # This is acceptable. Let's stick to the loop but optimize it slightly if needed.
    # However, we can vectorize the bit extraction.
    
    # Let's try to generate the states using numpy broadcasting on a smaller scale.
    # We can create an array of all indices 0..2^N-1, but that's 262k ints.
    # indices = np.arange(2**N, dtype=np.uint32)
    # Then for each w, mask = (indices >> w) & 1
    # Then we need to pack these bits into uint64 blocks.
    # This packing is the hard part.
    
    # Let's stick to the loop for simplicity and reliability, but ensure it's fast.
    # The loop is:
    for k in range(2**N):
        block_idx = k >> 6
        bit_idx = k & 63
        mask = 1 << bit_idx
        # Unroll the inner loop for speed?
        # for w in range(N):
        #    if (k >> w) & 1:
        #        states[w][block_idx] |= mask
        
        # We can compute the bits for all w at once?
        # k is an integer.
        # bits = [(k >> w) & 1 for w in range(N)]
        # This is still a loop.
        
        # Let's just use the loop. It's likely fast enough.
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
