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
        
        # Check sorted: for all w, if wire w has a 1, wire w+1 must have a 1
        # i.e., current[w] & ~current[w+1] must be all zeros
        for w in range(N - 1):
            if np.any(current[w] & ~current[w + 1]):
                return False
        return True

    best_network = _odd_even_transposition()
    best_len = len(best_network)
    
    if not _fast_verify(best_network):
        return best_network

    current_network = list(best_network)
    
    def greedy_reduce(net, time_budget=None):
        # Try to remove comparators; use a time budget if provided
        if time_budget is None:
            time_budget = deadline - 5
        changed = True
        while changed and time.time() < time_budget:
            changed = False
            # Try removals from the end first (often more removable), then middle
            for idx in range(len(net) - 1, -1, -1):
                if time.time() > time_budget:
                    return net
                trial = net[:idx] + net[idx+1:]
                if _fast_verify(trial):
                    net = trial
                    changed = True
                    break  # Restart from end after a removal
        return net

    current_network = greedy_reduce(current_network)
    if len(current_network) < best_len:
        best_network = current_network
        best_len = len(current_network)

    np.random.seed(42)
    
    # Local Search with multiple move types
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
            # Pick new pair - bias toward nearby wires for better locality
            a = np.random.randint(N)
            b = np.random.randint(N)
            if a == b:
                b = (a + 1) % N
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
             a = np.random.randint(N)
             b = np.random.randint(N)
             if a == b:
                 b = (a + 1) % N
             if a > b: a, b = b, a
             trial = current_network[:idx] + [(a,b)] + current_network[idx:]
             # Try to reduce; if it reduces to something shorter, accept
             reduced = greedy_reduce(trial)
             if len(reduced) < len(current_network):
                 current_network = reduced
                 if len(current_network) < best_len:
                     best_network = list(current_network)
                     best_len = len(current_network)

        # Move 4: Swap two non-adjacent comparators (position swap)
        if len(current_network) > 2 and time.time() < deadline - 10:
            idx1 = np.random.randint(len(current_network))
            idx2 = np.random.randint(len(current_network))
            if idx1 != idx2:
                trial = current_network.copy()
                trial[idx1], trial[idx2] = trial[idx2], trial[idx1]
                if _fast_verify(trial):
                    current_network = trial
                    reduced = greedy_reduce(current_network)
                    if len(reduced) < len(current_network):
                        current_network = reduced
                    if len(current_network) < best_len:
                        best_network = list(current_network)
                        best_len = len(current_network)

    return best_network
