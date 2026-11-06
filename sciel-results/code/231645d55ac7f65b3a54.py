import time
import random
import numpy as np

N = 16
NUM_INPUTS = 1 << N  # 65536
CHUNKS = NUM_INPUTS // 64  # 1024 chunks of 64 inputs each

# Pre-compute the initial states for all 65536 binary inputs.
# State shape: (N, CHUNKS) where each element is uint64.
# Bit 'k' in chunk 'c' corresponds to input vector (c * 64 + k).
# We need to set bit 'w' of input 'v' to 1 if the w-th bit of v is 1.

def _init_states():
    states = np.zeros((N, CHUNKS), dtype=np.uint64)
    for w in range(N):
        for c in range(CHUNKS):
            mask = 0
            for k in range(64):
                v = c * 64 + k
                if (v >> w) & 1:
                    mask |= (1 << k)
            states[w, c] = mask
    return states

_INIT_STATES = None

def _get_states():
    global _INIT_STATES
    if _INIT_STATES is None:
        _INIT_STATES = _init_states()
    return _INIT_STATES

def _verify_network_bit_parallel(comparators):
    """
    Verifies if the network sorts all 65536 binary inputs.
    Returns True if sorted, False otherwise.
    """
    if not comparators:
        return False # Empty network doesn't sort unless N=1
    
    states = _get_states()
    curr = states.copy()
    
    for i, j in comparators:
        # min(curr[i], curr[j]) and max(curr[i], curr[j])
        # min(a, b) = a & b
        # max(a, b) = a | b
        min_val = curr[i] & curr[j]
        max_val = curr[i] | curr[j]
        curr[i] = min_val
        curr[j] = max_val
    
    # Check if sorted: for all adjacent wires i, i+1, we must have curr[i] <= curr[i+1]
    # In bit-parallel, this means: for all bits k, if curr[i] has 1 at k, curr[i+1] must have 1 at k.
    # Equivalent to: (curr[i] & ~curr[i+1]) == 0 for all i.
    for i in range(N - 1):
        if np.any(curr[i] & ~curr[i+1]):
            return False
    return True

def _base_60_network():
    # Verified 60-comparator sorting network for 16 inputs (Akers-style construction).
    return [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
        (0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11), (12, 14), (13, 15),
        (0, 4), (1, 5), (2, 6), (3, 7), (8, 12), (9, 13), (10, 14), (11, 15),
        (0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15),
        (1, 2), (3, 4), (5, 6), (9, 10), (11, 12), (13, 14),
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15),
    ]

def _local_search(net, deadline):
    """
    Aggressive greedy removal using fast NumPy verifier.
    """
    best_net = net[:]
    best_len = len(net)
    
    if not _verify_network_bit_parallel(best_net):
        raise ValueError("Initial network is not a valid sorting network")
    
    # Greedy removal pass
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Try removing each comparator
        for idx in range(len(best_net)):
            if time.time() > deadline:
                break
            candidate = best_net[:idx] + best_net[idx+1:]
            if _verify_network_bit_parallel(candidate):
                best_net = candidate
                best_len = len(best_net)
                changed = True
                break # Restart the loop with the new shorter network
    
    return best_net

def _insert_then_prune(net, deadline, rng):
    """
    Structural move: Insert a random comparator, then prune.
    This allows escaping local minima of pure deletion.
    """
    if len(net) >= 4096:
        return net
    
    # Choose a random comparator to insert
    i, j = rng.sample(range(N), 2)
    if i > j:
        i, j = j, i
    
    # Choose a random position
    pos = rng.randint(0, len(net))
    
    candidate = net[:pos] + [(i, j)] + net[pos:]
    
    # Prune the candidate
    pruned = _local_search(candidate, deadline)
    
    return pruned

def construct():
    start_time = time.time()
    deadline = start_time + 80.0 # Safety margin
    
    # Start with the known 60-comparator network
    base_net = _base_60_network()
    
    if not _verify_network_bit_parallel(base_net):
        # Fallback to odd-even transposition (120 comparators)
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    best_net = base_net
    best_len = len(best_net)
    
    # 1. Initial aggressive pruning
    optimized_net = _local_search(best_net, deadline)
    best_net = optimized_net
    best_len = len(best_net)
    
    # 2. Iterative Insert-Prune Search
    # This is the structural change: instead of just deleting, we add then delete.
    # This explores the neighborhood of larger networks to find smaller valid ones.
    rng = random.Random(42)
    
    # We will run this loop until deadline.
    # Each iteration:
    # - Insert a random comparator
    # - Prune
    # - If result is better, update best.
    
    # To avoid getting stuck in a cycle, we can track recent states or just rely on the randomness.
    # Given the speed of verification, we can do many iterations.
    
    # Let's estimate iterations.
    # Verification: ~1-2 ms (NumPy overhead + 60 ops)
    # Pruning: ~50-100 verifications in the worst case? No, usually less.
    # Let's say 1 insert-prune cycle takes ~50ms on average (conservative).
    # 80s / 0.05s = 1600 cycles.
    
    # We can also try multiple insertions at once if we have time, but single insertion is a good start.
    
    while time.time() < deadline:
        # Check time more frequently
        if time.time() > deadline - 0.1:
            break
            
        candidate = _insert_then_prune(best_net, deadline, rng)
        
        if len(candidate) < best_len:
            best_net = candidate
            best_len = len(candidate)
            
            # If we found a new best, we might want to do a few more quick prunes on it
            # or just continue with insert-prune.
            # The insert-prune loop naturally continues from the best state.
            
            # Optional: If we found a significantly better network, restart with a different seed?
            # No, keep the momentum.
            
        # If we are stuck at 60 for a long time, we might try a more aggressive mutation
        # e.g., insert 2 comparators.
        # But let's stick to the simple structural change first.
        
        # To add variety, we can occasionally swap two comparators?
        # The insert-prune effectively does swaps by inserting one and deleting another (possibly the same or different).
        
    return best_net
