N = 18
import time
import random

# EVOLVE-BLOCK-START

# Precompute initial wire states for bit-parallel verification
# wires[w] is an integer where bit k is 1 if input k has 1 on wire w
# Input k is represented by its binary value. Bit w of k is the value on wire w initially.
_M = 1 << N
_wires_init = [0] * N
for w in range(N):
    # Construct the mask for bit w across all 2^N inputs
    # Bit w is 1 for k where (k >> w) & 1 == 1
    # This pattern repeats every 2^(w+1) with period 2^w.
    # We can build this efficiently.
    mask = 0
    # For k in range(_M):
    #     if (k >> w) & 1:
    #         mask |= (1 << k)
    # This loop is 2^18 * 18 ~ 4M ops, acceptable at startup.
    # But we can do it faster with bit manipulation or just let it run once.
    # Let's use a simple loop for clarity, it runs once.
    for k in range(_M):
        if (k >> w) & 1:
            mask |= (1 << k)
    _wires_init[w] = mask

def _verify_network_bit_parallel(comparators):
    """
    Verify if the network sorts all 2^N boolean inputs.
    Uses bit-parallelism: state is a list of N big integers.
    """
    if not comparators:
        return N <= 1
    
    # Copy initial state
    # List of big ints
    state = _wires_init[:]
    
    for i, j in comparators:
        # Find inputs where wire i is 1 and wire j is 0
        # These need swapping (min to i, max to j)
        swap_mask = state[i] & ~state[j]
        
        if swap_mask == 0:
            continue
            
        # Update wires i and j
        # new_i = (old_i & ~swap_mask) | (old_j & swap_mask)
        # new_j = (old_j & ~swap_mask) | (old_i & swap_mask)
        
        old_i = state[i]
        old_j = state[j]
        
        state[i] = (old_i & ~swap_mask) | (old_j & swap_mask)
        state[j] = (old_j & ~swap_mask) | (old_i & swap_mask)
        
    # Check if sorted
    # Sorted means for all adjacent wires w, w+1: NOT (wire w is 1 AND wire w+1 is 0)
    # bad_mask = 0
    # for w in range(N-1):
    #     bad_mask |= state[w] & ~state[w+1]
    # return bad_mask == 0
    
    # Optimization: Just check the last wire? No.
    # Check adjacent pairs.
    bad = 0
    for w in range(N - 1):
        bad |= state[w] & ~state[w + 1]
        if bad:
            return False
            
    return True

def _odd_even_transposition(n=N):
    """Generate odd-even transposition sorting network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 10s safety margin
    
    # Start with odd-even transposition
    best = _odd_even_transposition(N)
    best_len = len(best)
    
    if not _verify_network_bit_parallel(best):
        # Should never happen, but handle it
        return []
        
    random.seed(42)
    current = list(best)
    
    max_iterations = 1000000
    consecutive_failures = 0
    max_consecutive_failures = 5000
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        if len(current) <= 70:
            break
        if consecutive_failures > max_consecutive_failures:
            # Restart with fresh random seed from best
            random.seed(random.randint(0, 2**32 - 1))
            consecutive_failures = 0
            current = list(best)
        
        # Try random modification
        mod_type = random.random()
        
        if mod_type < 0.70 and len(current) > 1:
            # Try removing a random comparator
            idx = random.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_bit_parallel(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
                    # Greedy cascade: try to remove more
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        if len(current) <= 70:
                            break
                        idx2 = random.randint(0, len(current) - 1)
                        candidate2 = current[:idx2] + current[idx2+1:]
                        if _verify_network_bit_parallel(candidate2):
                            current = candidate2
                            if len(current) < best_len:
                                best = list(current)
                                best_len = len(current)
                        else:
                            break
            else:
                consecutive_failures += 1
        
        elif mod_type < 0.85:
            # Try swapping two comparators
            if len(current) < 2:
                continue
            i, j = random.sample(range(len(current)), 2)
            candidate = list(current)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if _verify_network_bit_parallel(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1
        
        elif mod_type < 0.95:
            # Try replacing a comparator with a different one
            if len(current) == 0:
                continue
            idx = random.randint(0, len(current) - 1)
            # Generate random valid comparator
            # Bias towards adjacent or short distance
            dist = random.randint(1, N - 1)
            if random.random() < 0.5:
                dist = random.randint(1, 3)
            a = random.randint(0, N - 1 - dist)
            b = a + dist
            candidate = list(current)
            candidate[idx] = (a, b)
            if _verify_network_bit_parallel(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1
        
        else:
            # Try batch removal: remove 2-3 random comparators at once
            if len(current) < 3:
                continue
            num_remove = random.randint(2, 3)
            if len(current) <= num_remove:
                continue
            remove_indices = set(random.sample(range(len(current)), num_remove))
            candidate = [c for idx, c in enumerate(current) if idx not in remove_indices]
            if _verify_network_bit_parallel(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1
    
    return best

# EVOLVE-BLOCK-END
