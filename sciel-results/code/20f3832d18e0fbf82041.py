"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _build_sorted_masks(n):
    """Build frozenset of all sorted masks (0^a * 1^b) for n bits."""
    masks = set()
    for k in range(n + 1):
        mask = ((1 << k) - 1) << (n - k)
        masks.add(mask)
    return frozenset(masks)

_SORTED_MASKS = _build_sorted_masks(N)
_SORTED_MASKS_SET = set(_SORTED_MASKS)

def _verify_network_numpy(comparators, n=N):
    """Verify network sorts all 2^n Boolean inputs using numpy vectorization."""
    if not comparators:
        # Empty network only sorts if n <= 1
        return n <= 1
    
    total = 1 << n
    # Create array of all input masks
    inputs = np.arange(total, dtype=np.uint32)
    # Apply each comparator to all inputs at once
    for i, j in comparators:
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        # Where bit_i == 0 and bit_j == 1, swap
        swap_mask = (bit_i == 0) & (bit_j == 1)
        # XOR both bits where swap needed
        toggle = (1 << i) | (1 << j)
        inputs = np.where(swap_mask, inputs ^ toggle, inputs)
    
    # Check if all results are in sorted mask set
    # Convert to python ints and check membership
    results = inputs.tolist()
    sorted_set = _SORTED_MASKS_SET
    for r in results:
        if r not in sorted_set:
            return False
    return True

def _fast_verify_network(comparators, n=N):
    """Verify network sorts all 2^n Boolean inputs using bit manipulation (fallback)."""
    sorted_set = _SORTED_MASKS
    for mask in range(1 << n):
        state = mask
        for i, j in comparators:
            if (state >> i) & 1 == 0 and (state >> j) & 1 == 1:
                state ^= (1 << i) | (1 << j)
        if state not in sorted_set:
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
    deadline = start_time + 85  # 5 second safety margin before 90s kill
    
    # Start with odd-even transposition
    best = _odd_even_transposition(N)
    best_len = len(best)
    
    if time.time() > deadline:
        return best
    
    random.seed(42)
    current = list(best)
    
    # Verify initial
    if not _verify_network_numpy(current):
        return best
    
    # Local search with vectorized verification
    max_iterations = 500000
    consecutive_failures = 0
    max_consecutive_failures = 5000
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        if len(current) <= 70:
            break
        if consecutive_failures > max_consecutive_failures:
            # Restart with fresh random seed
            random.seed(random.randint(0, 2**32 - 1))
            consecutive_failures = 0
            current = list(best)
        
        # Try random modification
        mod_type = random.random()
        
        if mod_type < 0.55 and len(current) > 1:
            # Try removing a random comparator
            idx = random.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_numpy(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
                    # Greedy cascade: try to remove more
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        if len(current) <= 70:
                            break
                        idx2 = random.randint(0, len(current) - 1)
                        candidate2 = current[:idx2] + current[idx2+1:]
                        if _verify_network_numpy(candidate2):
                            current = candidate2
                            if len(current) < best_len:
                                best = list(current)
                                best_len = len(current)
                        else:
                            break
            else:
                consecutive_failures += 1
        
        elif mod_type < 0.75:
            # Try swapping two comparators
            if len(current) < 2:
                continue
            i, j = random.sample(range(len(current)), 2)
            candidate = list(current)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if _verify_network_numpy(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1
        
        elif mod_type < 0.90:
            # Try replacing a comparator with a different one
            if len(current) == 0:
                continue
            idx = random.randint(0, len(current) - 1)
            # Generate random valid comparator, biased toward adjacent
            if random.random() < 0.7:
                a = random.randint(0, N - 2)
                b = a + 1
            else:
                while True:
                    a = random.randint(0, N - 1)
                    b = random.randint(0, N - 1)
                    if a != b:
                        if a > b:
                            a, b = b, a
                        break
            candidate = list(current)
            candidate[idx] = (a, b)
            if _verify_network_numpy(candidate):
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
            if _verify_network_numpy(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1
    
    return best

# EVOLVE-BLOCK-END
