"""Size-efficient sorting network search on 17 wires."""
import time
import random

N = 17
TOTAL_BITS = 1 << N  # 131072

# EVOLVE-BLOCK-START

def _apply_comparator(bitmask, i, j):
    """Apply compare-exchange to bitmask: min to i, max to j."""
    bit_i = (bitmask >> i) & 1
    bit_j = (bitmask >> j) & 1
    if bit_i == bit_j:
        return bitmask
    if bit_i > bit_j:
        # swap: set i to 0, j to 1
        bitmask ^= (1 << i) | (1 << j)
    else:
        # already sorted: i=0, j=1 (no change needed)
        pass
    return bitmask


def _is_sorted_bitmask(bitmask):
    """Check if bitmask is sorted (all 1s before all 0s, i.e., 0...01...1)."""
    # Sorted means: there exists k such that bits 0..k-1 are 0 and bits k..N-1 are 1
    # Equivalently, the bitmask is of the form 0...01...1
    # This means: for all i < j, if bit_j is 0 then bit_i must be 0
    # Simpler: find the highest set bit and lowest set bit; all bits in between must be set
    if bitmask == 0:
        return True
    if bitmask == TOTAL_BITS - 1:
        return True
    # Find position of lowest set bit
    low = (bitmask & -bitmask).bit_length() - 1
    # Find position of highest set bit
    high = bitmask.bit_length() - 1
    # All bits from low to high must be set
    mask_range = ((1 << (high - low + 1)) - 1) << low
    return (bitmask & mask_range) == mask_range


def _verify_network(comparators):
    """Verify network sorts all 2^17 inputs using bitmasks."""
    for b in range(TOTAL_BITS):
        state = b
        for (i, j) in comparators:
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1
            if bit_i == bit_j:
                continue
            if bit_i > bit_j:
                state ^= (1 << i) | (1 << j)
        if not _is_sorted_bitmask(state):
            return False
    return True


def _verify_partial(comparators, test_bits=None):
    """Faster partial verification on a subset of bitmasks."""
    if test_bits is None:
        test_bits = list(range(0, TOTAL_BITS, 4))  # sample every 4th
    for b in test_bits:
        state = b
        for (i, j) in comparators:
            bit_i = (state >> i) & 1
            bit_j = (state >> j) & 1
            if bit_i == bit_j:
                continue
            if bit_i > bit_j:
                state ^= (1 << i) | (1 << j)
        if not _is_sorted_bitmask(state):
            return False
    return True


def _make_odd_even_transposition():
    """Generate odd-even transposition network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _make_batcher_odd_even():
    """Batcher's odd-even merge sort network for N wires."""
    # For 17 wires, use a recursive construction
    # Start with simple approach: merge sort network
    def merge_sort_network(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        mid = n // 2
        left = merge_sort_network(mid)
        right = merge_sort_network(n - mid)
        # Shift right
        right = [(i + mid, j + mid) for (i, j) in right]
        # Merge
        merge = _odd_even_merge(mid, n - mid)
        # Shift merge
        merge = [(i, j) for (i, j) in merge]
        return left + right + merge

    def _odd_even_merge(n1, n2):
        """Odd-even merge network for merging two sorted sequences."""
        total = n1 + n2
        if total <= 2:
            return [(0, 1)] if total == 2 else []
        if total == 3:
            return [(0, 1), (1, 2)]
        # Odd-even merge
        comps = []
        # First pass: compare even-indexed pairs
        for i in range(0, total - 1, 2):
            comps.append((i, i + 1))
        # Second pass: compare odd-indexed pairs
        for i in range(1, total - 1, 2):
            comps.append((i, i + 1))
        return comps

    return merge_sort_network(N)


def _local_search(comparators, deadline, rng):
    """Local search: try removing, adding, swapping comparators."""
    best = list(comparators)
    best_count = len(best)
    
    # Precompute for speed
    def verify_full(comps):
        for b in range(TOTAL_BITS):
            state = b
            for (i, j) in comps:
                bit_i = (state >> i) & 1
                bit_j = (state >> j) & 1
                if bit_i == bit_j:
                    continue
                if bit_i > bit_j:
                    state ^= (1 << i) | (1 << j)
            if not _is_sorted_bitmask(state):
                return False
        return True

    # Verify initial
    if not verify_full(best):
        return best

    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() < deadline:
        improved = False
        
        # Try removing each comparator
        for idx in range(len(best)):
            if time.time() >= deadline:
                break
            candidate = best[:idx] + best[idx+1:]
            if verify_full(candidate):
                best = candidate
                best_count = len(best)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping adjacent comparators
        for idx in range(len(best) - 1):
            if time.time() >= deadline:
                break
            candidate = best[:idx] + [best[idx+1], best[idx]] + best[idx+2:]
            if verify_full(candidate):
                best = candidate
                improved = True
                break
        
        if improved:
            continue
        
        # Try replacing a comparator with a different one
        for idx in range(len(best)):
            if time.time() >= deadline:
                break
            old_i, old_j = best[idx]
            # Try a random different pair
            for _ in range(20):
                new_i, new_j = rng.choice(all_pairs)
                if new_i == old_i and new_j == old_j:
                    continue
                candidate = best[:idx] + [(new_i, new_j)] + best[idx+1:]
                if verify_full(candidate):
                    best = candidate
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Try inserting a new comparator at a random position
        if time.time() < deadline - 2:
            pos = rng.randint(0, len(best))
            new_i, new_j = rng.choice(all_pairs)
            candidate = best[:pos] + [(new_i, new_j)] + best[pos:]
            # Only accept if it still sorts (we're not trying to add, just exploring)
            # Actually, adding won't help minimize, so skip this
            pass
        
        # If no improvement, try a random perturbation
        if time.time() >= deadline:
            break
        if len(best) > 0 and rng.random() < 0.3:
            # Randomly remove a few and re-add
            n_remove = min(3, len(best))
            remove_idx = rng.sample(range(len(best)), n_remove)
            candidate = [c for i, c in enumerate(best) if i not in set(remove_idx)]
            if verify_full(candidate):
                best = candidate
                best_count = len(best)
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    rng = random.Random(42)
    
    # Start with a reasonable initial network
    # Use a modified odd-even transposition or a known small network
    # For 17 wires, let's start with a merge-sort-like network
    
    # Initial: odd-even transposition (136 comparators)
    best = _make_odd_even_transposition()
    
    # Quick verification
    if not _verify_network(best):
        # Fallback: identity
        return []
    
    best_count = len(best)
    
    # Local search to minimize
    best = _local_search(best, deadline, rng)
    
    # Final verification
    if not _verify_network(best):
        # Return the odd-even transposition as fallback
        return _make_odd_even_transposition()
    
    return best

# EVOLVE-BLOCK-END
