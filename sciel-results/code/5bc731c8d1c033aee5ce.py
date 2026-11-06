"""Size-efficient sorting networks on 17 wires - simulated annealing approach."""
import time
import random
import numpy as np

N = 17
NUM_INPUTS = 1 << N  # 131072
DEADLINE = 85.0  # safety margin from 90s


def _build_all_inputs():
    """Build all 131072 Boolean inputs as a bit-packed array for fast verification."""
    # Each input is a bitmask of N bits
    bits = np.arange(NUM_INPUTS, dtype=np.uint32)
    return bits


_ALL_BITS = _build_all_inputs()


def _expected_sorted_bits():
    """For each bitmask, the sorted output has the 1s in the lowest positions."""
    # For a bitmask with k ones, sorted output has ones in positions 0..k-1
    # We need to check: after sorting, wire w has 1 iff the count of 1s > N-1-w
    # i.e., sorted output = bitmask with k ones in the k lowest positions
    result = np.zeros(NUM_INPUTS, dtype=np.uint32)
    for k in range(N + 1):
        mask = ((1 << k) - 1)  # k ones in lowest positions
        # Find all inputs with exactly k ones
        # Use popcount
        popcounts = np.array([bin(x).count('1') for x in _ALL_BITS], dtype=np.uint8)
        result[popcounts == k] = mask
    return result


# Precompute popcounts for speed
def _popcount_array(arr):
    """Compute popcount for an array of uint32 values."""
    result = np.zeros(len(arr), dtype=np.uint8)
    for i in range(len(arr)):
        v = arr[i]
        c = 0
        while v:
            v &= v - 1
            c += 1
        result[i] = c
    return result


# More efficient popcount using lookup
def _popcount_fast(arr):
    """Fast popcount using 8-bit lookup table."""
    lookup = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lookup[i] = bin(i).count('1')
    b0 = (arr & 0xFF).astype(np.uint8)
    b1 = ((arr >> 8) & 0xFF).astype(np.uint8)
    b2 = ((arr >> 16) & 0xFF).astype(np.uint8)
    b3 = ((arr >> 24) & 0xFF).astype(np.uint8)
    return (lookup[b0] + lookup[b1] + lookup[b2] + lookup[b3]).astype(np.uint8)


_POPCOUNTS = _popcount_fast(_ALL_BITS)

# Expected sorted output: for input with k ones, output has k ones in lowest k positions
_EXPECTED_BITS = np.zeros(NUM_INPUTS, dtype=np.uint32)
for k in range(N + 1):
    mask = (1 << k) - 1
    _EXPECTED_BITS[_POPCOUNTS == k] = mask


def _apply_comparator(bits, i, j):
    """Apply compare-exchange (i, j) to all inputs. Wire i gets min, wire j gets max.
    For Boolean: if wire i has 1 and wire j has 0, swap them.
    """
    bit_i = (bits >> i) & 1
    bit_j = (bits >> j) & 1
    # mask: positions where i=1 and j=0 (need to swap)
    mask = (bit_i & ~bit_j) & 1
    # After swap: i gets j's value, j gets i's value
    # i_new = i & ~mask | j & mask
    # j_new = j & ~mask | i & mask
    bits = bits & ~(mask << i)  # clear i where mask
    bits = bits & ~(mask << j)  # clear j where mask
    # Set i to j's original value where mask
    bits = bits | ((bit_j & mask) << i)
    # Set j to i's original value where mask
    bits = bits | ((bit_i & mask) << j)
    return bits


def _verify_network(comparators):
    """Verify network sorts all Boolean inputs using bit-packed representation."""
    bits = _ALL_BITS.copy()
    for (i, j) in comparators:
        bits = _apply_comparator(bits, i, j)
    return np.array_equal(bits, _EXPECTED_BITS)


def _depth(comparators):
    """Compute parallel depth of the network."""
    if not comparators:
        return 0
    wire_depth = [0] * N
    max_d = 0
    for (i, j) in comparators:
        d = max(wire_depth[i], wire_depth[j]) + 1
        max_d = max(max_d, d)
        wire_depth[i] = d
        wire_depth[j] = d
    return max_d


def _odd_even_transposition():
    """Generate odd-even transposition network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _odd_even_merge_sort():
    """Generate odd-even merge sort network for N wires."""
    def _odd_even_merge(n, a, b):
        """Generate comparators for odd-even merge of two sorted arrays of size n."""
        if n == 1:
            return [(a, b)]
        half = n // 2
        comps = []
        comps.extend(_odd_even_merge(half, a, b))
        comps.extend(_odd_even_merge(half, a + half, b + half))
        for i in range(half):
            comps.append((a + i, b + i))
        return comps

    def _odd_even_sort(n, a):
        """Generate comparators for odd-even sort of array starting at a, size n."""
        if n <= 1:
            return []
        # Find largest power of 2 <= n
        # For simplicity, handle exact powers of 2, pad for others
        # 17 is not a power of 2, so we need a different approach
        # Use a simpler recursive structure
        half = n // 2
        comps = []
        comps.extend(_odd_even_sort(half, a))
        comps.extend(_odd_even_sort(n - half, a + half))
        # Merge
        # This is approximate; for non-power-of-2, this may not be correct
        # Let's use a different approach for 17 wires
        return comps

    # For 17 wires, use a known construction
    # Actually, let's just use odd-even transposition as the base
    # and rely on the search to improve it
    return _odd_even_transposition()


def _initial_network():
    """Generate initial network. Use odd-even transposition as reliable base."""
    return _odd_even_transposition()


def _random_comparator(rng):
    """Generate a random comparator."""
    i = rng.randint(0, N - 2)
    j = rng.randint(i + 1, N - 1)
    return (i, j)


def construct():
    start_time = time.time()
    rng = random.Random(42)

    # Start with odd-even transposition (known correct, 136 comparators)
    best = _initial_network()
    if not _verify_network(best):
        # Should not happen, but safety
        return best

    best_count = len(best)
    best_depth = _depth(best)

    # Simulated annealing
    # Temperature schedule: start high, cool down
    T_start = 10.0
    T_end = 0.1
    # We'll run for the available time

    current = best.copy()
    current_count = len(current)
    current_depth = _depth(current)

    # Track best
    global_best = best.copy()
    global_best_count = best_count
    global_best_depth = best_depth

    # Precompute all possible comparators
    all_comps = []
    for i in range(N):
        for j in range(i + 1, N):
            all_comps.append((i, j))

    # Simulated annealing loop
    iterations = 0
    no_improve_count = 0

    while time.time() - start_time < DEADLINE - 3.0:
        iterations += 1
        no_improve_count += 1

        # Generate a random move
        move_type = rng.random()
        candidate = current.copy()

        if move_type < 0.4 and len(candidate) > 10:
            # Remove a random comparator
            idx = rng.randint(0, len(candidate) - 1)
            candidate.pop(idx)
        elif move_type < 0.6 and len(candidate) < 4096:
            # Add a random comparator
            comp = _random_comparator(rng)
            pos = rng.randint(0, len(candidate))
            candidate.insert(pos, comp)
        elif move_type < 0.8 and len(candidate) > 1:
            # Move a comparator to a new position
            idx = rng.randint(0, len(candidate) - 1)
            comp = candidate.pop(idx)
            new_pos = rng.randint(0, len(candidate))
            candidate.insert(new_pos, comp)
        else:
            # Swap two comparators
            if len(candidate) >= 2:
                idx1 = rng.randint(0, len(candidate) - 2)
                idx2 = rng.randint(idx1 + 1, len(candidate) - 1)
                candidate[idx1], candidate[idx2] = candidate[idx2], candidate[idx1]

        # Verify candidate
        if not _verify_network(candidate):
            # Reject invalid moves
            continue

        candidate_count = len(candidate)
        candidate_depth = _depth(candidate)

        # Acceptance criterion
        # Primary objective: minimize count
        # Secondary: minimize depth
        delta_count = candidate_count - current_count
        delta_depth = candidate_depth - current_depth

        # Compute temperature based on progress
        elapsed = time.time() - start_time
        remaining = max(0, DEADLINE - 3.0 - elapsed)
        # Linear cooling
        T = T_start * (T_end / T_start) ** (1.0 - remaining / (DEADLINE - 3.0 - start_time))
        T = max(T, T_end)

        if delta_count < 0:
            # Better count, always accept
            current = candidate
            current_count = candidate_count
            current_depth = candidate_depth
            if candidate_count < global_best_count:
                global_best = candidate.copy()
                global_best_count = candidate_count
                global_best_depth = candidate_depth
                no_improve_count = 0
        elif delta_count == 0:
            # Same count, accept if depth is better or with probability
            if delta_depth < 0:
                current = candidate
                current_depth = candidate_depth
                if candidate_depth < global_best_depth:
                    global_best = candidate.copy()
                    global_best_depth = candidate_depth
                    no_improve_count = 0
            else:
                # Accept with probability exp(-delta_depth / T)
                if rng.random() < np.exp(-delta_depth / T):
                    current = candidate
                    current_depth = candidate_depth
        else:
            # Worse count, accept with probability exp(-delta_count / T)
            if rng.random() < np.exp(-delta_count / T):
                current = candidate
                current_count = candidate_count
                current_depth = candidate_depth

        # Periodically try greedy removal from current best
        if iterations % 100 == 0 and time.time() - start_time < DEADLINE - 5.0:
            # Try to remove comparators from global best
            temp_best = global_best.copy()
            improved = True
            while improved and time.time() - start_time < DEADLINE - 5.0:
                improved = False
                for idx in range(len(temp_best)):
                    candidate = temp_best[:idx] + temp_best[idx+1:]
                    if _verify_network(candidate):
                        temp_best = candidate
                        improved = True
                        break
            if len(temp_best) < global_best_count:
                global_best = temp_best
                global_best_count = len(temp_best)
                global_best_depth = _depth(global_best)
                current = global_best.copy()
                current_count = global_best_count
                current_depth = global_best_depth
                no_improve_count = 0

    # Final greedy removal pass
    final = global_best.copy()
    improved = True
    while improved and time.time() - start_time < DEADLINE - 1.0:
        improved = False
        for idx in range(len(final)):
            if time.time() - start_time > DEADLINE - 1.0:
                break
            candidate = final[:idx] + final[idx+1:]
            if _verify_network(candidate):
                final = candidate
                improved = True
                break

    # Ensure we return a valid network
    if _verify_network(final):
        return final
    else:
        return global_best
