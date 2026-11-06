"""Size-efficient sorting network synthesis for 18 wires.

Combines bit-parallel verification (from inspiration 1) with a robust
local-search loop featuring multiple mutation types (from inspiration 1)
and a deadline guard.

Strategy:
  1. Pre-compute initial wire states as Python integers (bitsets) for fast
     0-1 principle verification over all 2**18 Boolean inputs.
  2. Start from the odd-even transposition seed network (153 comparators).
  3. Run a local search loop with random mutations:
     - Remove a random comparator
     - Swap two comparators
     - Replace a comparator with a random one
     - Remove 2-3 comparators at once
     After each successful mutation, do a short greedy cascade.
  4. Keep the best (fewest-comparator) valid network found.
  5. Hard deadline guard: return the best result with a safety margin before 90s.
"""
import time
import random

N = 18
_M = 1 << N  # 262144

# Precompute initial wire states as big integers.
# _wires_init[w] has bit k set iff wire w is 1 in input k (i.e., (k >> w) & 1).
_wires_init = [0] * N
for w in range(N):
    mask = 0
    # Build mask: bit k is 1 if (k >> w) & 1
    # Pattern: for each block of size 2**(w+1), the top half has bit w set.
    # We can build this iteratively.
    # Simpler: loop over k. 2^18 * 18 = 4.7M iterations, done once.
    for k in range(_M):
        if (k >> w) & 1:
            mask |= (1 << k)
    _wires_init[w] = mask


def _verify_network_bit_parallel(comparators):
    """Return True iff the comparator list sorts all 2**N Boolean inputs.

    Uses bit-parallelism: state is a list of N Python integers, each representing
    the set of inputs where that wire is 1.
    """
    if not comparators:
        return N <= 1

    state = _wires_init[:]

    for i, j in comparators:
        # Find inputs where wire i is 1 and wire j is 0 -> need to swap
        swap_mask = state[i] & ~state[j]

        if swap_mask == 0:
            continue

        old_i = state[i]
        old_j = state[j]

        state[i] = (old_i & ~swap_mask) | (old_j & swap_mask)
        state[j] = (old_j & ~swap_mask) | (old_i & swap_mask)

    # Check if sorted: no adjacent pair (w, w+1) with wire w = 1 and wire w+1 = 0
    for w in range(N - 1):
        if state[w] & ~state[w + 1]:
            return False

    return True


def _odd_even_transposition(n=N):
    """Generate the odd-even transposition sorting network for n wires."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators


def _greedy_cascade(current, deadline, max_steps=20):
    """After a successful mutation, try to remove more comparators greedily."""
    for _ in range(max_steps):
        if time.time() > deadline:
            break
        if len(current) <= 70:
            break
        idx = random.randint(0, len(current) - 1)
        candidate = current[:idx] + current[idx + 1:]
        if _verify_network_bit_parallel(candidate):
            current = candidate
        else:
            break
    return current


def construct():
    """Construct a compare-exchange network that sorts all 18-input sequences.

    Returns an ordered list of integer pairs (i, j), 0 <= i < j < N.
    """
    start_time = time.time()
    deadline = start_time + 50  # 10s safety margin under 60s limit (evaluator uses 90s but task says 60s)

    best = _odd_even_transposition(N)
    best_len = len(best)

    if not _verify_network_bit_parallel(best):
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

        mod_type = random.random()

        if mod_type < 0.70 and len(current) > 1:
            # Try removing a random comparator
            idx = random.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx + 1:]
            if _verify_network_bit_parallel(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
                # Greedy cascade
                current = _greedy_cascade(current, deadline)
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
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
            # Generate random valid comparator, biased towards short distances
            if random.random() < 0.5:
                dist = random.randint(1, 3)
            else:
                dist = random.randint(1, N - 1)
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

    # Final safety: ensure the returned network is valid.
    if not _verify_network_bit_parallel(best):
        best = _odd_even_transposition(N)

    return best
