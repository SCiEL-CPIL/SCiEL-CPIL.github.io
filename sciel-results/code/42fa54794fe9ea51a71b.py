"""Rewritten program: fast verification + aggressive local search with 2-for-1 swaps."""
N = 18
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

# Precompute sorted-mask LUT
def _build_sorted_lut(n):
    lut = np.zeros(1 << n, dtype=np.bool_)
    for k in range(n + 1):
        mask = ((1 << k) - 1) << (n - k)
        lut[mask] = True
    return lut

_SORTED_LUT = _build_sorted_lut(N)

# Precompute swap masks for each possible comparator (i,j)
# For comparator (i,j): swap_mask[state] = 1 if bit_i==0 and bit_j==1
# We precompute the toggle value and use vectorized operations
_TOGGLES = {}
for i in range(N):
    for j in range(i + 1, N):
        _TOGGLES[(i, j)] = (1 << i) | (1 << j)


def _verify_full(comparators, n=N):
    """Full verification over all 2^n inputs using numpy."""
    if not comparators:
        return n <= 1
    total = 1 << n
    inputs = np.arange(total, dtype=np.uint32)
    for i, j in comparators:
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        swap_mask = (bit_i == 0) & (bit_j == 1)
        toggle = _TOGGLES[(i, j)]
        inputs = np.where(swap_mask, inputs ^ toggle, inputs)
    if not np.all(_SORTED_LUT[inputs]):
        return False
    return True


def _verify(comparators, n=N):
    """Verify a network: full check."""
    return _verify_full(comparators, n)


def _odd_even_transposition(n=N):
    """Generate odd-even transposition sorting network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators


def _bitonic_network(n):
    """Build a Batcher bitonic sorting network for n wires (n power of 2)."""
    def bitonic_merge(comps, lo, hi, direction):
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        bitonic_merge(comps, lo, lo + half, direction)
        bitonic_merge(comps, lo + half, hi, direction)
        step = half
        while step >= 1:
            for i in range(lo, lo + half, step):
                a = i
                b = i + step
                if b < hi:
                    if direction:
                        comps.append((a, b))
                    else:
                        comps.append((b, a))
            step //= 2

    def bitonic_sort(comps, lo, hi, direction):
        length = hi - lo
        if length <= 1:
            return
        half = length // 2
        bitonic_sort(comps, lo, lo + half, True)
        bitonic_sort(comps, lo + half, hi, False)
        bitonic_merge(comps, lo, hi, direction)

    comps = []
    bitonic_sort(comps, 0, n, True)
    return comps


def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    # Seed with bitonic network
    seed = _bitonic_network(N)
    if not _verify(seed):
        seed = _odd_even_transposition(N)

    best = list(seed)
    best_len = len(best)

    if time.time() > deadline:
        return best

    random.seed(42)
    current = list(best)

    # Greedy removal cascade
    def greedy_remove(net):
        changed = True
        while changed:
            if time.time() > deadline:
                return net
            changed = False
            for idx in range(len(net)):
                if time.time() > deadline:
                    return net
                cand = net[:idx] + net[idx + 1:]
                if _verify(cand):
                    net = cand
                    changed = True
                    break
        return net

    current = greedy_remove(current)
    if len(current) < best_len:
        best = list(current)
        best_len = len(current)

    # Local search with multiple move types
    max_iterations = 500000
    consecutive_failures = 0
    max_consecutive_failures = 5000

    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        if len(current) <= 70:
            break
        if consecutive_failures > max_consecutive_failures:
            random.seed(random.randint(0, 2**32 - 1))
            consecutive_failures = 0
            current = list(best)

        mod_type = random.random()

        if mod_type < 0.40 and len(current) > 1:
            # Remove a random comparator
            idx = random.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx + 1:]
            if _verify(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
                    # Cascade removal
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        if len(current) <= 70:
                            break
                        idx2 = random.randint(0, len(current) - 1)
                        candidate2 = current[:idx2] + current[idx2 + 1:]
                        if _verify(candidate2):
                            current = candidate2
                            if len(current) < best_len:
                                best = list(current)
                                best_len = len(current)
                        else:
                            break
            else:
                consecutive_failures += 1

        elif mod_type < 0.55:
            # 2-for-1: remove two, add one
            if len(current) < 2:
                consecutive_failures += 1
                continue
            idx1, idx2 = random.sample(range(len(current)), 2)
            # Generate a random comparator to add
            if random.random() < 0.5:
                a = random.randint(0, N - 2)
                b = a + 1
            else:
                a = random.randint(0, N - 1)
                b = random.randint(0, N - 1)
                if a == b:
                    consecutive_failures += 1
                    continue
                if a > b:
                    a, b = b, a
            # Remove the two (handle index adjustment)
            if idx1 > idx2:
                idx1 -= 1
            candidate = current[:idx2] + current[idx2+1:idx1] + current[idx1+1:]
            # Insert new comparator at random position
            pos = random.randint(0, len(candidate))
            candidate = candidate[:pos] + [(a, b)] + candidate[pos:]
            if _verify(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1

        elif mod_type < 0.65:
            # Add a random comparator
            if len(current) >= 4095:
                consecutive_failures += 1
                continue
            if random.random() < 0.6:
                a = random.randint(0, N - 2)
                b = a + 1
            else:
                a = random.randint(0, N - 1)
                b = random.randint(0, N - 1)
                if a == b:
                    consecutive_failures += 1
                    continue
                if a > b:
                    a, b = b, a
            pos = random.randint(0, len(current))
            candidate = current[:pos] + [(a, b)] + current[pos:]
            if _verify(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1

        elif mod_type < 0.80:
            # Swap two comparators
            if len(current) < 2:
                continue
            i, j = random.sample(range(len(current)), 2)
            candidate = list(current)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if _verify(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1

        elif mod_type < 0.90:
            # Replace a comparator
            if len(current) == 0:
                continue
            idx = random.randint(0, len(current) - 1)
            if random.random() < 0.7:
                a = random.randint(0, N - 2)
                b = a + 1
            else:
                a = random.randint(0, N - 1)
                b = random.randint(0, N - 1)
                if a == b:
                    consecutive_failures += 1
                    continue
                if a > b:
                    a, b = b, a
            candidate = list(current)
            candidate[idx] = (a, b)
            if _verify(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1

        else:
            # Batch removal (2-3 at once)
            if len(current) < 3:
                continue
            num_remove = random.randint(2, 3)
            if len(current) <= num_remove:
                continue
            remove_indices = set(random.sample(range(len(current)), num_remove))
            candidate = [c for idx, c in enumerate(current) if idx not in remove_indices]
            if _verify(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1

    return best

# EVOLVE-BLOCK-END
