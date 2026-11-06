"""Rewritten program: fast necessary-condition filter + bitonic seed + add/remove local search."""
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

# Precompute a boolean lookup table for fast sorted-mask check
_SORTED_LUT = np.zeros(1 << N, dtype=np.bool_)
for _m in _SORTED_MASKS:
    _SORTED_LUT[_m] = True


def _apply_comparators(comparators, state):
    """Apply a list of (i,j) compare-exchange to a single integer state.
    min goes to wire i, max to wire j."""
    for i, j in comparators:
        bi = (state >> i) & 1
        bj = (state >> j) & 1
        if bi == 1 and bj == 0:
            state ^= (1 << i) | (1 << j)
    return state


def _necessary_filter(comparators, n=N):
    """Fast necessary condition: for each wire w, the mask with only bit w set
    must end with bit w = 1 after the network (otherwise that input is unsorted).
    This is necessary for sorting and very cheap (~n * len comps)."""
    for w in range(n):
        state = 1 << w
        for i, j in comparators:
            bi = (state >> i) & 1
            bj = (state >> j) & 1
            if bi == 1 and bj == 0:
                state ^= (1 << i) | (1 << j)
        if ((state >> w) & 1) == 0:
            return False
    return True


def _verify_full(comparators, n=N):
    """Full verification over all 2^n inputs using numpy with LUT lookup."""
    if not comparators:
        return n <= 1
    total = 1 << n
    inputs = np.arange(total, dtype=np.uint32)
    for i, j in comparators:
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        swap_mask = (bit_i == 0) & (bit_j == 1)
        toggle = (1 << i) | (1 << j)
        inputs = np.where(swap_mask, inputs ^ toggle, inputs)
    # Use precomputed LUT for O(1) per-element check
    if not np.all(_SORTED_LUT[inputs]):
        return False
    return True


def _verify(comparators, n=N):
    """Verify a network: quick necessary filter then full check."""
    if not _necessary_filter(comparators, n):
        return False
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
    deadline = start_time + 80  # safety margin before 90s kill

    # Seed with compact bitonic network (96 comps for N=18)
    seed = _bitonic_network(N)
    if not _verify(seed):
        seed = _odd_even_transposition(N)

    best = list(seed)
    best_len = len(best)

    if time.time() > deadline:
        return best

    random.seed(42)
    current = list(best)

    # Greedy removal cascade from the start
    def greedy_remove(net, budget_iters=200):
        changed = True
        iters = 0
        while changed and iters < budget_iters:
            iters += 1
            changed = False
            if time.time() > deadline:
                return net
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

    # Local search: removal, addition, swap, mutation
    max_iterations = 400000
    consecutive_failures = 0
    max_consecutive_failures = 3000

    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        if len(current) <= 70:
            break
        if consecutive_failures > max_consecutive_failures:
            random.seed(random.randint(0, 2 ** 32 - 1))
            consecutive_failures = 0
            current = list(best)

        mod_type = random.random()

        if mod_type < 0.45 and len(current) > 1:
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
                    for _ in range(15):
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

        elif mod_type < 0.60:
            # Add a random comparator (helps escape local minima)
            if len(current) >= 4095:
                consecutive_failures += 1
                continue
            if random.random() < 0.6:
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
            # Insert at random position
            pos = random.randint(0, len(current))
            candidate = current[:pos] + [(a, b)] + current[pos:]
            if _verify(candidate):
                current = candidate
                consecutive_failures = 0
                # Try to remove the just-added one if it's redundant, or keep
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1

        elif mod_type < 0.75:
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
            # Replace a comparator with a different one
            if len(current) == 0:
                continue
            idx = random.randint(0, len(current) - 1)
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
            if _verify(candidate):
                current = candidate
                consecutive_failures = 0
                if len(current) < best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                consecutive_failures += 1

        else:
            # Batch removal
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
