"""Size-efficient sorting network search for N=16 wires."""
import time
import random
import numpy as np

N = 16

def _apply_comparators_mask(mask, comparators):
    """Apply comparators to a single 16-bit mask."""
    for i, j in comparators:
        a = (mask >> i) & 1
        b = (mask >> j) & 1
        if a > b:
            mask ^= (1 << i) | (1 << j)
    return mask

def _is_sorted_mask(mask):
    """Check if 16-bit mask is sorted (0s at low indices, 1s at high)."""
    for z in range(N + 1):
        if mask == (0xFFFF ^ ((1 << z) - 1)) & 0xFFFF:
            return True
    return False

def _verify_network_fast(comparators):
    """Verify sorting network over all 2^16 inputs using numpy."""
    if not comparators:
        return N <= 1
    all_inputs = np.arange(1 << N, dtype=np.uint32)
    for i, j in comparators:
        mask_i = 1 << i
        mask_j = 1 << j
        cond = ((all_inputs & mask_i) != 0) & ((all_inputs & mask_j) == 0)
        swap_mask = mask_i | mask_j
        all_inputs = np.where(cond, all_inputs ^ swap_mask, all_inputs)
    valid = set()
    for z in range(N + 1):
        valid.add((0xFFFF ^ ((1 << z) - 1)) & 0xFFFF)
    for x in all_inputs:
        if x not in valid:
            return False
    return True

def _odd_even_network():
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _batcher_bitonic_network():
    """Generate Batcher's bitonic sorting network for N=16."""
    comparators = []
    def bitonic_merge(n, offset, direction):
        if n <= 1:
            return
        half = n // 2
        for i in range(half):
            a = offset + i
            b = offset + i + half
            if direction == 0:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
    def bitonic_sort(n, offset, direction):
        if n <= 1:
            return
        half = n // 2
        bitonic_sort(half, offset, 0)
        bitonic_sort(half, offset + half, 1)
        bitonic_merge(n, offset, direction)
    bitonic_sort(N, 0, 0)
    return comparators

def _try_remove(comparators, index, deadline):
    if time.time() > deadline:
        return None
    candidate = comparators[:index] + comparators[index + 1:]
    if _verify_network_fast(candidate):
        return candidate
    return None

def _local_search(start, deadline):
    current = list(start)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(current)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            new = _try_remove(current, idx, deadline)
            if new is not None:
                current = new
                improved = True
                break
        if not improved:
            for _ in range(25):
                if time.time() > deadline:
                    break
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                if (i, j) not in current:
                    pos = random.randint(0, len(current))
                    candidate = current[:pos] + [(i, j)] + current[pos:]
                    if _verify_network_fast(candidate):
                        current = candidate
                        improved = True
                        break
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 80

    batcher = _batcher_bitonic_network()
    if batcher and _verify_network_fast(batcher):
        current = list(batcher)
    else:
        current = _odd_even_network()
        if not _verify_network_fast(current):
            return current

    best = list(current)
    best_len = len(best)

    while time.time() < deadline:
        result = _local_search(current, deadline)
        if len(result) < best_len:
            best = result
            best_len = len(result)
            current = list(result)
        else:
            perturbed = list(current)
            for _ in range(20):
                if time.time() > deadline:
                    break
                if len(perturbed) > 1:
                    idx = random.randint(0, len(perturbed) - 1)
                    candidate = perturbed[:idx] + perturbed[idx + 1:]
                    if _verify_network_fast(candidate):
                        perturbed = candidate
                        break
            for _ in range(20):
                if time.time() > deadline:
                    break
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                if (i, j) not in perturbed:
                    pos = random.randint(0, len(perturbed))
                    candidate = perturbed[:pos] + [(i, j)] + perturbed[pos:]
                    if _verify_network_fast(candidate):
                        perturbed = candidate
                        break
            current = perturbed

        if best_len <= 55:
            for idx in range(len(best)):
                if time.time() > deadline:
                    break
                candidate = best[:idx] + best[idx + 1:]
                if _verify_network_fast(candidate):
                    best = candidate
                    best_len = len(best)
                    break
            if best_len <= 50:
                break

    return best
