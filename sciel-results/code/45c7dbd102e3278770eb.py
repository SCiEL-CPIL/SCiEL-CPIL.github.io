"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = 55.0  # Safety margin under 90s wall-clock

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    weights = [1.0 / num for num in numbers]

    # conflict[i] = bitmask of j such that lcm(numbers[i], numbers[j]) <= N
    conflict = [0] * n
    for i in range(n):
        a = numbers[i]
        mask = 0
        for j in range(n):
            if i != j and math.lcm(a, numbers[j]) <= N:
                mask |= (1 << j)
        conflict[i] = mask

    # order: indices by descending weight (i.e., ascending number)
    order = sorted(range(n), key=lambda i: -weights[i])

    def elapsed():
        return time.time() - start_time

    def mask_val(mask):
        return sum(weights[i] for i in range(n) if (mask >> i) & 1)

    # Greedy by weight
    def greedy():
        sel = 0
        for i in order:
            if conflict[i] & sel:
                continue
            sel |= (1 << i)
        return sel

    best_mask = greedy()
    best_val = mask_val(best_mask)

    # Structural base: numbers in (128, 256]
    base_mask = 0
    for i in range(n):
        if numbers[i] > 128:
            base_mask |= (1 << i)
    # Verify base independence
    base_list = [i for i in range(n) if (base_mask >> i) & 1]
    base_valid = True
    for ai in range(len(base_list)):
        for bi in range(ai + 1, len(base_list)):
            if conflict[base_list[ai]] & (1 << base_list[bi]):
                base_valid = False
                break
        if not base_valid:
            break

    # If base not independent, fall back to greedy base
    if not base_valid:
        base_mask = greedy()

    # Greedy add small numbers (<=128) to base
    small = [i for i in range(n) if numbers[i] <= 128]
    small.sort(key=lambda i: -weights[i])
    cur = base_mask
    for i in small:
        if conflict[i] & cur:
            continue
        cur |= (1 << i)
    if mask_val(cur) > best_val:
        best_val = mask_val(cur)
        best_mask = cur

    # Local search: single adds and single swaps
    all_idx = list(range(n))
    def local_search(mask):
        nonlocal best_mask, best_val
        improved = True
        while improved and elapsed() < deadline:
            improved = False
            # single add
            for i in all_idx:
                if (mask >> i) & 1:
                    continue
                if conflict[i] & mask:
                    continue
                new = mask | (1 << i)
                val = mask_val(new)
                if val > best_val + 1e-12:
                    best_val = val
                    best_mask = new
                    mask = new
                    improved = True
                    break
            # single swap: remove r, add a
            for a in small:
                if (mask >> a) & 1:
                    continue
                for r in all_idx:
                    if not ((mask >> r) & 1):
                        continue
                    new = (mask & ~(1 << r)) | (1 << a)
                    if conflict[a] & (new & ~(1 << a)):
                        continue
                    val = mask_val(new)
                    if val > best_val + 1e-12:
                        best_val = val
                        best_mask = new
                        mask = new
                        improved = True
                        break
                if improved:
                    break
        return mask

    best_mask = local_search(best_mask)

    result = [numbers[i] for i in range(n) if (best_mask >> i) & 1]
    if not result:
        result = [numbers[order[0]]]

    # Final validation
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.lcm(result[i], result[j]) <= N:
                valid = False
                break
        if not valid:
            break
    if not valid:
        result = [numbers[i] for i in range(n) if (greedy() >> i) & 1]

    return result

# EVOLVE-BLOCK-END
