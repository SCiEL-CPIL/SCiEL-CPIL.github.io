import numpy as np
import time
import random

N = 256

def count_3aps(mask):
    """Count distinct 3-term APs (a, a+d, a+2d) with d>0 using convolution."""
    c = np.convolve(mask.astype(np.int32), mask.astype(np.int32))
    total = 0
    for d in range(1, N // 2 + 1):
        total += c[2 * d]
    return total

def count_4aps(mask):
    """Count nonconstant 4-term APs (a, a+d, a+2d, a+3d) with d>0."""
    total = 0
    for d in range(1, N // 3 + 1):
        total += np.sum(mask[:-3 * d] * mask[1:-2 * d] * mask[2 * d:-d] * mask[3 * d:])
    return total

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Initial solution: random subset of moderate density
    # Start with a random set that has no 4-APs by construction (greedy random)
    def greedy_random_init():
        mask = np.zeros(N, dtype=np.int32)
        order = list(range(N))
        random.shuffle(order)
        for i in order:
            pos = i
            # Check if adding pos creates a 4-AP
            can_add = True
            for d in range(1, (N - pos) // 3 + 1):
                if pos + 3 * d < N:
                    if mask[pos] & mask[pos + d] & mask[pos + 2 * d] & mask[pos + 3 * d]:
                        can_add = False
                        break
            # Also check if pos is the second, third, or fourth term
            for d in range(1, pos // 3 + 1):
                if mask[pos - 3 * d] & mask[pos - 2 * d] & mask[pos - d]:
                    can_add = False
                    break
            for d in range(1, pos // 2 + 1):
                if pos + d < N and mask[pos - 2 * d] & mask[pos - d] & mask[pos + d]:
                    can_add = False
                    break
            for d in range(1, pos + 1):
                if pos + 2 * d < N and mask[pos - d] & mask[pos + d] & mask[pos + 2 * d]:
                    can_add = False
                    break
            if can_add:
                mask[pos] = 1
        return mask

    # Try multiple initializations, keep best
    best_mask = None
    best_3ap = -1
    for _ in range(20):
        m = greedy_random_init()
        if count_4aps(m) == 0:
            c3 = count_3aps(m)
            if c3 > best_3ap:
                best_3ap = c3
                best_mask = m.copy()

    if best_mask is None:
        # Fallback: use the original greedy
        selected = set()
        for x in range(1, N + 1):
            idx = x - 1
            ok = True
            for d in range(1, (idx) // 3 + 1):
                if all(selected.has(idx - j * d) for j in (1, 2, 3)):
                    ok = False
                    break
            if ok:
                selected.add(idx)
        best_mask = np.zeros(N, dtype=np.int32)
        for x in selected:
            best_mask[x] = 1
        best_3ap = count_3aps(best_mask)

    current_mask = best_mask.copy()
    current_3ap = best_3ap
    current_4ap = 0

    # Simulated annealing
    T_start = 10.0
    T_end = 0.1
    T_current = T_start
    cooling_rate = 0.999

    random.seed(42)

    while time.time() < deadline:
        # Pick a random move: add or remove a random element
        idx = random.randint(0, N - 1)
        old_val = current_mask[idx]
        current_mask[idx] = 1 - old_val

        new_4ap = count_4aps(current_mask)
        if new_4ap > 0:
            # Reject if invalid (or accept with very low prob at high T)
            if random.random() > 0.01:
                current_mask[idx] = old_val
                T_current *= cooling_rate
                continue

        new_3ap = count_3aps(current_mask)
        delta = new_3ap - current_3ap

        if delta > 0 or random.random() < np.exp(delta / T_current):
            current_3ap = new_3ap
            if new_4ap == 0 and current_3ap > best_3ap:
                best_3ap = current_3ap
                best_mask = current_mask.copy()
        else:
            current_mask[idx] = old_val

        T_current = max(T_end, T_current * cooling_rate)

    # Final: ensure best_mask is valid
    if count_4aps(best_mask) > 0:
        # Fallback to a known valid set
        selected = set()
        for x in range(1, N + 1):
            idx = x - 1
            ok = True
            for d in range(1, (idx) // 3 + 1):
                if all(selected.has(idx - j * d) for j in (1, 2, 3)):
                    ok = False
                    break
            if ok:
                selected.add(idx)
        best_mask = np.zeros(N, dtype=np.int32)
        for x in selected:
            best_mask[x] = 1

    return sorted(int(i + 1) for i in range(N) if best_mask[i])
