"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time
import random
import math

_INITIAL_BITSETS = None

def _init_bitsets():
    total_bits = 1 << N
    bitsets = [0] * N
    for k in range(N):
        period = 1 << (k + 1)
        half_period = 1 << k
        pattern = (1 << half_period) - 1
        pattern <<= half_period
        current_val = pattern
        current_len = period
        while current_len < total_bits:
            current_val |= (current_val << current_len)
            current_len *= 2
        bitsets[k] = current_val
    return bitsets

def _get_initial_bitsets():
    global _INITIAL_BITSETS
    if _INITIAL_BITSETS is None:
        _INITIAL_BITSETS = _init_bitsets()
    return _INITIAL_BITSETS

def _is_sorting_network(comparators):
    if not comparators:
        return N <= 1
    bitsets = list(_get_initial_bitsets())
    for i, j in comparators:
        bi = bitsets[i]
        bj = bitsets[j]
        bitsets[i] = bi & bj
        bitsets[j] = bi | bj
    for k in range(N - 1):
        if bitsets[k] & ~bitsets[k + 1]:
            return False
    return True

def _greedy_reduce_single(comparators, deadline):
    """Remove comparators one at a time if redundant."""
    current = list(comparators)
    i = 0
    while i < len(current):
        if time.time() > deadline:
            break
        saved = current[i]
        current.pop(i)
        if _is_sorting_network(current):
            pass
        else:
            current.insert(i, saved)
            i += 1
    return current

def _greedy_reduce_pairwise(comparators, deadline):
    """Try removing pairs of comparators simultaneously to escape local optima."""
    current = list(comparators)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        n = len(current)
        for a in range(n):
            if time.time() > deadline:
                return current
            for b in range(a + 1, n):
                if time.time() > deadline:
                    return current
                # Build list without a and b
                reduced = []
                for idx in range(n):
                    if idx != a and idx != b:
                        reduced.append(current[idx])
                if _is_sorting_network(reduced):
                    current = reduced
                    changed = True
                    break
            if changed:
                break
    return current

def _simulated_annealing(start_network, deadline, seed=None):
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random
    current = list(start_network)
    best = list(current)
    best_size = len(current)
    T = 2.0
    cooling_rate = 0.997
    min_T = 0.03
    steps_since_improvement = 0
    max_stall_steps = 80

    while time.time() < deadline and T > min_T:
        r = rng.random()
        neighbor = list(current)
        move_valid = False
        delta = 0

        if r < 0.30:
            if len(neighbor) > 0:
                idx = rng.randint(0, len(neighbor) - 1)
                neighbor.pop(idx)
                delta = -1
                move_valid = True
        elif r < 0.50:
            if len(neighbor) > 0:
                idx = rng.randint(0, len(neighbor) - 1)
                i, j = neighbor[idx]
                if rng.random() < 0.5:
                    if i + 1 < N - 1 and j + 1 < N:
                        ni, nj = i + 1, j + 1
                        if ni < nj:
                            neighbor[idx] = (ni, nj)
                            delta = 0
                            move_valid = True
                else:
                    if i - 1 >= 0 and j - 1 > 0:
                        ni, nj = i - 1, j - 1
                        if ni < nj:
                            neighbor[idx] = (ni, nj)
                            delta = 0
                            move_valid = True
        elif r < 0.65:
            if len(neighbor) > 0:
                idx = rng.randint(0, len(neighbor) - 1)
                if rng.random() < 0.6:
                    dist = rng.randint(1, 3)
                else:
                    dist = rng.randint(1, N - 1)
                a = rng.randint(0, N - 1 - dist)
                b = a + dist
                neighbor[idx] = (a, b)
                delta = 0
                move_valid = True
        elif r < 0.75:
            if len(neighbor) > 1:
                idx1 = rng.randint(0, len(neighbor) - 1)
                idx2 = rng.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    comp = neighbor.pop(idx1)
                    neighbor.insert(idx2, comp)
                    delta = 0
                    move_valid = True
        elif r < 0.85:
            if len(neighbor) > 1:
                idx1 = rng.randint(0, len(neighbor) - 1)
                idx2 = rng.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                    delta = 0
                    move_valid = True
        else:
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            neighbor.append((i, j))
            delta = 1
            move_valid = True

        if not move_valid:
            T *= cooling_rate
            continue

        if _is_sorting_network(neighbor):
            new_size = len(neighbor)
            if new_size < best_size:
                best = neighbor
                best_size = new_size
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            if delta < 0:
                current = neighbor
            elif delta == 0:
                current = neighbor
            else:
                if rng.random() < math.exp(-1.0 / T):
                    current = neighbor
        else:
            steps_since_improvement += 1

        if steps_since_improvement > max_stall_steps:
            T = max(T * 1.5, 0.5)
            steps_since_improvement = 0
            if rng.random() < 0.4 and len(current) > 0:
                for _ in range(rng.randint(1, 3)):
                    i = rng.randint(0, N - 2)
                    j = rng.randint(i + 1, N - 1)
                    current.append((i, j))

        T *= cooling_rate

    return best

def construct():
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))

    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill

    best = list(seed_network)
    # Initial aggressive reduction
    best = _greedy_reduce_single(best, deadline)
    best = _greedy_reduce_pairwise(best, deadline)

    # Multi-start SA with restarts
    restart_budget = 8
    restart_idx = 0
    while time.time() < deadline and restart_idx < restart_budget:
        remaining = deadline - time.time()
        if remaining < 5:
            break
        # Allocate time for this SA run
        sa_time = min(remaining - 2, 10 + restart_idx * 2)
        sa_deadline = time.time() + sa_time

        # Perturb best to create a new starting point
        perturbed = list(best)
        num_perturb = random.randint(2, 5)
        for _ in range(num_perturb):
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            perturbed.append((i, j))

        sa_result = _simulated_annealing(perturbed, sa_deadline, seed=restart_idx * 12345 + 7)

        if len(sa_result) < len(best):
            best = sa_result

        # Reduce after SA
        best = _greedy_reduce_single(best, deadline)
        if time.time() < deadline:
            best = _greedy_reduce_pairwise(best, deadline)

        restart_idx += 1

    # Final reduction pass
    best = _greedy_reduce_single(best, deadline)
    if time.time() < deadline:
        best = _greedy_reduce_pairwise(best, deadline)

    return best
# EVOLVE-BLOCK-END
