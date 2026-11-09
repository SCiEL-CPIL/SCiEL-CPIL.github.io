"""Maximize 3-term APs subject to no 4-term AP in [1,256]."""
import time
import random

N = 256
DEADLINE = time.time() + 80.0

# Precompute all 3-APs (a, a+d, a+2d)
TRIPLES = []
TRIPLE_IDX = []
for d in range(1, N // 2 + 1):
    for a in range(1, N - 2 * d + 1):
        TRIPLES.append((a, a + d, a + 2 * d))
        TRIPLE_IDX.append((a - 1, a + d - 1, a + 2 * d - 1))
NT = len(TRIPLES)

# Precompute all 4-APs (a, a+d, a+2d, a+3d)
QUADS = []
QUAD_IDX = []
for d in range(1, N // 3 + 1):
    for a in range(1, N - 3 * d + 1):
        QUADS.append((a, a + d, a + 2 * d, a + 3 * d))
        QUAD_IDX.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
NQ = len(QUADS)


def count_3aps(mask):
    """Count 3-APs in a bitmask (0-indexed bits)."""
    c = 0
    for i, j, k in TRIPLE_IDX:
        if (mask >> i) & 1 and (mask >> j) & 1 and (mask >> k) & 1:
            c += 1
    return c


def count_4aps(mask):
    """Count 4-APs in a bitmask."""
    c = 0
    for i, j, k, l in QUAD_IDX:
        if (mask >> i) & 1 and (mask >> j) & 1 and (mask >> k) & 1 and (mask >> l) & 1:
            c += 1
    return c


def greedy_seed(rng):
    """Greedy: add elements in random order, keep if no 4-AP formed."""
    order = list(range(N))
    rng.shuffle(order)
    mask = 0
    for x in order:
        m2 = mask | (1 << x)
        if count_4aps(m2) == 0:
            mask = m2
    return mask


def score_mask(mask):
    """Objective: 3-AP count if no 4-AP, else -infinity."""
    if count_4aps(mask) > 0:
        return -10**9
    return count_3aps(mask)


def add_gain(mask, x):
    """Gain in 3-APs if we add x (assuming no 4-AP introduced)."""
    g = 0
    for i, j, k in TRIPLE_IDX:
        if (mask >> i) & 1 and (mask >> j) & 1 and (mask >> k) & 1:
            continue
        if i == x or j == x or k == x:
            if (mask >> i) & 1 and (mask >> j) & 1 and (mask >> k) & 1:
                g += 1
    return g


def add_loss(mask, x):
    """Loss in 3-APs if we remove x."""
    l = 0
    for i, j, k in TRIPLE_IDX:
        if (mask >> i) & 1 and (mask >> j) & 1 and (mask >> k) & 1:
            if i == x or j == x or k == x:
                l += 1
    return l


def add_4ap_count(mask, x):
    """Number of 4-APs that would be formed by adding x."""
    c = 0
    for i, j, k, l in QUAD_IDX:
        if i == x or j == x or k == x or l == x:
            if (mask >> i) & 1 and (mask >> j) & 1 and (mask >> k) & 1 and (mask >> l) & 1:
                c += 1
    return c


def remove_4ap_count(mask, x):
    """Number of 4-APs that would be broken by removing x."""
    c = 0
    for i, j, k, l in QUAD_IDX:
        if i == x or j == x or k == x or l == x:
            if (mask >> i) & 1 and (mask >> j) & 1 and (mask >> k) & 1 and (mask >> l) & 1:
                c += 1
    return c


def polish(mask):
    """Hill-climb: add, remove, swap moves. Returns improved mask."""
    improved = True
    while improved:
        improved = False
        cur = score_mask(mask)
        if cur < 0:
            # Invalid: remove elements that break 4-APs
            for x in range(N):
                if (mask >> x) & 1:
                    m2 = mask & ~(1 << x)
                    if count_4aps(m2) < count_4aps(mask):
                        mask = m2
                        improved = True
                        break
            continue
        # Try add
        best_add = None
        best_add_gain = 0
        for x in range(N):
            if (mask >> x) & 1:
                continue
            if add_4ap_count(mask, x) > 0:
                continue
            g = add_gain(mask, x)
            if g > best_add_gain:
                best_add_gain = g
                best_add = x
        if best_add is not None and best_add_gain > 0:
            mask |= (1 << best_add)
            improved = True
            continue
        # Try remove (if it frees up more adds later - skip for now, keep simple)
        # Try swap: remove x, add y
        for x in range(N):
            if not ((mask >> x) & 1):
                continue
            for y in range(N):
                if (mask >> y) & 1:
                    continue
                if add_4ap_count(mask & ~(1 << x), y) > 0:
                    continue
                g = add_gain(mask & ~(1 << x), y) - add_loss(mask, x)
                if g > 0:
                    mask = (mask & ~(1 << x)) | (1 << y)
                    improved = True
                    break
            if improved:
                break
    return mask


def simulated_annealing(mask, rng, time_budget):
    """SA with add/remove/swap moves."""
    t_start = time.time()
    cur_score = score_mask(mask)
    best_mask = mask
    best_score = cur_score
    T = 10.0
    T_min = 0.01
    steps = 0
    while T > T_min:
        steps += 1
        # Pick a random move
        r = rng.random()
        if r < 0.4:
            # Add
            x = rng.randrange(N)
            if (mask >> x) & 1:
                continue
            new_mask = mask | (1 << x)
        elif r < 0.7:
            # Remove
            bits = [i for i in range(N) if (mask >> i) & 1]
            if not bits:
                continue
            x = rng.choice(bits)
            new_mask = mask & ~(1 << x)
        else:
            # Swap
            bits = [i for i in range(N) if (mask >> i) & 1]
            nonbits = [i for i in range(N) if not ((mask >> i) & 1)]
            if not bits or not nonbits:
                continue
            x = rng.choice(bits)
            y = rng.choice(nonbits)
            new_mask = (mask & ~(1 << x)) | (1 << y)
        new_score = score_mask(new_mask)
        if new_score >= cur_score:
            mask = new_mask
            cur_score = new_score
            if cur_score > best_score:
                best_score = cur_score
                best_mask = mask
        else:
            delta = new_score - cur_score
            if delta < -500:  # too bad
                continue
            p = 1.0 / (1.0 + delta / (T + 1e-9))
            if rng.random() < p:
                mask = new_mask
                cur_score = new_score
        T *= 0.9995
        if time.time() - t_start > time_budget:
            break
    return best_mask


def construct():
    global DEADLINE
    deadline = time.time() + 80.0
    rng = random.Random(42)
    best_mask = 0
    best_score = -1

    # Phase 1: Multiple greedy seeds
    n_greedy = 6
    for i in range(n_greedy):
        if time.time() > deadline - 20:
            break
        m = greedy_seed(rng)
        m = polish(m)
        s = score_mask(m)
        if s > best_score:
            best_score = s
            best_mask = m

    # Phase 2: SA from each seed, then polish
    for i in range(n_greedy):
        if time.time() > deadline - 10:
            break
        m = greedy_seed(rng)
        m = polish(m)
        remaining = deadline - time.time() - 5
        if remaining < 3:
            break
        m = simulated_annealing(m, rng, remaining)
        m = polish(m)
        s = score_mask(m)
        if s > best_score:
            best_score = s
            best_mask = m

    # Final polish
    best_mask = polish(best_mask)
    s = score_mask(best_mask)
    if s > best_score:
        best_score = s

    # Convert mask to sorted list
    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    if not result:
        result = [1]
    return result


if __name__ == "__main__":
    res = construct()
    print(len(res), res[:20])
    print("3-APs:", count_3aps(sum(1 << (x-1) for x in res)))
    print("4-APs:", count_4aps(sum(1 << (x-1) for x in res)))
