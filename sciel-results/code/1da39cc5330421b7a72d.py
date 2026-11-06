"""Bitmask + degree-greedy + simulated-annealing construction for the
4-AP-free / 3-AP-maximizing problem on [1,256]."""
# EVOLVE-BLOCK-START
import time
import random

N = 256
NBIT = 1 << (N - 1)  # bit for element N


def bit(i):
    return 1 << (i - 1)


def popcount(m):
    return bin(m).count("1")


def count_3aps(mask):
    """Number of distinct 3-APs (positive difference) fully inside mask."""
    cnt = 0
    for a in range(1, N + 1):
        if not (mask & bit(a)):
            continue
        d = 1
        c = a + 2 * d
        while c <= N:
            if (mask & bit(a + d)) and (mask & c and (mask & bit(c))):
                cnt += 1
            d += 1
            c = a + 2 * d
    return cnt


def delta_3aps(mask, x):
    """New 3-APs created by adding x to mask."""
    new = 0
    d = 1
    c = x + 2 * d
    while c <= N:  # x as first term
        if (mask & bit(x + d)) and (mask & bit(c)):
            new += 1
        d += 1
        c = x + 2 * d
    d = 1
    a = x - d
    while a >= 1:  # x as middle term
        if (mask & bit(a)) and (mask & bit(x + d)):
            new += 1
        d += 1
        a = x - d
    d = 1
    a = x - 2 * d
    while a >= 1:  # x as last term
        if (mask & bit(a)) and (mask & bit(x - d)):
            new += 1
        d += 1
        a = x - 2 * d
    return new


def has_4ap(mask, x):
    """Would adding x to mask create a nonconstant 4-AP?"""
    # x as first term: a=x, need x+d, x+2d, x+3d
    d = 1
    while x + 3 * d <= N:
        if (mask & bit(x + d)) and (mask & bit(x + 2 * d)) and (mask & bit(x + 3 * d)):
            return True
        d += 1
    # x as second term: a=x-d, need a, x+d, x+2d
    d = 1
    while x - d >= 1 and x + 2 * d <= N:
        if (mask & bit(x - d)) and (mask & bit(x + d)) and (mask & bit(x + 2 * d)):
            return True
        d += 1
    # x as third term: a=x-2d, need a, x-d, x+d
    d = 1
    while x - 2 * d >= 1 and x + d <= N:
        if (mask & bit(x - 2 * d)) and (mask & bit(x - d)) and (mask & bit(x + d)):
            return True
        d += 1
    # x as fourth term: a=x-3d, need a, x-2d, x-d
    d = 1
    while x - 3 * d >= 1:
        if (mask & bit(x - 3 * d)) and (mask & bit(x - 2 * d)) and (mask & bit(x - d)):
            return True
        d += 1
    return False


def safe_adds(mask):
    """List of elements not in mask that can be added without a 4-AP."""
    out = []
    for x in range(1, N + 1):
        if mask & bit(x):
            continue
        if not has_4ap(mask, x):
            out.append(x)
    return out


def degree_greedy():
    """Greedy: repeatedly add the 4-AP-safe element with the highest 3-AP degree."""
    mask = 0
    for _ in range(N):
        best_x = None
        best_deg = -1
        for x in range(1, N + 1):
            if mask & bit(x):
                continue
            if has_4ap(mask, x):
                continue
            deg = delta_3aps(mask, x)
            if deg > best_deg:
                best_deg = deg
                best_x = x
        if best_x is None:
            break
        mask |= bit(best_x)
    return mask


def add_all_safe(mask):
    """Add every currently-safe element (greedy completion)."""
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if mask & bit(x):
                continue
            if not has_4ap(mask, x):
                mask |= bit(x)
                changed = True
    return mask


def local_search(mask, deadline):
    """Steepest-ascent: add positive-gain elements, then 2-opt swaps."""
    score = count_3aps(mask)
    while time.time() < deadline:
        improved = False
        # add pass
        for x in range(1, N + 1):
            if mask & bit(x):
                continue
            if has_4ap(mask, x):
                continue
            g = delta_3aps(mask, x)
            if g > 0:
                mask |= bit(x)
                score += g
                improved = True
        if not improved:
            # swap pass
            swapped = False
            elems = [x for x in range(1, N + 1) if mask & bit(x)]
            for x in elems:
                if time.time() >= deadline:
                    break
                mask2 = mask & ~bit(x)
                loss = delta_3aps(mask, x)  # 3-APs lost by removing x
                score2 = score - loss
                best_y = None
                best_g = loss  # must strictly beat loss
                for y in range(1, N + 1):
                    if mask2 & bit(y):
                        continue
                    if has_4ap(mask2, y):
                        continue
                    g = delta_3aps(mask2, y)
                    if g > best_g:
                        best_g = g
                        best_y = y
                if best_y is not None:
                    mask = mask2 | bit(best_y)
                    score = score2 + best_g
                    improved = True
                    swapped = True
                    break
            if not swapped:
                break
    return mask, score


def simulated_annealing(mask, deadline, rng):
    """SA over single add/remove/swap moves, using delta evaluation."""
    score = count_3aps(mask)
    best_mask = mask
    best_score = score
    t = 30.0
    cool = 0.985
    it = 0
    while time.time() < deadline and t > 0.05:
        it += 1
        if it % 64 == 0 and it > 100:
            cool = 0.98
        move = rng.random()
        elems = [x for x in range(1, N + 1) if mask & bit(x)]
        if not elems:
            break
        if move < 0.45:
            # add
            x = rng.randint(1, N)
            if mask & bit(x) or has_4ap(mask, x):
                continue
            g = delta_3aps(mask, x)
            d = g
            new = mask | bit(x)
            new_score = score + g
        elif move < 0.85:
            # remove
            x = rng.choice(elems)
            loss = delta_3aps(mask, x)
            d = -loss
            new = mask & ~bit(x)
            new_score = score - loss
        else:
            # swap
            x = rng.choice(elems)
            mask2 = mask & ~bit(x)
            loss = delta_3aps(mask, x)
            y = rng.randint(1, N)
            if mask2 & bit(y) or has_4ap(mask2, y):
                continue
            g = delta_3aps(mask2, y)
            d = g - loss
            new = mask2 | bit(y)
            new_score = score + d
        if d >= 0:
            accept = True
        else:
            # accept with prob exp(d/t); d in small integer range
            if d > -40:
                accept = rng.random() < (2.718281828 ** (d / t))
            else:
                accept = False
        if accept:
            mask = new
            score = new_score
            if score > best_score:
                best_score = score
                best_mask = mask
        t *= cool
    return best_mask, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    rng = random.Random(20240517)

    # 1) degree-based greedy initialization
    mask = degree_greedy()
    mask = add_all_safe(mask)
    best_mask, best_score = local_search(mask, min(deadline, start_time + 20.0))

    # 2) simulated annealing from the local optimum
    if time.time() < deadline - 15.0:
        am, ascore = simulated_annealing(best_mask, min(deadline, start_time + 55.0), rng)
        if ascore > best_score:
            best_mask, best_score = am, ascore

    # 3) final local polish
    if time.time() < deadline - 2.0:
        best_mask, best_score = local_search(best_mask, deadline)

    result = [x for x in range(1, N + 1) if best_mask & bit(x)]
    if not result:
        result = [1]
    return sorted(result)
# EVOLVE-BLOCK-END
