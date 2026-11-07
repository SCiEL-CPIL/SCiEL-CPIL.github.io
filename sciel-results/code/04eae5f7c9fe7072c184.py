"""New design: bitmask subset-sum validity + greedy + simulated annealing."""
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 256, 'maximum_size': 30}

SUM_BOUND = 256 * 30 + 1  # 7681

def _build_div_masks(bound):
    masks = [0] * (bound + 1)
    for s in range(1, bound + 1):
        m = 0
        k = 2
        while k * s < bound:
            m |= (1 << (k * s))
            k += 1
        masks[s] = m
    return masks

_DIV_MASKS = _build_div_masks(SUM_BOUND)

def _subset_sums(elements):
    """Return dict of sum->count for all nonempty subset sums."""
    sums = {0: 1}
    for x in elements:
        new = dict(sums)
        for s, c in sums.items():
            ns = s + x
            new[ns] = new.get(ns, 0) + c
        sums = new
    if 0 in sums:
        del sums[0]
    return sums

def _is_valid(elements):
    sums = _subset_sums(elements)
    for c in sums.values():
        if c > 1:
            return False
    bitset = 0
    conflict = 0
    for s in sums:
        bitset |= (1 << s)
        conflict |= _DIV_MASKS[s]
    return (bitset & conflict) == 0

def _greedy_build(seed):
    rng = random.Random(seed)
    candidates = list(range(1, 257))
    rng.shuffle(candidates)
    # bias: large numbers first (fewer multiples)
    candidates.sort(key=lambda x: (-x, rng.random()))
    current = []
    for x in candidates:
        trial = current + [x]
        if _is_valid(trial):
            current = trial
    return current

def _anneal(start, deadline, seed):
    rng = random.Random(seed)
    current = list(start)
    best = list(start)
    best_size = len(start)
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    while time.time() < deadline:
        move = rng.random()
        if move < 0.5:
            if not current:
                continue
            i = rng.randrange(len(current))
            x = rng.randint(1, 256)
            if x in current:
                continue
            new = list(current)
            new[i] = x
        elif move < 0.8:
            x = rng.randint(1, 256)
            if x in current or len(current) >= 30:
                continue
            new = current + [x]
        else:
            if len(current) <= 1:
                continue
            i = rng.randrange(len(current))
            new = list(current)
            del new[i]
        new_valid = _is_valid(new)
        new_size = len(new)
        if new_valid:
            if new_size >= len(current):
                current = new
                if new_size > best_size:
                    best = list(new)
                    best_size = new_size
            else:
                if rng.random() < 0.05 * T:
                    current = new
        else:
            if rng.random() < 0.005 * T:
                current = new
        T = max(T_min, T * cooling)
    return best, best_size

def construct():
    deadline = time.time() + 80
    best = []
    best_size = 0
    # greedy restarts
    for r in range(50):
        if time.time() > deadline:
            break
        g = _greedy_build(r)
        if len(g) > best_size:
            best = g
            best_size = len(g)
    # anneal from best few times
    for a in range(5):
        if time.time() > deadline:
            break
        an, asz = _anneal(best, deadline, 1000 + a)
        if asz > best_size:
            best = an
            best_size = asz
    result = [int(x) for x in best]
    return result
# EVOLVE-BLOCK-END
