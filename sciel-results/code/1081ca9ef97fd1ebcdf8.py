import time
from math import gcd
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before 90s kill
    n = 128
    N = n + 1  # 1..128 -> bits 1..128

    # Precompute gcd table
    gcd_table = [[0] * N for _ in range(N)]
    for i in range(1, N):
        for j in range(i, N):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # For each x, precompute a bitmask of all elements y that "conflict" with x,
    # i.e. there exists a distinct z such that {x,y,z} is a bad triple.
    # If both y1 and y2 are in the selected set and both conflict with x,
    # then adding x creates a bad triple.
    # conflict_mask[x] = bitmask over y in 1..128.
    conflict_mask = [0] * N
    for x in range(1, N):
        mask = 0
        for y in range(1, N):
            if y == x:
                continue
            gxy = gcd_table[x][y]
            # Is there a z (distinct from x,y) with gcd(x,z)==gxy and gcd(y,z)==gxy?
            for z in range(1, N):
                if z == x or z == y:
                    continue
                if gcd_table[x][z] == gxy and gcd_table[y][z] == gxy:
                    mask |= (1 << y)
                    break
        conflict_mask[x] = mask

    def can_add_mask(mask, x):
        """Return True if x can be added to the set represented by bitmask `mask`."""
        # Need at least two selected elements that both conflict with x.
        # (x is not in mask by assumption.)
        return (mask & conflict_mask[x]).bit_count() < 2

    def is_valid_final(lst):
        """Exact O(m^3) check for the final answer."""
        m = len(lst)
        for i in range(m):
            a = lst[i]
            for j in range(i + 1, m):
                b = lst[j]
                gab = gcd_table[a][b]
                for k in range(j + 1, m):
                    c = lst[k]
                    if gcd_table[a][c] == gab and gcd_table[b][c] == gab:
                        return False
        return True

    def greedy_build(order, rng):
        mask = 0
        for x in order:
            if can_add_mask(mask, x):
                mask |= (1 << x)
        return mask

    def local_search(mask, rng):
        """Improve mask via add-first, then 1-for-1 swaps, then 2-for-1 swaps."""
        best = mask
        best_size = best.bit_count()

        # Phase 1: add any single element
        while True:
            added = False
            remaining = [x for x in range(1, N) if not (best >> x) & 1]
            rng.shuffle(remaining)
            for x in remaining:
                if can_add_mask(best, x):
                    best |= (1 << x)
                    best_size += 1
                    added = True
                    break
            if not added:
                break

        # Phase 2: 1-for-1 swaps
        improved = True
        while improved:
            improved = False
            elems = [x for x in range(1, N) if (best >> x) & 1]
            remaining = [x for x in range(1, N) if not (best >> x) & 1]
            rng.shuffle(elems)
            rng.shuffle(remaining)
            for u in elems:
                if time.time() > deadline - 3:
                    break
                trial = best & ~(1 << u)
                for x in remaining:
                    if can_add_mask(trial, x):
                        new_mask = trial | (1 << x)
                        if new_mask != best:
                            best = new_mask
                            best_size = best.bit_count()
                            improved = True
                            break
                if time.time() > deadline - 3:
                    break

        # Phase 3: 2-for-1 swaps (remove two, add up to three)
        improved = True
        while improved:
            improved = False
            elems = [x for x in range(1, N) if (best >> x) & 1]
            remaining = [x for x in range(1, N) if not (best >> x) & 1]
            rng.shuffle(elems)
            rng.shuffle(remaining)
            tried = 0
            for i in range(len(elems)):
                if tried > 200 or time.time() > deadline - 3:
                    break
                u = elems[i]
                for j in range(i + 1, len(elems)):
                    tried += 1
                    if tried > 200 or time.time() > deadline - 3:
                        break
                    v = elems[j]
                    trial = best & ~(1 << u) & ~(1 << v)
                    added_any = False
                    for x in remaining:
                        if can_add_mask(trial, x):
                            trial |= (1 << x)
                            added_any = True
                    if added_any and trial.bit_count() > best_size:
                        best = trial
                        best_size = best.bit_count()
                        improved = True
                        break
                if improved:
                    break

        return best

    best_mask = 0

    # Strategy 1: structured seeds
    seeds = []
    # Powers of 2
    seeds.append([1, 2, 4, 8, 16, 32, 64, 128])
    # Numbers coprime to 6
    seeds.append([x for x in range(1, N) if gcd(x, 6) == 1])
    # Multiples of 4
    seeds.append([x for x in range(4, N, 4)])
    # Odd numbers
    seeds.append([x for x in range(1, N, 2)])
    # High numbers (65..128)
    seeds.append(list(range(65, N)))

    for seed in seeds:
        # Validate seed is itself valid before using
        if is_valid_final(seed):
            mask = 0
            for x in seed:
                mask |= (1 << x)
            improved = local_search(mask, Random(hash(tuple(seed)) & 0xFFFFFFFF))
            if improved.bit_count() > best_mask.bit_count():
                best_mask = improved

    # Strategy 2: random restarts (bitmask-based, fast)
    rng = Random(12345)
    restart = 0
    while time.time() < deadline - 5:
        restart += 1
        order = list(range(1, N))
        rng.shuffle(order)
        mask = greedy_build(order, rng)
        improved = local_search(mask, rng)
        if improved.bit_count() > best_mask.bit_count():
            best_mask = improved

    # Extract final list
    result = [x for x in range(1, N) if (best_mask >> x) & 1]

    # Final exact validity check; if invalid, fall back to a safe small set
    if not result:
        return [1]
    if not is_valid_final(result):
        # Fallback: build a guaranteed-valid set greedily in sorted order
        mask = 0
        for x in range(1, N):
            if can_add_mask(mask, x):
                mask |= (1 << x)
        result = [x for x in range(1, N) if (mask >> x) & 1]
        if not result:
            return [1]

    return sorted(result)
