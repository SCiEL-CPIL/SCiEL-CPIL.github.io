"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm

_START = time.time()
_DEADLINE = _START + 82.0  # safety margin under 90s kill


def construct():
    import time
    from math import gcd, lcm

    deadline = _DEADLINE
    N = 256

    # Process numbers in descending order: large numbers conflict with few partners
    # (lcm(x,y) <= 256 requires y | 256/gcd(x,256), a small set for large x).
    seq = list(range(N, 1, -1))  # descending
    m = len(seq)

    # Precompute conflict bitmask: conflict[i] has bit j set iff lcm(seq[i],seq[j]) <= 256
    conflict = [0] * m
    for i, x in enumerate(seq):
        g = gcd(x, N)
        limit = N // g
        mask = 0
        for j, y in enumerate(seq):
            if i == j:
                continue
            if y <= limit and limit % y == 0:
                mask |= (1 << j)
        conflict[i] = mask

    recip = [1.0 / seq[i] for i in range(m)]

    # Greedy seed: include in descending order if no conflict with already chosen
    chosen_mask = 0
    cur_sum = 0.0
    for i in range(m):
        if (conflict[i] & chosen_mask) == 0:
            chosen_mask |= (1 << i)
            cur_sum += recip[i]

    best_mask = chosen_mask
    best_sum = cur_sum

    def mask_of(posset):
        mk = 0
        for i in posset:
            mk |= (1 << i)
        return mk

    def sum_of(posset):
        return sum(recip[i] for i in posset)

    chosen_pos = set()
    for i in range(m):
        if (chosen_mask >> i) & 1:
            chosen_pos.add(i)

    # Local search: add, swap, and 2-for-1 moves until no improvement
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        base_mask = mask_of(chosen_pos)
        base_sum = sum_of(chosen_pos)

        # Move 1: add a single unchosen number
        for i in range(m):
            if i in chosen_pos:
                continue
            if (conflict[i] & base_mask) == 0:
                if base_sum + recip[i] > base_sum + 1e-12:
                    chosen_pos.add(i)
                    improved = True
                    break
        if improved:
            continue

        # Move 2: swap one chosen for one unchosen
        unchosen = [i for i in range(m) if i not in chosen_pos]
        best_swap = None
        best_gain = 1e-12
        for r in list(chosen_pos):
            mask_after = base_mask & ~(1 << r)
            for a in unchosen:
                if (conflict[a] & mask_after) == 0:
                    gain = recip[a] - recip[r]
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (r, a)
        if best_swap is not None:
            r, a = best_swap
            chosen_pos.remove(r)
            chosen_pos.add(a)
            improved = True
            continue

        # Move 3: remove one, add multiple (2-for-1 and beyond)
        for r in list(chosen_pos):
            mask_after = base_mask & ~(1 << r)
            addable = [a for a in unchosen if (conflict[a] & mask_after) == 0]
            if not addable:
                continue
            newmask = mask_after
            gain = -recip[r]
            to_add = []
            for a in sorted(addable, key=lambda t: -recip[t]):
                if (conflict[a] & newmask) == 0:
                    newmask |= (1 << a)
                    gain += recip[a]
                    to_add.append(a)
            if gain > 1e-12 and to_add:
                chosen_pos.remove(r)
                for a in to_add:
                    chosen_pos.add(a)
                improved = True
                break

    final_set = [seq[i] for i in chosen_pos]
    final_sum = sum(1.0 / x for x in final_set)

    # Verify the lcm constraint strictly
    valid = True
    for i in range(len(final_set)):
        for j in range(i + 1, len(final_set)):
            if lcm(final_set[i], final_set[j]) <= 256:
                valid = False
                break
        if not valid:
            break

    if valid and final_sum >= best_sum:
        best_mask = mask_of(chosen_pos)
        best_sum = final_sum
        best_set = final_set
    else:
        # reconstruct best_set from best_mask
        best_set = [seq[i] for i in range(m) if (best_mask >> i) & 1]

    if not best_set:
        best_set = [N]
    return best_set
# EVOLVE-BLOCK-END
