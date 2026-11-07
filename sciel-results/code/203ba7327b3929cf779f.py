"""Improved program: bitmask + local search with deadline guard."""
import time
import random

N = 128
DEADLINE = 55.0  # safety margin below 60s

def _is_valid(mask):
    # mask: int bitmask over elements 1..N (bit i set means i in set)
    # Valid iff no distinct a<b<c in S with a+b,a+c,b+c in S.
    # For each pair (a,b) in S with a<b and a+b in S, check no c in S
    # (c != a,b) with a+c in S and b+c in S.
    elems = [i for i in range(1, N + 1) if (mask >> i) & 1]
    m = len(elems)
    # Precompute membership via mask for O(1)
    def has(x):
        return x <= N and ((mask >> x) & 1) != 0
    for i in range(m):
        a = elems[i]
        for j in range(i + 1, m):
            b = elems[j]
            s = a + b
            if s > N:
                continue
            if not has(s):
                continue
            # s in S, a,b in S. Now find c in S, c != a,b, with a+c in S and b+c in S.
            # c must be in S. Iterate over S elements.
            for k in range(m):
                c = elems[k]
                if c == a or c == b:
                    continue
                if has(a + c) and has(b + c):
                    return False
    return True

def _popcount(x):
    return bin(x).count('1')

def _greedy_start():
    # Start from upper half (known valid, size 64) and try to add.
    mask = 0
    for i in range(N // 2 + 1, N + 1):
        mask |= (1 << i)
    # Try to add elements from lower half greedily
    for i in range(1, N // 2 + 1):
        if (mask >> i) & 1:
            continue
        new_mask = mask | (1 << i)
        if _is_valid(new_mask):
            mask = new_mask
    return mask

def _local_search(mask, time_limit, rng):
    best = mask
    best_size = _popcount(best)
    start = time.time()
    while time.time() < time_limit:
        # Try remove-one-add-many: pick a random element to remove,
        # then greedily add back as many as possible (including the removed one last).
        elems = [i for i in range(1, N + 1) if (best >> i) & 1]
        if not elems:
            break
        remove = rng.choice(elems)
        trial = best & ~(1 << remove)
        # Greedily add elements not in trial, in random order, keep if valid
        candidates = [i for i in range(1, N + 1) if not (trial >> i) & 1]
        rng.shuffle(candidates)
        cur = trial
        for i in candidates:
            new_mask = cur | (1 << i)
            if _is_valid(new_mask):
                cur = new_mask
        if _popcount(cur) > best_size:
            best = cur
            best_size = _popcount(best)
        # Also try simple single-add
        for i in range(1, N + 1):
            if (best >> i) & 1:
                continue
            new_mask = best | (1 << i)
            if _is_valid(new_mask):
                best = new_mask
                best_size += 1
                break
    return best

def construct():
    start = time.time()
    deadline = start + DEADLINE
    rng = random.Random(12345)
    best = _greedy_start()
    best_size = _popcount(best)
    # Local search with time budget
    while time.time() < deadline - 2.0:
        mask = _local_search(best, deadline - 2.0, rng)
        if _popcount(mask) > best_size:
            best = mask
            best_size = _popcount(best)
        else:
            # Perturb: start fresh from greedy with different seed occasionally
            if rng.random() < 0.3:
                mask = _greedy_start()
    result = [i for i in range(1, N + 1) if (best >> i) & 1]
    return list(result)
