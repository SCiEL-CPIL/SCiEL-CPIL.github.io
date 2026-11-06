"""4-AP-free subset of [1,512]: greedy seed + fast local search with a time budget."""
import time
import random
import numpy as np

N = 512
NWORDS = (N + 63) // 64  # 8 words
DEADLINE = time.time() + 78.0  # safety margin under the 90s kill


def _can_add(bs, x0):
    """x0 in [0, N-1]; returns True if adding x0 creates no 4-AP."""
    if bs[x0 // 64] >> (x0 % 64) & 1:
        return False  # already present
    w = bs
    for d in range(1, 171):
        a = x0 - 3 * d
        if a >= 0 and w[a // 64] >> (a % 64) & 1 and w[(a + d) // 64] >> ((a + d) % 64) & 1 \
                and w[(a + 2 * d) // 64] >> ((a + 2 * d) % 64) & 1:
            return False
        a = x0 - 2 * d
        if a >= 0 and w[a // 64] >> (a % 64) & 1 and w[(a + d) // 64] >> ((a + d) % 64) & 1 \
                and w[(a + 2 * d) // 64] >> ((a + 2 * d) % 64) & 1:
            return False
        a = x0 - d
        if a >= 0 and w[a // 64] >> (a % 64) & 1 and w[(a + d) // 64] >> ((a + d) % 64) & 1 \
                and w[(a + 2 * d) // 64] >> ((a + 2 * d) % 64) & 1:
            return False
        a = x0 + 3 * d
        if a < N and w[a // 64] >> (a % 64) & 1 and w[(a - d) // 64] >> ((a - d) % 64) & 1 \
                and w[(a - 2 * d) // 64] >> ((a - 2 * d) % 64) & 1:
            return False
    return True


def _ap_count(bs, x0):
    """Number of 4-APs containing x0 (all four terms present)."""
    c = 0
    w = bs
    for d in range(1, 171):
        a = x0 - 3 * d
        if a >= 0 and w[a // 64] >> (a % 64) & 1 and w[(a + d) // 64] >> ((a + d) % 64) & 1 \
                and w[(a + 2 * d) // 64] >> ((a + 2 * d) % 64) & 1:
            c += 1
        a = x0 - 2 * d
        if a >= 0 and w[a // 64] >> (a % 64) & 1 and w[(a + d) // 64] >> ((a + d) % 64) & 1 \
                and w[(a + 2 * d) // 64] >> ((a + 2 * d) % 64) & 1:
            c += 1
        a = x0 - d
        if a >= 0 and w[a // 64] >> (a % 64) & 1 and w[(a + d) // 64] >> ((a + d) % 64) & 1 \
                and w[(a + 2 * d) // 64] >> ((a + 2 * d) % 64) & 1:
            c += 1
        a = x0 + 3 * d
        if a < N and w[a // 64] >> (a % 64) & 1 and w[(a - d) // 64] >> ((a - d) % 64) & 1 \
                and w[(a - 2 * d) // 64] >> ((a - 2 * d) % 64) & 1:
            c += 1
    return c


def _greedy(rng):
    bs = np.zeros(NWORDS, dtype=np.uint64)
    order = list(range(N))
    rng.shuffle(order)
    for x0 in order:
        if _can_add(bs, x0):
            bs[x0 // 64] |= np.uint64(1) << (x0 % 64)
    return bs


def _local_search(bs, rng):
    size = int(bs.sum() if hasattr(bs, 'sum') else 0)
    size = int(np.count_nonzero(bs))
    best = size
    best_bs = bs.copy()
    no_add = 0
    while time.time() < DEADLINE:
        # Try to add a random missing element.
        added = False
        for _ in range(6000):
            x0 = rng.randrange(N)
            if not (bs[x0 // 64] >> (x0 % 64) & 1):
                if _can_add(bs, x0):
                    bs[x0 // 64] |= np.uint64(1) << (x0 % 64)
                    size += 1
                    added = True
                    if size > best:
                        best = size
                        best_bs = bs.copy()
                    break
        if added:
            no_add = 0
            continue
        # No add found this round: remove a least-used element from some 4-AP.
        removed = False
        for _ in range(2000):
            x0 = rng.randrange(N)
            if not (bs[x0 // 64] >> (x0 % 64) & 1):
                continue
            c = _ap_count(bs, x0)
            if c > 0:
                # remove it
                bs[x0 // 64] &= ~(np.uint64(1) << (x0 % 64))
                size -= 1
                removed = True
                break
        if not removed:
            break
        no_add += 1
        if no_add > 40:
            break
    # restore best if we ever shrank
    if best > size:
        return best_bs
    return bs


def construct():
    rng = random.Random(12345)
    best_bs = None
    best_size = 0
    while time.time() < DEADLINE:
        bs = _greedy(rng)
        bs = _local_search(bs, rng)
        s = int(np.count_nonzero(bs))
        if s > best_size:
            best_size = s
            best_bs = bs.copy()
    if best_bs is None:
        best_bs = _greedy(random.Random(0))
    vals = [x0 + 1 for x0 in range(N) if best_bs[x0 // 64] >> (x0 % 64) & 1]
    return sorted(vals)
