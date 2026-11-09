import time
import random
from functools import lru_cache

N = 512
MAX_CARDS = 48
DEADLINE = 85.0  # safety margin under 90s kill

def _subset_sum_bitset(nums, target):
    """Return True if some non-empty subset of nums sums to target.
    Uses a Python int as a bitset: bit s set means sum s is reachable."""
    bits = 1  # sum 0 reachable
    mask = (1 << (target + 1)) - 1
    for v in nums:
        if v <= target:
            bits |= (bits << v)
        bits &= mask
    return (bits >> target) & 1

def _is_valid(selected, candidate):
    """Check whether adding candidate to selected keeps the mean-free property.
    selected must already be valid. We check:
      (a) no existing a has candidate as mean of >=2 others in selected
      (b) candidate not mean of >=2 others in selected
      (c) no existing a is mean of >=2 others including candidate
    """
    k = len(selected)
    # (a) for each a in selected, is candidate = sum of >=2 others / count?
    for idx, a in enumerate(selected):
        others = selected[:idx] + selected[idx+1:]
        if not _subset_sum_bitset(others, candidate * k):
            continue
        # need to verify subset size in [2, k]
        # enumerate subsets of 'others' summing to candidate*k, check size
        # others length <= 47, but candidate*k up to 512*48 ~ 24576; do DP with sizes
        # build dp: dict size->bitset of sums
        dp = {0: 1}
        for v in others:
            for sz in list(dp.keys()):
                dp[sz+1] = dp.get(sz+1, 0) | (dp[sz] << v)
        for sz in range(2, len(others)+1):
            b = dp.get(sz, 0)
            if b and ((b >> (candidate*k)) & 1):
                return False
    # (b) candidate not mean of >=2 others in selected
    if _subset_sum_bitset(selected, candidate * k):
        dp = {0: 1}
        for v in selected:
            for sz in list(dp.keys()):
                dp[sz+1] = dp.get(sz+1, 0) | (dp[sz] << v)
        for sz in range(2, k+1):
            b = dp.get(sz, 0)
            if b and ((b >> (candidate*k)) & 1):
                return False
    # (c) existing a mean of >=2 others including candidate
    for idx, a in enumerate(selected):
        others = selected[:idx] + selected[idx+1:] + [candidate]
        if not _subset_sum_bitset(others, a * (k+1)):
            continue
        dp = {0: 1}
        for v in others:
            for sz in list(dp.keys()):
                dp[sz+1] = dp.get(sz+1, 0) | (dp[sz] << v)
        for sz in range(2, len(others)+1):
            b = dp.get(sz, 0)
            if b and ((b >> (a*(k+1))) & 1):
                return False
    return True

def _local_search(selected, rng, deadline):
    """Try to grow selected by repeatedly adding a valid candidate, and by swaps."""
    selected = list(selected)
    used = set(selected)
    while time.time() < deadline:
        # try adding
        improved = False
        for cand in sorted(random.sample(range(1, N+1), min(N, 200))):
            if cand in used:
                continue
            if _is_valid(selected, cand):
                selected.append(cand)
                used.add(cand)
                improved = True
                break
        if not improved:
            # try swap: remove one, add a better one
            swapped = False
            for i in range(len(selected)):
                for cand in range(1, N+1):
                    if cand in used:
                        continue
                    trial = selected[:i] + selected[i+1:]
                    if _is_valid(trial, cand):
                        selected = trial + [cand]
                        used.discard(selected[i])
                        used.add(cand)
                        swapped = True
                        break
                if swapped:
                    break
            if not swapped:
                break
    return selected

def construct():
    start = time.time()
    deadline = start + DEADLINE
    rng = random.Random(12345)
    best = []

    # deterministic warm start: greedy low-to-high
    selected = []
    for x in range(1, N+1):
        if _is_valid(selected, x):
            selected.append(x)
        if time.time() > deadline:
            break
    best = max(best, selected, key=len)

    # randomized greedy trials
    trial = 0
    while time.time() < deadline - 10:
        trial += 1
        order = list(range(1, N+1))
        rng.shuffle(order)
        selected = []
        for x in order:
            if _is_valid(selected, x):
                selected.append(x)
            if len(selected) >= MAX_CARDS:
                break
            if time.time() > deadline - 10:
                break
        if len(selected) > len(best):
            best = selected
        if len(best) >= MAX_CARDS:
            break

    # local search polish on best
    if time.time() < deadline - 2:
        polished = _local_search(best, rng, deadline - 2)
        if len(polished) > len(best):
            best = polished

    return best[:MAX_CARDS]
