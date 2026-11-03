"""Covering design C(v,k,t) via randomized greedy with urgency scoring + restarts."""
import itertools
import random
import time

_START = None


def _remaining_count(subset, used_counts, t):
    """Number of uncovered t-subsets that contain `subset`."""
    total = 0
    v = len(used_counts)
    for r in itertools.combinations(subset, t - len(subset)):
        c = 0
        for x in r:
            c += used_counts[x]
        total += max(0, t - c)
    return total


def _random_greedy(v, k, t, rng):
    """One randomized greedy pass. Returns (blocks, uncovered_count)."""
    used = [0] * v
    blocks = []
    while True:
        rem = 0
        for r in range(t + 1):
            c = 0
            for x in range(v):
                c += used[x]
            rem += max(0, t - c) * (1 if r == t else 1)
        # rem is total uncovered t-subsets
        if rem == 0:
            break
        # Pick a random uncovered t-subset to open the block
        while True:
            s = tuple(rng.randrange(v) for _ in range(t))
            s = tuple(sorted(s))
            c = 0
            for x in s:
                c += used[x]
            if c < t:
                break
        block = list(s)
        used_set = set(block)
        for x in block:
            used[x] += 1
        # Grow the block
        while len(block) < k:
            best_x = -1
            best_score = -1
            for x in range(v):
                if x in used_set:
                    continue
                # gain: uncovered t-subsets covered by adding x
                gain = 0
                for r in itertools.combinations(block, t - 1):
                    c = 0
                    for y in r:
                        c += used[y]
                    if c < t:
                        gain += 1
                # urgency bonus: prefer points that appear in many uncovered subsets
                urg = 0
                for r in itertools.combinations([x] + list(block), t):
                    c = 0
                    for y in r:
                        c += used[y]
                    if c < t:
                        urg += 1
                score = gain + 0.5 * urg + rng.random()
                if score > best_score:
                    best_score = score
                    best_x = x
            if best_x < 0:
                break
            block.append(best_x)
            used_set.add(best_x)
            used[best_x] += 1
        block.sort()
        blocks.append(block)
    # compute uncovered count
    return blocks, rem


def _verify(blocks, v, t):
    """Return number of uncovered t-subsets."""
    if not blocks:
        return sum(1 for _ in itertools.combinations(range(v), t))
    covered = set()
    for b in blocks:
        for r in itertools.combinations(b, t):
            covered.add(r)
    total = 0
    for r in itertools.combinations(range(v), t):
        if r not in covered:
            total += 1
    return total


def construct(v, k, t):
    global _START
    if _START is None:
        _START = time.time()
    deadline = _START + 150 - 5  # 5s safety margin
    rng = random.Random(12345)

    # Quick deterministic greedy to get an initial valid covering
    def deterministic_greedy():
        used = [0] * v
        blocks = []
        while True:
            rem = 0
            for r in range(t + 1):
                c = 0
                for x in range(v):
                    c += used[x]
                rem += max(0, t - c)
            if rem == 0:
                break
            # find an uncovered t-subset
            found = False
            for r in itertools.combinations(range(v), t):
                c = 0
                for x in r:
                    c += used[x]
                if c < t:
                    found = True
                    break
            if not found:
                break
            block = list(r)
            used_set = set(block)
            for x in block:
                used[x] += 1
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in used_set:
                        continue
                    gain = 0
                    for rr in itertools.combinations(block, t - 1):
                        c = 0
                        for y in rr:
                            c += used[y]
                        if c < t:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x < 0:
                    break
                block.append(best_x)
                used_set.add(best_x)
                used[best_x] += 1
            block.sort()
            blocks.append(block)
        return blocks

    best_blocks = deterministic_greedy()
    best_count = _verify(best_blocks, v, t)

    # Randomized greedy with multiple restarts
    while time.time() < deadline:
        rng.seed()
        blocks, rem = _random_greedy(v, k, t, rng)
        if rem == 0:
            cnt = len(blocks)
            if cnt < best_count:
                best_count = cnt
                best_blocks = blocks
        elif len(blocks) < best_count:
            # might still be valid
            cnt = _verify(blocks, v, t)
            if cnt == 0 and len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks

    # Final improvement: try to reduce block count via local search
    if time.time() < deadline - 10:
        # Try removing redundant blocks
        def is_redundant(blocks, idx, v, t):
            # check if block idx covers any t-subset not covered by others
            for r in itertools.combinations(blocks[idx], t):
                for j in range(len(blocks)):
                    if j == idx:
                        continue
                    if r in itertools.combinations(blocks[j], t):
                        break
                else:
                    return False
            return True

        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for i in range(len(best_blocks) - 1, -1, -1):
                if time.time() > deadline - 10:
                    break
                if is_redundant(best_blocks, i, v, t):
                    best_blocks.pop(i)
                    changed = True
                    break

    return best_blocks
