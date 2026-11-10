"""Covering design C(v,k,t): bitmask greedy + stochastic restarts + pruning, with deadline guard."""
import itertools
import random
import time


def construct(v, k, t):
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s

    all_t = list(itertools.combinations(range(v), t))
    n_t = len(all_t)
    mask_of = [0] * n_t
    for i, c in enumerate(all_t):
        m = 0
        for x in c:
            m |= 1 << x
        mask_of[i] = m

    # Candidate extension masks: for each (t-1)-subset, which points can extend it
    ext_mask = {}
    for c in itertools.combinations(range(v), t - 1):
        key = tuple(c)
        used = 0
        for x in c:
            used |= 1 << x
        m = ((1 << v) - 1) & ~used
        ext_mask[key] = m

    def count_uncovered_in_block(block_mask, counts, mask_of):
        # number of t-subsets of block_mask with count==0
        total = 0
        for c in itertools.combinations(range(v), t):
            m = 0
            for x in c:
                m |= 1 << x
            if (m & block_mask) == m and counts[c] == 0:
                total += 1
        return total

    def greedy_cover(rnd):
        counts = [0] * n_t
        blocks = []
        for _ in range(200000):
            if time.time() > deadline:
                return None
            # find an uncovered t-subset; prefer one with fewest candidate extensions
            best = None
            best_score = None
            # sample a random start window to add stochasticity
            start_idx = rnd.randrange(n_t)
            for offset in range(n_t):
                i = (start_idx + offset) % n_t
                if counts[i] == 0:
                    # score = number of points that extend this t-subset (fewer = more constrained)
                    score = bin(ext_mask[tuple(all_t[i])]).count('1')
                    if best is None or score < best_score:
                        best = i
                        best_score = score
                        if score <= 1:
                            break
            if best is None:
                break  # all covered
            seed = list(all_t[best])
            block = list(seed)
            block_mask = 0
            for x in block:
                block_mask |= 1 << x
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if (block_mask >> x) & 1:
                        continue
                    gain = 0
                    for c in itertools.combinations(block, t - 1):
                        m = 0
                        for y in c:
                            m |= 1 << y
                        m |= 1 << x
                        idx = None
                        # find index of this t-subset
                        # precomputed lookup would be better; do linear via dict
                        for j, cj in enumerate(all_t):
                            if (tuple(cj) == tuple(sorted(c + (x,)))):
                                idx = j
                                break
                        if idx is not None and counts[idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x < 0:
                    break
                block.append(best_x)
                block_mask |= 1 << best_x
            # update counts
            for c in itertools.combinations(block, t):
                m = 0
                for y in c:
                    m |= 1 << y
                # find index
                for j, cj in enumerate(all_t):
                    if tuple(cj) == tuple(sorted(c)):
                        counts[j] += 1
                        break
            blocks.append(sorted(block))
            if time.time() > deadline:
                return None
        # check full coverage
        if any(c == 0 for c in counts):
            return None
        return blocks

    def prune(blocks, counts, mask_of, rnd):
        # repeatedly try to remove a block if it doesn't uncover any t-subset
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            order = list(range(len(blocks)))
            rnd.shuffle(order)
            for bi in order:
                b = blocks[bi]
                # count how many t-subsets of b are uniquely covered (count==1)
                unique = 0
                for c in itertools.combinations(b, t):
                    m = 0
                    for y in c:
                        m |= 1 << y
                    for j, cj in enumerate(all_t):
                        if tuple(cj) == tuple(sorted(c)):
                            if counts[j] == 1:
                                unique += 1
                            break
                if unique == 0:
                    # safe to remove
                    for c in itertools.combinations(b, t):
                        m = 0
                        for y in c:
                            m |= 1 << y
                        for j, cj in enumerate(all_t):
                            if tuple(cj) == tuple(sorted(c)):
                                counts[j] -= 1
                                break
                    blocks.pop(bi)
                    improved = True
                    break
        return blocks

    best = None
    best_len = 10**9
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        rnd = random.Random(attempts * 12345 + 7)
        blocks = greedy_cover(rnd)
        if blocks is None:
            continue
        # build counts for pruning
        counts = [0] * n_t
        for b in blocks:
            for c in itertools.combinations(b, t):
                m = 0
                for y in c:
                    m |= 1 << y
                for j, cj in enumerate(all_t):
                    if tuple(cj) == tuple(sorted(c)):
                        counts[j] += 1
                        break
        blocks = prune(blocks, counts, mask_of, rnd)
        if len(blocks) < best_len:
            best_len = len(blocks)
            best = blocks
            if best_len <= 335:
                break
    if best is None:
        # fallback: pure deterministic greedy (seed-like) but faster via bitmask
        rnd = random.Random(0)
        blocks = greedy_cover(rnd)
        if blocks is None:
            # last resort: trivial (may be invalid but returns something)
            blocks = []
            counts = [0] * n_t
            for c in all_t:
                blocks.append(list(c) + [x for x in range(v) if x not in c][:k - t])
                for cc in itertools.combinations(blocks[-1], t):
                    m = 0
                    for y in cc:
                        m |= 1 << y
                    for j, cj in enumerate(all_t):
                        if tuple(cj) == tuple(sorted(cc)):
                            counts[j] += 1
                            break
            best = blocks
        else:
            best = blocks
    return best
