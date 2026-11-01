"""Covering design C(24,7,3): fast greedy + redundancy removal + SA-style local search."""
import time
import random
import itertools
from collections import defaultdict


def construct(v, k, t):
    start = time.time()
    deadline = start + 145.0  # safety margin under 150s wall-clock

    # ---- precompute t-subset index ----
    all_tsub = list(itertools.combinations(range(v), t))
    n_tsub = len(all_tsub)
    tsub2idx = {ts: i for i, ts in enumerate(all_tsub)}

    # ---- for each pair (a,b): list of t-subset indices containing both ----
    # A block covers triple (a,b,x) exactly when x is added to a pair (a,b) in the block.
    pair2triples = {}
    for a in range(v):
        for b in range(a + 1, v):
            lst = []
            for x in range(v):
                if x == a or x == b:
                    continue
                idx = tsub2idx[(a, b, x)]
                lst.append(idx)
            pair2triples[(a, b)] = lst

    def build_greedy(rng, time_limit):
        """Randomized greedy: repeatedly build a block covering the most uncovered triples."""
        uncovered = set(range(n_tsub))
        blocks = []
        while uncovered and time.time() < time_limit:
            # pick an uncovered triple as the seed
            seed_idx = rng.choice(list(uncovered))
            a, b, c = all_tsub[seed_idx]
            block = [a, b, c]
            bset = {a, b, c}
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in bset:
                        continue
                    gain = 0
                    for y in block:
                        key = (x, y) if x < y else (y, x)
                        for idx in pair2triples[key]:
                            if idx in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x >= 0 and rng.random() < 0.5:
                        best_x = x
                if best_x < 0:
                    break
                block.append(best_x)
                bset.add(best_x)
            block.sort()
            blocks.append(block)
            # remove covered triples
            for ts in itertools.combinations(block, t):
                uncovered.discard(tsub2idx[ts])
        if uncovered:
            return None
        return blocks

    def remove_redundant(blocks, time_limit):
        """Drop blocks whose removal leaves no triple uncovered."""
        cover_count = defaultdict(int)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cover_count[tsub2idx[ts]] += 1
        result = list(blocks)
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            for i in range(len(result)):
                if time.time() > time_limit:
                    break
                block = result[i]
                removable = True
                for ts in itertools.combinations(block, t):
                    if cover_count[tsub2idx[ts]] == 1:
                        removable = False
                        break
                if removable:
                    for ts in itertools.combinations(block, t):
                        cover_count[tsub2idx[ts]] -= 1
                    result.pop(i)
                    changed = True
                    break
        return result

    def local_search(blocks, rng, time_limit):
        """SA-style: try replacing a block with a better one; probabilistically accept
        temporary increases in uncovered triples to escape local minima."""
        cover_count = defaultdict(int)
        for block in blocks:
            for ts in itertools.combinations(block, t):
                cover_count[tsub2idx[ts]] += 1
        uncovered = [i for i in range(n_tsub) if cover_count[i] == 0]
        best_blocks = [list(b) for b in blocks]
        best_uncovered = len(uncovered)
        cur_blocks = [list(b) for b in blocks]
        cur_uncovered = len(uncovered)
        T = 3.0
        t0 = time.time()
        while time.time() < time_limit:
            T *= 0.985
            i = rng.randrange(len(cur_blocks))
            old = cur_blocks[i]
            # build a replacement block targeting the most uncovered triples
            # seed with a random uncovered triple if any, else random triple
            if uncovered:
                seed_idx = rng.choice(uncovered)
                a, b, c = all_tsub[seed_idx]
            else:
                a, b, c = rng.sample(range(v), t)
            new = [a, b, c]
            nset = {a, b, c}
            while len(new) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in nset:
                        continue
                    gain = 0
                    for y in new:
                        key = (x, y) if x < y else (y, x)
                        for idx in pair2triples[key]:
                            if cover_count[idx] == 0:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x >= 0 and rng.random() < 0.5:
                        best_x = x
                if best_x < 0:
                    break
                new.append(best_x)
                nset.add(best_x)
            new.sort()
            if new == old:
                continue
            # compute change in uncovered count
            delta = 0
            for ts in itertools.combinations(old, t):
                idx = tsub2idx[ts]
                if cover_count[idx] == 1:
                    delta += 1
            for ts in itertools.combinations(new, t):
                idx = tsub2idx[ts]
                if cover_count[idx] == 0:
                    delta -= 1
            new_uncovered = cur_uncovered + delta
            if new_uncovered < cur_uncovered or (T > 0.01 and rng.random() < min(1.0, (-delta) / max(T, 1e-9))):
                # accept
                for ts in itertools.combinations(old, t):
                    cover_count[tsub2idx[ts]] -= 1
                for ts in itertools.combinations(new, t):
                    cover_count[tsub2idx[ts]] += 1
                cur_blocks[i] = new
                cur_uncovered = new_uncovered
                if cur_uncovered < best_uncovered:
                    best_uncovered = cur_uncovered
                    best_blocks = [list(b) for b in cur_blocks]
            # update uncovered list lazily (only when small)
            if best_uncovered == 0:
                break
        if best_uncovered == 0:
            return best_blocks
        # fall back to best found (may still be a full covering if it was)
        return best_blocks if best_uncovered <= cur_uncovered else [list(b) for b in blocks]

    best_blocks = None
    best_count = n_tsub

    # ---- main search loop ----
    seed = 0
    while time.time() < deadline:
        remaining = deadline - time.time()
        if remaining < 4:
            break
        rng = random.Random(seed)
        # greedy with a time budget
        gtime = min(remaining - 2, 8.0)
        blocks = build_greedy(rng, time.time() + gtime)
        if blocks is not None:
            blocks = remove_redundant(blocks, time.time() + min(remaining - 2, 3.0))
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = [list(b) for b in blocks]
            # local search on the best so far
            ls_time = time.time() + min(remaining - 2, 12.0)
            improved = local_search(best_blocks, rng, ls_time)
            improved = remove_redundant(improved, time.time() + min(remaining - 2, 3.0))
            if len(improved) < best_count:
                best_count = len(improved)
                best_blocks = [list(b) for b in improved]
        seed += 1

    if best_blocks is None:
        # deterministic fallback (should not happen)
        uncovered = set(range(n_tsub))
        blocks = []
        while uncovered:
            a, b, c = all_tsub[min(uncovered)]
            block = [a, b, c]
            bset = {a, b, c}
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in bset:
                        continue
                    gain = 0
                    for y in block:
                        key = (x, y) if x < y else (y, x)
                        for idx in pair2triples[key]:
                            if idx in uncovered:
                                gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x < 0:
                    break
                block.append(best_x)
                bset.add(best_x)
            block.sort()
            blocks.append(block)
            for ts in itertools.combinations(block, t):
                uncovered.discard(tsub2idx[ts])
        return blocks

    return best_blocks
