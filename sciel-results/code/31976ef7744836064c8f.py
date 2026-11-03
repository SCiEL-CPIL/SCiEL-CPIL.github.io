# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import comb
import numpy as np


def _combinations_lex(seq):
    n = len(seq)
    t = n
    if t <= 0:
        yield ()
        return
    idx = list(range(n))
    while True:
        yield tuple(idx[:t])
        for i in range(t - 1, -1, -1):
            if idx[i] != i + (n - t):
                break
        else:
            return
        idx[i] += 1
        for j in range(i + 1, t):
            idx[j] = idx[j - 1] + 1


def construct(v, k, t):
    start = time.time()
    deadline = start + 108.0  # safety margin under 120s

    total_tsub = comb(v, t)

    # all t-combos in lex order
    all_combos = list(_combinations_lex(range(v)))
    rank_map = {c: i for i, c in enumerate(all_combos)}

    # point_rank[x]: dict frozenset((t-1)-subset of others) -> rank of sorted(sub+(x,))
    point_rank = [None] * v
    for x in range(v):
        others_seq = [y for y in range(v) if y != x]
        d = {}
        for sub in _combinations_lex(others_seq):
            if len(sub) == t - 1:
                full = tuple(sorted(sub + (x,)))
                d[frozenset(sub)] = rank_map[full]
        point_rank[x] = d

    covered = np.zeros(total_tsub, dtype=bool)

    def build_one(seed):
        nonlocal covered
        rng = random.Random(seed)
        covered[:] = False
        uncov = set(range(total_tsub))
        blocks = []
        while uncov:
            idx = rng.choice(list(uncov))
            start_sub = all_combos[idx]
            block = list(start_sub)
            block_set = set(block)
            while len(block) < k:
                best_gain = -1
                best_ties = []
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for sub in combinations(block, t - 1):
                        r = point_rank[x].get(frozenset(sub))
                        if r is not None and not covered[r]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_ties = [x]
                    elif gain == best_gain and gain > 0:
                        best_ties.append(x)
                if best_gain <= 0:
                    unused = [x for x in range(v) if x not in block_set]
                    if not unused:
                        break
                    x = rng.choice(unused)
                else:
                    x = rng.choice(best_ties)
                block.append(x)
                block_set.add(x)
            block.sort()
            new_cov = []
            for sub in combinations(block, t):
                r = rank_map[tuple(sub)]
                if not covered[r]:
                    covered[r] = True
                    new_cov.append(r)
            uncov.difference_update(new_cov)
            blocks.append(block)
            if time.time() > deadline:
                break
        return blocks

    def remove_redundancy(blocks):
        n = len(blocks)
        block_sets = []
        for b in blocks:
            s = set()
            for sub in combinations(b, t):
                s.add(rank_map[tuple(sub)])
            block_sets.append(s)
        kept = [False] * n
        covered_by_kept = set()
        for i in range(n - 1, -1, -1):
            if block_sets[i] - covered_by_kept:
                kept[i] = True
                covered_by_kept |= block_sets[i]
        return [blocks[i] for i in range(n) if kept[i]]

    best_blocks = None
    best_len = float('inf')
    seed = 0
    while time.time() < deadline - 3.0:
        blocks = build_one(seed)
        # quick coverage check
        cov = np.zeros(total_tsub, dtype=bool)
        for b in blocks:
            for sub in combinations(b, t):
                cov[rank_map[tuple(sub)]] = True
        if cov.all():
            if len(blocks) < best_len:
                try:
                    red = remove_redundancy(blocks)
                    cov2 = np.zeros(total_tsub, dtype=bool)
                    for b in red:
                        for sub in combinations(b, t):
                            cov2[rank_map[tuple(sub)]] = True
                    if cov2.all() and len(red) < best_len:
                        best_blocks = red
                        best_len = len(red)
                    else:
                        best_blocks = blocks
                        best_len = len(blocks)
                except Exception:
                    best_blocks = blocks
                    best_len = len(blocks)
        seed += 1

    if best_blocks is None:
        best_blocks = build_one(seed)
    return best_blocks
# EVOLVE-BLOCK-END
