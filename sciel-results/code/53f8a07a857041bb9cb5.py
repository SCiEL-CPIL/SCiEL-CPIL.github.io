"""Stronger covering design C(19,7,4): randomized greedy + dominance deletion + local search."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

_DEADLINE = 110.0          # hard safety margin before the 120s wall-clock kill
_START = None


def _now():
    return time.perf_counter() - _START


def construct(v, k, t, _seed=12345):
    global _START
    _START = _now()
    rng = random.Random(_seed)

    all_points = list(range(v))
    all_blocks = list(itertools.combinations(all_points, k))
    block_to_tsets = [set(itertools.combinations(b, t)) for b in all_blocks]
    tset_to_blocks = defaultdict(list)
    for bi, ts in enumerate(block_to_tsets):
        for ts_ in ts:
            tset_to_blocks[ts_].append(bi)
    all_tsets = list(tset_to_blocks.keys())
    n_tsets = len(all_tsets)
    full_set = set(all_tsets)
    potential = [len(tset_to_blocks[ts]) for ts in all_tsets]

    def cov_of(blocks):
        cov = set()
        for b in blocks:
            for ts_ in itertools.combinations(b, t):
                cov.add(ts_)
        return cov

    def is_valid(blocks):
        return cov_of(blocks) == full_set

    def greedy_build():
        active = set(range(n_tsets))
        blocks = []
        while active and _now() < _DEADLINE:
            if len(active) <= 8:
                candidates = list(active)
            else:
                candidates = rng.sample(sorted(active), 8)
            seed_ts = max(candidates, key=lambda i: (potential[i], rng.random()))
            block = list(all_tsets[seed_ts])
            while len(block) < k and _now() < _DEADLINE:
                block_set = set(block)
                best_x = None
                best_gain = -1
                best_rand = -1.0
                for x in all_points:
                    if x in block_set:
                        continue
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        cand = tuple(sorted(rest + (x,)))
                        if cand in active:
                            gain += 1
                    r = rng.random()
                    if gain > best_gain or (gain == best_gain and gain > 0 and r > best_rand):
                        best_gain = gain
                        best_x = x
                        best_rand = r
                if best_x is None:
                    break
                block.append(best_x)
            block.sort()
            blocks.append(block)
            for ts_ in itertools.combinations(block, t):
                active.discard(ts_)
        return blocks

    def dominance_delete(blocks):
        blocks = list(blocks)
        changed = True
        while changed and _now() < _DEADLINE:
            changed = False
            for i in range(len(blocks)):
                if _now() >= _DEADLINE:
                    return blocks
                others_cov = set()
                for j, b in enumerate(blocks):
                    if j == i:
                        continue
                    for ts_ in itertools.combinations(b, t):
                        others_cov.add(ts_)
                unique = set(itertools.combinations(blocks[i], t)) - others_cov
                if not unique:
                    blocks.pop(i)
                    changed = True
                    break
        return blocks

    def local_search(blocks):
        blocks = list(blocks)
        improved = True
        while improved and _now() < _DEADLINE:
            improved = False
            n = len(blocks)
            for i in range(n):
                if _now() >= _DEADLINE:
                    return blocks
                for j in range(i + 1, n):
                    if _now() >= _DEADLINE:
                        return blocks
                    rest_cov = set()
                    for m, b in enumerate(blocks):
                        if m == i or m == j:
                            continue
                        for ts_ in itertools.combinations(b, t):
                            rest_cov.add(ts_)
                    need = set(itertools.combinations(blocks[i], t)) | set(itertools.combinations(blocks[j], t))
                    need -= rest_cov
                    for bi, ts_list in enumerate(block_to_tsets):
                        if ts_list.issuperset(need):
                            new_block = list(all_blocks[bi])
                            if new_block in blocks:
                                continue
                            new_blocks = [b for m, b in enumerate(blocks) if m != i and m != j]
                            new_blocks.append(new_block)
                            if is_valid(new_blocks):
                                blocks = new_blocks
                                improved = True
                                break
                    if improved:
                        break
        return blocks

    best = None
    best_len = 10 ** 9
    while _now() < _DEADLINE - 5:
        raw = greedy_build()
        if not is_valid(raw):
            continue
        reduced = dominance_delete(raw)
        reduced = local_search(reduced)
        if is_valid(reduced) and len(reduced) < best_len:
            best = reduced
            best_len = len(reduced)

    if best is None:
        best = greedy_build()
        if not is_valid(best):
            best = dominance_delete(best)
    return best
# EVOLVE-BLOCK-END
