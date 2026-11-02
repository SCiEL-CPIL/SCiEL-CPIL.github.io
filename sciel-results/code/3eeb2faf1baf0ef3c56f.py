"""Covering design C(19,7,4): randomized weighted greedy + local search."""
import itertools
import random


def construct(v, k, t):
    subs = list(itertools.combinations(range(v), t))
    idx = {s: i for i, s in enumerate(subs)}

    def build_block(seed, uncovered, rng):
        block = list(seed)
        bset = set(block)
        while len(block) < k:
            best_x, best_g = None, -1
            for x in range(v):
                if x in bset:
                    continue
                g = 0
                for rest in itertools.combinations(block, t - 1):
                    s = tuple(sorted(rest + (x,)))
                    if s in uncovered:
                        g += 1
                if g > best_g:
                    best_g, best_x = g, x
            if best_x is None:
                cands = [x for x in range(v) if x not in bset]
                best_x = rng.choice(cands) if cands else 0
            block.append(best_x)
            bset.add(best_x)
        return sorted(block)

    def greedy(rng):
        uncovered = set(subs)
        blocks = []
        while uncovered:
            # Pick seed: the uncovered subset that appears in the most potential blocks
            seed = next(iter(uncovered))
            blk = build_block(seed, uncovered, rng)
            blocks.append(blk)
            uncovered -= set(itertools.combinations(blk, t))
        return blocks

    def local_search(blocks, rng, rounds=10):
        blocks = [list(b) for b in blocks]
        for _ in range(rounds):
            changed = False
            order = list(range(len(blocks)))
            rng.shuffle(order)
            for bi in order:
                if bi >= len(blocks):
                    continue
                b = blocks[bi]
                # Check if every t-subset of b is covered by some other block
                all_covered = True
                for s in itertools.combinations(b, t):
                    covered = False
                    for j, other in enumerate(blocks):
                        if j == bi:
                            continue
                        if s in set(itertools.combinations(other, t)):
                            covered = True
                            break
                    if not covered:
                        all_covered = False
                        break
                if all_covered:
                    blocks.pop(bi)
                    changed = True
            if not changed:
                break
        return blocks

    best = None
    best_len = float('inf')
    for r in range(12):
        rng = random.Random(100 + r)
        blks = greedy(rng)
        if len(blks) < best_len:
            best_len = len(blks)
            best = blks
    if best is not None:
        best = local_search(best, random.Random(999), rounds=8)
    return best
