"""Improved greedy covering for C(24,7,3) with precomputation and restarts."""

import itertools
import random
import time
import numpy as np


def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    n_triples = len(all_triples)
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}

    # uncovered_arr: numpy array, 1 = uncovered, 0 = covered
    uncovered_arr = np.ones(n_triples, dtype=np.uint8)

    def mark_block_covered(block):
        for tr in itertools.combinations(block, t):
            idx = triple_to_idx.get(tr)
            if idx is not None:
                uncovered_arr[idx] = 0

    def greedy_cover(seed_rng):
        nonlocal uncovered_arr
        uncovered_arr[:] = 1
        n_uncovered = n_triples
        blocks = []

        while n_uncovered > 0:
            idx = int(np.argmax(uncovered_arr))
            if uncovered_arr[idx] == 0:
                break
            seed_triple = all_triples[idx]

            block = list(seed_triple)
            block_set = set(block)
            remaining_points = [p for p in points if p not in block_set]

            while len(block) < k:
                best_gain = -1
                best_point = None
                for p in remaining_points:
                    gain = 0
                    for rest in itertools.combinations(block, t - 1):
                        tr = tuple(sorted(rest + (p,)))
                        tri_idx = triple_to_idx.get(tr)
                        if tri_idx is not None and uncovered_arr[tri_idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_point = p
                    elif gain == best_gain and best_point is not None:
                        if seed_rng.random() < 0.5:
                            best_point = p

                if best_point is None:
                    break

                block.append(best_point)
                block_set.add(best_point)
                remaining_points.remove(best_point)

            block.sort()
            blocks.append(tuple(block))
            mark_block_covered(tuple(block))
            n_uncovered = int(np.sum(uncovered_arr))

        return blocks

    def remove_redundant(blocks):
        if len(blocks) <= 1:
            return blocks

        triple_to_blocks = {}
        for bi, block in enumerate(blocks):
            for tr in itertools.combinations(block, t):
                if tr not in triple_to_blocks:
                    triple_to_blocks[tr] = []
                triple_to_blocks[tr].append(bi)

        keep = [True] * len(blocks)
        for bi, block in enumerate(blocks):
            redundant = True
            for tr in itertools.combinations(block, t):
                covering_blocks = triple_to_blocks.get(tr, [])
                if len(covering_blocks) == 1:
                    redundant = False
                    break
            if redundant:
                keep[bi] = False

        return [blocks[i] for i in range(len(blocks)) if keep[i]]

    # First deterministic run
    rng = random.Random(0)
    blocks = greedy_cover(rng)
    blocks = remove_redundant(blocks)
    best_blocks = blocks
    best_len = len(blocks)

    # More restarts with different seeds
    seed = 1
    while True:
        if time.time() > deadline - 5.0:
            break
        rng = random.Random(seed)
        blocks = greedy_cover(rng)
        blocks = remove_redundant(blocks)
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
        seed += 1

    return [list(b) for b in best_blocks]
