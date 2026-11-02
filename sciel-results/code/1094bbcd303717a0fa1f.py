"""Covering design C(19,7,4) — fast incremental-gain greedy + deletion."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    start = time.time()
    deadline = start + 130

    def time_ok():
        return time.time() < deadline

    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}

    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)

    # Precompute block_tsubs: for each block, list of t-subset indices it covers
    block_tsubs = [None] * n_blocks
    for bi, blk in enumerate(all_blocks):
        block_tsubs[bi] = [t_sub_to_idx[sub] for sub in itertools.combinations(blk, t)]

    # Precompute tsub_to_blocks: for each t-subset, list of block indices covering it
    tsub_to_blocks = [[] for _ in range(n_t)]
    for bi, subs in enumerate(block_tsubs):
        for si in subs:
            tsub_to_blocks[si].append(bi)

    best_blocks = None
    best_count = float('inf')

    n_restarts = 80

    for restart in range(n_restarts):
        if not time_ok():
            break
        random.seed(42 + restart * 7919)

        covered = bytearray(n_t)
        covered_count = 0
        selected = []

        # gain[bi] = number of currently-uncovered t-subsets covered by block bi
        gain = [len(block_tsubs[bi]) for bi in range(n_blocks)]

        while covered_count < n_t and time_ok():
            # Find max gain block — scan all (50388 * simple compare = fast)
            best_bi = -1
            best_g = -1
            for bi in range(n_blocks):
                g = gain[bi]
                if g > best_g:
                    best_g = g
                    best_bi = bi
                    if best_g == 35:  # max possible C(7,4)
                        break

            if best_bi < 0 or best_g <= 0:
                break

            selected.append(best_bi)
            for si in block_tsubs[best_bi]:
                if not covered[si]:
                    covered[si] = 1
                    covered_count += 1
                    # Update gain for all blocks that also cover si
                    for bj in tsub_to_blocks[si]:
                        gain[bj] -= 1

        # Deletion phase
        if time_ok() and covered_count == n_t:
            tsub_count = [0] * n_t
            for bi in selected:
                for si in block_tsubs[bi]:
                    tsub_count[si] += 1

            changed = True
            while changed and time_ok():
                changed = False
                for i in range(len(selected)):
                    if not time_ok():
                        break
                    bi = selected[i]
                    redundant = True
                    for si in block_tsubs[bi]:
                        if tsub_count[si] <= 1:
                            redundant = False
                            break
                    if redundant:
                        for si in block_tsubs[bi]:
                            tsub_count[si] -= 1
                        selected.pop(i)
                        changed = True
                        break

        if len(selected) < best_count:
            best_count = len(selected)
            best_blocks = [all_blocks[bi] for bi in selected]

        if best_count <= 148:
            break

    if best_blocks is None:
        return []

    return [list(b) for b in best_blocks]
# EVOLVE-BLOCK-END
