"""Construct the Witt design S(5,6,12) via exact cover."""
import time
from itertools import combinations

_DEADLINE = time.time() + 145.0

def _time_ok():
    return time.time() < _DEADLINE


def construct():
    n = 12
    all_5subs = list(combinations(range(n), 5))
    num_5subs = len(all_5subs)  # 792
    subset_to_idx = {s: i for i, s in enumerate(all_5subs)}

    all_6subs = list(combinations(range(n), 6))
    num_6subs = len(all_6subs)  # 924

    # For each 6-subset, compute which 5-subset indices it covers
    block_5subs = []
    for b in all_6subs:
        subs = [subset_to_idx[s] for s in combinations(b, 5)]
        block_5subs.append(subs)

    # For each 5-subset, which blocks cover it
    from collections import defaultdict
    sub_to_blocks = defaultdict(list)
    for bi, subs in enumerate(block_5subs):
        for s in subs:
            sub_to_blocks[s].append(bi)

    # Bitset for each block
    block_bitsets = []
    for subs in block_5subs:
        mask = 0
        for s in subs:
            mask |= (1 << s)
        block_bitsets.append(mask)

    all_mask = (1 << num_5subs) - 1

    best_solution = []

    def backtrack(uncovered_mask, selected):
        nonlocal best_solution
        if not _time_ok():
            return
        if uncovered_mask == 0:
            if len(selected) > len(best_solution):
                best_solution = list(selected)
            return

        # Find uncovered 5-subset with fewest candidate blocks
        min_cands = None
        min_count = float('inf')
        m = uncovered_mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            # Candidates: blocks covering idx that don't conflict
            cands = []
            for b in sub_to_blocks[idx]:
                if block_bitsets[b] & uncovered_mask == block_bitsets[b]:
                    cands.append(b)
            if len(cands) < min_count:
                min_count = len(cands)
                min_cands = cands
                if min_count == 0:
                    break
            m ^= lsb

        if min_count == 0:
            return

        for b in min_cands:
            selected.append(b)
            backtrack(uncovered_mask & ~block_bitsets[b], selected)
            selected.pop()

    backtrack(all_mask, [])

    if best_solution:
        return [list(all_6subs[i]) for i in best_solution]

    # Fallback greedy
    covered, blocks = set(), []
    for b in all_6subs:
        subsets = set(combinations(b, 5))
        if not any(s in covered for s in subsets):
            blocks.append(list(b))
            covered.update(subsets)
    return blocks
