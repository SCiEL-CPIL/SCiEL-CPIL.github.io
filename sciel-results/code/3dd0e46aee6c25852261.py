import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin for 150s limit

    # Precompute all 5-subsets and map to indices
    all_5subsets = list(combinations(range(12), 5))
    subset_to_idx = {s: i for i, s in enumerate(all_5subsets)}
    n_5subs = len(all_5subsets)  # 792

    # Precompute all 6-subsets and their bitmask of covered 5-subsets
    all_6subsets = list(combinations(range(12), 6))
    n_6subs = len(all_6subsets)  # 924
    block_masks = [0] * n_6subs
    for i, b in enumerate(all_6subsets):
        mask = 0
        for sub in combinations(b, 5):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
        block_masks[i] = mask

    best_blocks = []
    best_count = 0

    def greedy_packing(order):
        covered = 0
        blocks_found = []
        for idx in order:
            mask = block_masks[idx]
            if (covered & mask) == 0:
                blocks_found.append(all_6subsets[idx])
                covered |= mask
                if len(blocks_found) == 132:
                    break
        return blocks_found

    # Deterministic orderings
    # 1. Lexicographic
    order_lex = list(range(n_6subs))
    b_lex = greedy_packing(order_lex)
    if len(b_lex) > best_count:
        best_count = len(b_lex)
        best_blocks = b_lex

    # 2. Reverse Lexicographic
    order_rev = list(range(n_6subs - 1, -1, -1))
    b_rev = greedy_packing(order_rev)
    if len(b_rev) > best_count:
        best_count = len(b_rev)
        best_blocks = b_rev

    # Random restarts
    if time.time() < deadline:
        rng = random.Random(42)
        # Run as many restarts as possible
        while time.time() < deadline:
            order_rand = list(range(n_6subs))
            rng.shuffle(order_rand)
            b_rand = greedy_packing(order_rand)
            if len(b_rand) > best_count:
                best_count = len(b_rand)
                best_blocks = b_rand
                if best_count == 132:
                    break

    return [list(b) for b in best_blocks]
