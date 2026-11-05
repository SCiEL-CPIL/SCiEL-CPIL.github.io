"""Greedy covering with bitset acceleration + redundant-block local search."""
# EVOLVE-BLOCK-START
import time
import itertools


def construct(v, k, t):
    deadline = time.time() + 135.0  # safety margin before 150s kill

    # Precompute all t-subsets and map each to a dense index.
    combos = list(itertools.combinations(range(v), t))
    ncomb = len(combos)
    # For each point, the bitset of t-subsets that contain it.
    pt_bits = [0] * v
    for i, c in enumerate(combos):
        bit = 1 << i
        for x in c:
            pt_bits[x] |= bit

    def bits_of_block(block):
        m = 0
        for x in block:
            m |= pt_bits[x]
        return m

    # --- Greedy construction (bitset-accelerated); always runs to completion ---
    uncovered = (1 << ncomb) - 1
    blocks = []
    while uncovered:
        low = uncovered & (-uncovered)
        i = low.bit_length() - 1
        block = list(combos[i])
        while len(block) < k:
            best = -1
            bestg = -1
            for x in range(v):
                if x in block:
                    continue
                g = (uncovered & pt_bits[x]).bit_count()
                if g > bestg:
                    bestg = g
                    best = x
            block.append(best)
        block.sort()
        blocks.append(block)
        uncovered &= ~bits_of_block(block)

    # --- Local search: drop redundant blocks (deadline-bounded) ---
    n = len(blocks)
    masks = [bits_of_block(b) for b in blocks]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            others = 0
            for j in range(n):
                if j != i:
                    others |= masks[j]
            if (masks[i] & ~others) == 0:
                blocks.pop(i)
                masks.pop(i)
                n -= 1
                improved = True
                break

    return blocks
# EVOLVE-BLOCK-END
