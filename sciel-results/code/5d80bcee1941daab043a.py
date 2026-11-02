"""Block-elimination + replacement local search over bitset coverage."""
# EVOLVE-BLOCK-START
import itertools
import time


def construct(v, k, t):
    start = time.time()
    deadline = start + 112.0          # safety margin vs 120s (150s kill)

    # ---------- precompute ----------
    combos = list(itertools.combinations(range(v), t))
    total = len(combos)
    idx = {c: i for i, c in enumerate(combos)}
    # pt_or[x]: big-int whose bit i is 1 iff t-subset i contains point x
    pt_or = [0] * v
    for i, c in enumerate(combos):
        bit = 1 << i
        for x in c:
            pt_or[x] |= bit

    def block_bits(S):
        b = 0
        for x in S:
            b |= pt_or[x]
        return b

    def popcount(x):
        return x.bit_count()

    # ---------- greedy build (randomized, bitset) ----------
    def greedy_build():
        unc = (1 << total) - 1
        blocks = []
        while unc:
            low = unc & (-unc)
            i = low.bit_length() - 1
            block = list(combos[i])
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    g = 0
                    for rest in itertools.combinations(block, t - 1):
                        g += (unc >> idx[rest + (x,)]) & 1
                    if g > best_gain:
                        best_gain = g
                        best_x = x
                block.append(best_x)
            block.sort()
            blocks.append(block)
            unc &= ~block_bits(block)
            if time.time() > deadline:
                break
        return blocks

    # ---------- local search ----------
    def local_search(blocks):
        bits = [block_bits(b) for b in blocks]
        while True:
            improved = False
            # ---- elimination pass (drop redundant blocks) ----
            i = 0
            while i < len(bits):
                others = 0
                for j in range(len(bits)):
                    if j != i:
                        others |= bits[j]
                dropped = bits[i] & ~others
                if popcount(dropped) == 0:
                    bits.pop(i)
                    improved = True
                    continue
                i += 1
            # ---- replacement pass (swap a block for a better one) ----
            if len(bits) >= 2:
                for i in range(len(bits)):
                    if len(bits) < 2:
                        break
                    others = 0
                    for j in range(len(bits)):
                        if j != i:
                            others |= bits[j]
                    b = bits[i]
                    unique_i = b & ~others
                    bpts = blocks[i]
                    cand_set = {}
                    for a in range(v):
                        if a in bpts:
                            continue
                        for drop in bpts:
                            key = frozenset(bpts) - {drop} | {a}
                            cand_set[key] = 1
                    tried = 0
                    for cand in cand_set:
                        tried += 1
                        if tried > 40:
                            break
                        c = block_bits(cand)
                        if c == b:
                            continue
                        gain = popcount(unique_i) - popcount(c & ~others)
                        if gain < 0:
                            bits[i] = c
                            blocks[i] = sorted(cand)
                            improved = True
                            break
                    if improved:
                        break
            if time.time() > deadline:
                break
            if not improved:
                break
        return blocks

    # ---------- driver ----------
    best = greedy_build()
    if time.time() > deadline:
        return best
    best = local_search(best)
    best_n = len(best)

    while time.time() < deadline - 20.0:
        cand = greedy_build()
        if time.time() > deadline:
            break
        cand = local_search(cand)
        if len(cand) < best_n:
            best = cand
            best_n = len(cand)

    return best
# EVOLVE-BLOCK-END
