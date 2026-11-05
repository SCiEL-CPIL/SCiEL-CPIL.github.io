"""Superpermutation via recursive construction + VNS reordering of blocks."""
import time
import random

# EVOLVE-BLOCK-START
def _overlap_len(a, b):
    """Largest k such that suffix of a (len k) == prefix of b (len k)."""
    n = min(len(a), len(b))
    k = n
    while k > 0:
        if a[-k:] == b[:k]:
            return k
        k -= 1
    return 0

def _standard_word(n):
    """Classic recursive construction word for n (list of ints)."""
    if n == 1:
        return [1]
    word = _standard_word(n-1)
    blocks = []
    seen = set()
    for i in range(len(word) - (n-1) + 1):
        w = tuple(word[i:i+n-1])
        if len(set(w)) == n-1 and w not in seen:
            seen.add(w)
            blocks.append(w)
    new = []
    for p in blocks:
        block = list(p) + [n] + list(p)
        k = _overlap_len(new, block) if new else 0
        new.extend(block[k:])
    return new

def _extract_blocks(word, m):
    """Extract distinct m-length permutation blocks in order of appearance."""
    blocks = []
    seen = set()
    for i in range(len(word) - m + 1):
        w = tuple(word[i:i+m])
        if len(set(w)) == m and w not in seen:
            seen.add(w)
            blocks.append(w)
    return blocks

def _build_word_from_order(order, n):
    """Build the n-word given an ordering of (n-1)-blocks."""
    word = []
    for p in order:
        block = list(p) + [n] + list(p)
        k = _overlap_len(word, block) if word else 0
        word.extend(block[k:])
    return word

def _vns_improve(order, n, time_limit):
    """VNS on block ordering to maximize sum of adjacent overlaps (minimize length)."""
    blocks = [list(p) + [n] + list(p) for p in order]
    m = len(blocks)

    start = time.time()
    # Precompute overlap matrix
    overlap = [[0]*m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            overlap[i][j] = _overlap_len(blocks[i], blocks[j])
    if time.time() - start > time_limit:
        return list(order)

    def cur_overlap_sum(perm):
        s = 0
        for i in range(len(perm)-1):
            s += overlap[perm[i]][perm[i+1]]
        return s

    perm = list(range(m))
    current = cur_overlap_sum(perm)
    best = current
    best_perm = list(perm)

    rng = random.Random(42)
    improved = True
    while improved:
        improved = False
        if time.time() - start > time_limit:
            break
        for _ in range(300):
            if time.time() - start > time_limit:
                break
            typ = rng.random()
            if typ < 0.5:
                i = rng.randrange(m)
                j = rng.randrange(i+1, m) if i+1 < m else i
                if i == j:
                    continue
                new_perm = perm[:i] + perm[i:j+1][::-1] + perm[j+1:]
                ns = 0
                for k in range(m-1):
                    ns += overlap[new_perm[k]][new_perm[k+1]]
                if ns > current:
                    current = ns
                    perm = new_perm
                    improved = True
                    if current > best:
                        best = current
                        best_perm = list(perm)
            else:
                seglen = rng.choice([1, 2, 3])
                if m - seglen <= 0:
                    continue
                i = rng.randrange(m - seglen)
                j = rng.randrange(m - seglen)
                if i == j:
                    continue
                seg = perm[i:i+seglen]
                rest = perm[:i] + perm[i+seglen:]
                ins = j if j <= i else j
                new_perm = rest[:ins] + seg + rest[ins:]
                ns = 0
                for k in range(m-1):
                    ns += overlap[new_perm[k]][new_perm[k+1]]
                if ns > current:
                    current = ns
                    perm = new_perm
                    improved = True
                    if current > best:
                        best = current
                        best_perm = list(perm)
    return [order[i] for i in best_perm]

def construct(n):
    total_deadline = 150.0
    start = time.time()

    if n == 1:
        return "1"
    if n == 2:
        return "121"

    word = _standard_word(n-1)
    blocks = _extract_blocks(word, n-1)

    budget = min(130.0, total_deadline - (time.time() - start) - 5.0)
    budget = max(budget, 0.5)

    best_order = _vns_improve(blocks, n, budget)

    final = _build_word_from_order(best_order, n)

    if time.time() - start > total_deadline - 2.0:
        final = _standard_word(n)

    return "".join(map(str, final))
# EVOLVE-BLOCK-END
