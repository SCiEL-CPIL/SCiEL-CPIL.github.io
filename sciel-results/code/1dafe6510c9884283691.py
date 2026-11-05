"""Column-generation set covering for C(19,7,4)."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

# Precompute all 4-subsets of 0..18
_V4 = 7752  # C(19,4)


def _all_t_subsets(v, t):
    return list(itertools.combinations(range(v), t))


def construct(v, k, t):
    deadline = time.time() + 110.0  # safety margin vs 120s/150s kill
    all_t = _all_t_subsets(v, t)
    n_t = len(all_t)
    # map tuple -> index
    t_index = {s: i for i, s in enumerate(all_t)}

    # uncovered: bitset of which t-subsets are still uncovered
    uncovered = np.ones(n_t, dtype=np.uint8)
    blocks = []
    block_t_mask = []  # for each block, indices of t-subsets it covers
    # dual weights: 1 for uncovered, 0 for covered (uniform initially)
    dual = np.ones(n_t, dtype=np.float64)

    # Precompute for each point x: list of t-subset indices containing x
    # t-subsets are 4-sets; a point x is in those 4-sets that include x.
    # We need, for a candidate block B (7 points), the set of 4-subsets contained in B.
    # A 4-subset S is contained in B iff S is a subset of B.
    # Number of 4-subsets of a 7-set is C(7,4)=35.

    # Fast containment: for a 7-tuple B, enumerate its 35 4-subsets and look up index.
    # We'll cache block->mask.

    block_cache = {}

    def block_mask(B):
        B = tuple(sorted(B))
        if B in block_cache:
            return block_cache[B]
        idxs = []
        for s in itertools.combinations(B, t):
            idxs.append(t_index[s])
        m = np.zeros(n_t, dtype=np.uint8)
        m[idxs] = 1
        block_cache[B] = m
        return m

    def pricing():
        """Find the 7-block maximizing sum of dual over its 4-subsets.
        Use local search / hill climbing with random restarts.
        Objective: for block B, score = sum_{S in C(B,t)} dual[S].
        """
        best_score = -1.0
        best_block = None
        # Precompute dual as numpy for speed.
        # We'll do random restarts of greedy + local search.
        # A 7-block: we can build by starting from a random 7-set, then swap-improve.
        # Score of a block = sum of dual over its 35 4-subsets.
        # We can compute score incrementally via swaps.

        # Precompute for each 7-tuple its 4-subset index list (cached).
        # For swap: remove point a, add point b. New 4-subsets are those containing b
        # but not a, from the new 7-set.

        # Simpler: recompute score from scratch is 35 lookups; local search is cheap.
        # Do multiple random restarts.

        rng = np.random.default_rng(42)
        n_restart = 40
        for r in range(n_restart):
            if time.time() > deadline - 20:
                break
            # random start
            B = set(rng.choice(v, size=k, replace=False).tolist())
            cur_score = _score_block(B, dual, t_index, all_t)
            improved = True
            it = 0
            while improved and it < 200:
                improved = False
                it += 1
                for a in list(B):
                    B.remove(a)
                    for b in range(v):
                        if b not in B:
                            B.add(b)
                            new_score = _score_block(B, dual, t_index, all_t)
                            if new_score > cur_score + 1e-9:
                                cur_score = new_score
                                improved = True
                            else:
                                B.remove(b)
                    B.add(a)
            if cur_score > best_score:
                best_score = cur_score
                best_block = tuple(sorted(B))
        return best_block, best_score

    def _score_block(B, dual, t_index, all_t):
        s = 0.0
        for sub in itertools.combinations(B, t):
            s += dual[t_index[sub]]
        return s

    # Initial: a few greedy blocks to get going, then column generation.
    # Actually start with pricing directly.

    max_blocks = 400
    while True:
        if time.time() > deadline - 30:
            break
        if len(blocks) >= max_blocks:
            break
        B, sc = pricing()
        if B is None or sc <= 1e-6:
            break
        m = block_mask(B)
        # only add if it covers something new
        if not np.any(m & uncovered):
            break
        blocks.append(list(B))
        block_t_mask.append(m)
        # update uncovered and dual
        new_uncovered = uncovered & (1 - m)
        # if a t-subset becomes covered, set its dual to 0
        just_covered = (uncovered & m)
        if just_covered.any():
            dual[just_covered.astype(bool)] = 0.0
        uncovered = new_uncovered
        # early exit if all covered
        if not uncovered.any():
            break
        # adaptive: if dual is sparse, re-weight to focus on hard-to-cover
        # (columns with high dual are prioritized) - already done.

    # Fallback: if not fully covered, finish with greedy on remaining.
    if uncovered.any() and time.time() < deadline - 5:
        # greedy finish
        unc_set = set(i for i in range(n_t) if uncovered[i])
        while unc_set and time.time() < deadline - 2:
            # pick t-subset with min... just take first, build block greedily
            target = all_t[min(unc_set)]
            B = list(target)
            while len(B) < k and time.time() < deadline - 2:
                def gain(x):
                    c = 0
                    for rest in itertools.combinations(B, t - 1):
                        s = tuple(sorted(rest + (x,)))
                        if s in t_index and t_index[s] in unc_set:
                            c += 1
                    return c
                B.append(max((x for x in range(v) if x not in B), key=gain))
            B.sort()
            blocks.append(B)
            for s in itertools.combinations(B, t):
                unc_set.discard(t_index[s])

    # Deduplicate blocks (shouldn't happen but safe)
    seen = set()
    out = []
    for b in blocks:
        key = tuple(sorted(b))
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    return out
# EVOLVE-BLOCK-END
