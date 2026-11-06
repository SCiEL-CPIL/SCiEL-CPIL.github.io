"""Rewritten: branch-and-bound DFS with bitset conflict tracking."""
import time
import random

# EVOLVE-BLOCK-START

def _bit_check(sums, new_values, current_mask):
    """Check whether new_values conflict with existing sums (bitmask).
    Returns (new_mask, ok)."""
    m = current_mask
    for v in new_values:
        if v < 0 or v >= 20000:
            return current_mask, False
        bit = 1 << v
        if m & bit:
            return current_mask, False
        m |= bit
    return m, True

def _extend_prefix(seq, sums_mask, sums_set):
    """Forward-greedy continuation: pick the smallest unused value >= seq[-1]+1
    that doesn't conflict. Returns (extended_seq, extended_mask, extended_set)."""
    out = list(seq)
    mask = sums_mask
    sset = set(sums_set)
    n = len(out)
    prev = out[-1] if n else 0
    for x in range(prev + 1, 257):
        # compute new suffix sums ending at x
        new_vals = [x]
        total = x
        for i in range(n - 1, -1, -1):
            total += out[i]
            new_vals.append(total)
        m, ok = _bit_check(sset, new_vals, mask)
        if ok:
            out.append(x)
            mask = m
            sset.update(new_vals)
    return out, mask, sset

def _upper_bound(seq, deadline):
    """Crude upper bound on how many more elements we can append.
    Each additional element >= last+1, and we must stay <= 256."""
    if not seq:
        return 256
    last = seq[-1]
    return 256 - last

def construct():
    start = time.time()
    deadline = start + 80.0  # 10s safety margin before 90s kill

    best = {'seq': [], 'mask': 0, 'length': 0}

    def dfs(seq, mask, set_, depth, deadline):
        nonlocal best
        if len(seq) > best['length']:
            best['seq'] = list(seq)
            best['mask'] = mask
            best['length'] = len(seq)

        if time.time() > deadline:
            return

        if not seq:
            low = 1
        else:
            low = seq[-1] + 1
        if low > 256:
            return

        # upper bound prune: even if we could pick one per integer up to 256
        max_extra = 256 - (low - 1)
        if len(seq) + max_extra <= best['length']:
            return

        # Try a handful of candidate x values in [low, 256].
        # We bias toward smaller x (more headroom) but also try larger to explore.
        n = len(seq)
        # Build candidate set: always include low, low+1, a few randoms, and high
        cands = [low, low + 1]
        # A couple of mid/randoms
        if 256 - low > 5:
            for _ in range(4):
                cands.append(random.randint(low, 256))
        # De-dup
        cands = list(dict.fromkeys(cands))
        cands.sort()

        for x in cands:
            if time.time() > deadline:
                return
            new_vals = [x]
            total = x
            for i in range(n - 1, -1, -1):
                total += seq[i]
                new_vals.append(total)
            m, ok = _bit_check(set_, new_vals, mask)
            if not ok:
                continue
            new_seq = seq + [x]
            new_set = set(set_)
            new_set.update(new_vals)
            dfs(new_seq, m, new_set, depth + 1, deadline)

    # Phase 1: seed with a quick greedy to have a non-trivial best.
    seed_seq, seed_mask, seed_set = _extend_prefix([], 0, set())
    best['seq'] = list(seed_seq)
    best['mask'] = seed_mask
    best['length'] = len(seed_seq)

    # Phase 2: DFS from the empty prefix to explore alternative prefixes.
    dfs([], 0, set(), 0, deadline)

    # Phase 3: if we have time, try extending the best sequence greedily
    # with a different continuation strategy (skip-and-take) to possibly add more.
    if time.time() < deadline - 5.0:
        ext_seq, ext_mask, ext_set = _extend_prefix(best['seq'], best['mask'], set(best['seq'] and best['mask'] or set()))
        # Rebuild set from mask is expensive; instead extend from the greedy continuation
        # using set tracking. We don't have the original set, so we rebuild it.
        # Simpler: just greedily extend the best sequence by recomputing sums.
        cur = list(best['seq'])
        sset = set()
        nn = len(cur)
        for i in range(nn):
            s = 0
            for j in range(i, nn):
                s += cur[j]
                sset.add(s)
        for x in range(cur[-1] + 1 if cur else 1, 257):
            new_vals = [x]
            total = x
            for i in range(len(cur) - 1, -1, -1):
                total += cur[i]
                new_vals.append(total)
            if any(v in sset for v in new_vals):
                continue
            cur.append(x)
            sset.update(new_vals)
        if len(cur) > best['length']:
            best['seq'] = cur
            best['length'] = len(cur)

    return best['seq']

# EVOLVE-BLOCK-END
