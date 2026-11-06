"""Rewritten program: chain-partitioned antichain search with bitmask DP."""
# EVOLVE-BLOCK-START
import time

def _is_smooth(x):
    y = x
    while y % 2 == 0:
        y //= 2
    while y % 3 == 0:
        y //= 3
    while y % 5 == 0:
        y //= 5
    return y == 1

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 90s kill

    # ---- Enumerate smooth numbers <= 128, with (a,b,c) exponents ----
    smooth = []          # values
    exps = []            # (a,b,c)
    for a in range(0, 8):
        p2 = 1 << a
        if p2 > 128:
            break
        for b in range(0, 6):
            p3 = 3 ** b
            if p2 * p3 > 128:
                break
            for c in range(0, 5):
                p5 = 5 ** c
                v = p2 * p3 * p5
                if v > 128:
                    break
                smooth.append(v)
                exps.append((a, b, c))
    smooth.sort()
    n = len(smooth)

    # value -> index
    v2i = {v: i for i, v in enumerate(smooth)}

    # ---- Precompute divisibility conflict (i divides j or j divides i, i!=j) ----
    conflict = [0] * n  # bitmask of conflicting indices
    for i in range(n):
        vi = smooth[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            vj = smooth[j]
            if vj % vi == 0 or vi % vj == 0:
                mask |= (1 << j)
        conflict[i] = mask

    # ---- Chain decomposition: group by (a,b); within a chain pick at most one ----
    chain_of = [0] * n
    chains = []  # list of lists of indices
    for i in range(n):
        a, b, c = exps[i]
        key = (a, b)
        # find chain
        cid = None
        for ci, ch in enumerate(chains):
            if ch and exps[ch[0]][:2] == key:
                cid = ci
                break
        if cid is None:
            chains.append([])
            cid = len(chains) - 1
        chains[cid].append(i)
        chain_of[i] = cid
    nchains = len(chains)

    # Sort each chain by value (ascending)
    for ch in chains:
        ch.sort(key=lambda i: smooth[i])

    # For each chain, precompute: list of (index, value), and per-position max value
    chain_vals = [[smooth[i] for i in ch] for ch in chains]
    # suffix max of values in each chain from position p
    chain_sufmax = []
    for cv in chain_vals:
        suf = [0] * (len(cv) + 1)
        for p in range(len(cv) - 1, -1, -1):
            suf[p] = max(suf[p + 1], cv[p])
        chain_sufmax.append(suf)

    # ---- Memoized search: min elements to reach sum `remaining` using `k` more picks,
    #      given current conflict mask. Returns min count or None if infeasible. ----
    # We search for EXACTLY k elements summing to remaining.
    memo = {}

    def search(remaining, k, cmask, chain_id, pos):
        # chain_id: next chain to consider (chains are independent for the "pick at most one" local rule)
        if k == 0:
            return 0 if remaining == 0 else None
        if remaining < 0:
            return None
        if chain_id >= nchains:
            return None
        if time.time() > deadline:
            return None

        key = (remaining, k, cmask, chain_id)
        if key in memo:
            return memo[key]

        best = None
        ch = chains[chain_id]
        cv = chain_vals[chain_id]
        L = len(cv)

        # Option 1: skip this chain entirely
        r = search(remaining, k, cmask, chain_id + 1, 0)
        if r is not None:
            best = r

        # Option 2: pick exactly one element from this chain
        # We can only pick at most one (local antichain rule within chain).
        # Iterate positions; for each element, check global conflict.
        for p in range(L):
            idx = ch[p]
            val = cv[p]
            if val > remaining:
                # since cv sorted ascending, larger p have larger value -> break
                break
            # global conflict check
            if cmask & conflict[idx]:
                continue
            new_cmask = cmask | conflict[idx]
            r2 = search(remaining - val, k - 1, new_cmask, chain_id + 1, 0)
            if r2 is not None:
                cand = 1 + r2
                if best is None or cand < best:
                    best = cand

        memo[key] = best
        return best

    # ---- Reconstruct the actual set of values for a target with given k ----
    def reconstruct(target, k):
        # returns list of values or None
        path = []
        cmask = 0
        remaining = target
        kk = k
        for chain_id in range(nchains):
            if time.time() > deadline:
                return None
            ch = chains[chain_id]
            cv = chain_vals[chain_id]
            L = len(cv)
            # decide: skip or pick one
            # first try pick (we want to find a valid one; prefer to find any)
            picked = False
            for p in range(L):
                idx = ch[p]
                val = cv[p]
                if val > remaining:
                    break
                if cmask & conflict[idx]:
                    continue
                new_cmask = cmask | conflict[idx]
                # check feasibility of the rest
                r = search(remaining - val, kk - 1, new_cmask, chain_id + 1, 0)
                if r is not None:
                    path.append(val)
                    cmask = new_cmask
                    remaining -= val
                    kk -= 1
                    picked = True
                    break
            if not picked:
                # skip chain
                continue
            if kk == 0:
                break
        if remaining == 0 and kk == 0:
            return path
        return None

    # ---- Greedy fallback: find ANY antichain summing to target (non-minimal) ----
    def greedy_any(target):
        # simple backtracking over indices, antichain check via conflict
        chosen = []
        cmask = 0
        def bt(rem, start):
            if time.time() > deadline:
                return None
            if rem == 0:
                return list(chosen)
            if start >= n:
                return None
            for i in range(start, n):
                v = smooth[i]
                if v > rem:
                    continue
                if cmask & conflict[i]:
                    continue
                chosen.append(v)
                new_cmask = cmask | conflict[i]
                r = bt(rem - v, i + 1)
                if r is not None:
                    return r
                chosen.pop()
            return None
        return bt(target, 0)

    answers = [None] * 129
    total = 0

    for t in range(1, 129):
        if time.time() > deadline:
            break
        # clear memo per target to keep it small (state space depends on t)
        memo.clear()
        found = None
        for k in range(1, 31):
            if time.time() > deadline:
                break
            r = search(t, k, 0, 0, 0)
            if r is not None:
                vals = reconstruct(t, k)
                if vals is not None:
                    found = vals
                    break
        if found is None:
            # fallback
            if t in v2i:
                found = [t]
            else:
                found = greedy_any(t)
                if found is None:
                    found = []  # should not happen
        answers[t] = found
        total += len(found)

    # fill any gaps
    for t in range(1, 129):
        if answers[t] is None:
            if t in v2i:
                answers[t] = [t]
            else:
                g = greedy_any(t)
                answers[t] = g if g is not None else []

    return [answers[t] for t in range(1, 129)]

# EVOLVE-BLOCK-END
