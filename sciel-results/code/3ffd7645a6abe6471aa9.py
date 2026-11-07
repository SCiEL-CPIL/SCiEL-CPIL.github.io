import time
import random

P = {'n': 128}


def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # safety margin before the 90s kill

    N = P['n']
    M = 2 * N  # maximum possible pair sum

    # ------------------------------------------------------------------
    # Build the conflict graph.
    #   nodes 0..N-1            -> member x  (bit x-1)
    #   nodes N..N+M-1          -> sum s    (bit s-1)
    #   edge (u,v) means u and v cannot both be in the set.
    #
    # A forbidden triple (a,b,c) exists iff a+b, a+c, b+c are all present.
    # So two sums are in conflict when they are both pair-sums of some
    # member x (i.e. s1 = x+y1, s2 = x+y2, y1!=y2).  We also forbid
    # a sum equal to a member (they are the same value).
    # Any independent set of this graph is exactly a valid set.
    # ------------------------------------------------------------------
    nodes = N + M
    adj = [0] * nodes
    for s in range(2, M + 1):
        for a in range(1, min(s, N) + 1):
            b = s - a
            if b <= a or b > N:
                continue
            sb = N + s - 1
            # sum s conflicts with member a and member b (a+b=s -> a or b absent)
            adj[sb] |= (1 << (a - 1)) | (1 << (b - 1))
            # sum s conflicts with every sum that is also a pair-sum of a
            # (a+b, a+b' with b!=b' both present -> triple with a)
            adj[sb] |= (1 << (N + a + b - 1)) | (1 << (N + a - 1))
            adj[sb] |= (1 << (N + b + a - 1)) | (1 << (N + b - 1))
            # and vice versa (symmetry handled by OR on both sides below)
    for s in range(2, M + 1):
        for a in range(1, min(s, N) + 1):
            b = s - a
            if b <= a or b > N:
                continue
            sb = N + s - 1
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                s2a = a + c
                s2b = b + c
                if s2a <= M:
                    adj[sb] |= (1 << (N + s2a - 1))
                if s2b <= M:
                    adj[sb] |= (1 << (N + s2b - 1))
                if c + a <= M:
                    adj[N + c + a - 1] |= (1 << sb)
                if c + b <= M:
                    adj[N + c + b - 1] |= (1 << sb)
    # make symmetric
    for u in range(nodes):
        for v in range(u):
            if (adj[u] >> v) & 1:
                adj[v] |= (1 << u)

    def members_of(mask):
        out = []
        m = mask
        while m:
            lsb = m & -m
            out.append(lsb.bit_length())  # bit i-1 -> value i
            m ^= lsb
        return out

    def valid_members(mask):
        # independent-set test restricted to member bits (sums never selected)
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            if (adj[i] & mask) & ~lsb:
                return False
            m ^= lsb
        return True

    # ------------------------------------------------------------------
    # Structural seed: {64..128} U {22..31}  (65 elements, known valid).
    # ------------------------------------------------------------------
    seed_mask = 0
    for v in range(64, N + 1):
        seed_mask |= (1 << (v - 1))
    for v in range(22, 32):
        seed_mask |= (1 << (v - 1))
    if not valid_members(seed_mask):
        seed_mask = 0
        for v in range(64, N + 1):
            seed_mask |= (1 << (v - 1))

    best_mask = seed_mask
    best_size = seed_mask.bit_count()

    # ------------------------------------------------------------------
    # Conflict-graph maximum independent set (branch & bound), member-only.
    # Returns the best independent set (as member bitmask) it finds within
    # the time budget.  This is the structural core of the search.
    # ------------------------------------------------------------------
    def mis_search(seed, time_budget):
        t0 = time.time()
        limit = t0 + time_budget
        # degree of each member node (within member subgraph)
        deg = [0] * N
        for i in range(N):
            deg[i] = (adj[i] & ((1 << N) - 1)).bit_count()
        # greedy upper bound on remaining candidate set (branching factor ~2)
        def ub(cand):
            c = cand
            cnt = 0
            while c:
                lsb = c & -c
                cnt += 1
                c &= c - 1
                c &= ~(adj[lsb.bit_length() - 1] | lsb)
            return cnt

        best = [seed, seed.bit_count()]

        def rec(cand, chosen):
            if time.time() > limit:
                return
            if cand == 0:
                if chosen.bit_count() > best[1]:
                    best[0], best[1] = chosen, chosen.bit_count()
                return
            if ub(cand) + chosen.bit_count() <= best[1]:
                return
            # pick highest-degree candidate for tighter branching
            lsb = cand & -cand
            best_deg = (adj[lsb.bit_length() - 1] & cand).bit_count()
            for v in range(N):
                if (cand >> v) & 1:
                    d = (adj[v] & cand).bit_count()
                    if d > best_deg:
                        best_deg = d
                        lsb = 1 << v
            # branch 1: exclude
            rec(cand & ~lsb, chosen)
            # branch 2: include
            rec(cand & ~(adj[lsb.bit_length() - 1] | lsb), chosen | lsb)

        # initial greedy independent set to seed the bound
        greedy = 0
        c = (1 << N) - 1
        while c:
            lsb = c & -c
            greedy |= lsb
            c &= ~(adj[lsb.bit_length() - 1] | lsb)
        best[0], best[1] = greedy, greedy.bit_count()
        rec((1 << N) - 1, 0)
        return best[0]

    # ------------------------------------------------------------------
    # Fast local search on top of the MIS result: greedy add + 1-out/2-in
    # swaps using the conflict graph (a move is legal iff it adds no edge).
    # ------------------------------------------------------------------
    def addable(mask, x):
        return (adj[x - 1] & mask) == 0

    def local_search(mask):
        t0 = time.time()
        limit = time.time() + 18.0
        while time.time() < limit:
            # greedy add
            changed = False
            for x in range(1, N + 1):
                if not (mask >> (x - 1)) & 1 and addable(mask, x):
                    mask |= (1 << (x - 1))
                    changed = True
            if changed:
                continue
            # 1-out, 2-in swap
            elems = members_of(mask)
            random.shuffle(elems)
            found = False
            for xout in elems[:6]:
                base = mask & ~(1 << (xout - 1))
                adds = [x for x in range(1, N + 1)
                        if not (base >> (x - 1)) & 1 and addable(base, x)]
                random.shuffle(adds)
                for x1 in adds[:12]:
                    m1 = base | (1 << (x1 - 1))
                    for x2 in adds:
                        if x2 == x1:
                            continue
                        if addable(m1, x2):
                            mask = m1 | (1 << (x2 - 1))
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                continue
            # perturbation: drop 1-2 random elements
            if not elems:
                break
            k = 1 if random.random() < 0.7 else 2
            for _ in range(min(k, len(elems))):
                xout = random.choice(elems)
                mask &= ~(1 << (xout - 1))
                elems = members_of(mask)
            if not elems:
                break
        return mask

    # ------------------------------------------------------------------
    # Driver: structural MIS search, then local search, repeat with
    # perturbation to escape local optima.
    # ------------------------------------------------------------------
    if time.time() < deadline - 25.0:
        m = mis_search(seed_mask, deadline - time.time() - 25.0)
        m = local_search(m)
        if m.bit_count() > best_size:
            best_mask, best_size = m, m.bit_count()

    # additional local-search restarts from the best found so far
    while time.time() < deadline - 6.0:
        start = best_mask
        # random perturbation of the best
        elems = members_of(start)
        random.shuffle(elems)
        k = random.choice([1, 2, 3])
        for x in elems[:k]:
            start &= ~(1 << (x - 1))
        m = local_search(start)
        if m.bit_count() > best_size:
            best_mask, best_size = m, m.bit_count()

    # final polish: exhaustive greedy add pass
    if time.time() < deadline - 1.0:
        fm = best_mask
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            for x in range(1, N + 1):
                if not (fm >> (x - 1)) & 1 and addable(fm, x):
                    fm |= (1 << (x - 1))
                    changed = True
        if fm.bit_count() > best_size:
            best_mask, best_size = fm, fm.bit_count()

    result = members_of(best_mask)
    return sorted(result)
