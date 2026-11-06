import time, random, math

def construct():
    start = time.time()
    DEADLINE = start + 145.0
    n = 18
    rng = random.Random(12345)

    def edges_from_adj(adj):
        out = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    out.append([i, j])
        return out

    def has_k4(adj):
        for a in range(n):
            for b in range(a + 1, n):
                if not ((adj[a] >> b) & 1):
                    continue
                common = adj[a] & adj[b]
                for c in range(b + 1, n):
                    if not ((common >> c) & 1):
                        continue
                    dc = common & adj[c]
                    dc &= ~((1 << (c + 1)) - 1)
                    if dc:
                        return True
        return False

    def max_tf(adj, budget):
        neigh = adj[:]
        best = 0
        state = {'t0': time.time(), 'timeout': False}
        order = list(range(n))

        def indep_size(mask):
            s = 0
            m = mask
            while m:
                bit = m & (-m)
                v = bit.bit_length() - 1
                s += 1
                m &= ~neigh[v]
                m &= ~bit
            return s

        def rec(idx, cur_mask, cur_size, rem_mask):
            if state['timeout']:
                return
            if time.time() - state['t0'] > budget:
                state['timeout'] = True
                return
            if cur_size + indep_size(rem_mask) <= best:
                return
            if idx == n:
                nonlocal best
                if cur_size > best:
                    best = cur_size
                return
            v = order[idx]
            nrem = rem_mask & ~(1 << v)
            # exclude
            rec(idx + 1, cur_mask, cur_size, nrem)
            if state['timeout']:
                return
            # include: check triangle with already-selected neighbours
            nsel = cur_mask & neigh[v]
            bad = False
            t = nsel
            while t:
                bit = t & (-t)
                u = bit.bit_length() - 1
                rest = t & ~bit
                if (neigh[u] & rest):
                    bad = True
                    break
                t = rest
            if not bad:
                rec(idx + 1, cur_mask | (1 << v), cur_size + 1, nrem)

        for attempt in range(40):
            rng.shuffle(order)
            best = 0
            state['t0'] = time.time()
            state['timeout'] = False
            rec(0, 0, 0, (1 << n) - 1)
            if state['timeout']:
                break
            # converged: same result across shuffles -> trust it
            if attempt >= 2:
                break
        return best

    def remove_k4(adj):
        for _ in range(60):
            found = False
            for a in range(n):
                if found:
                    break
                for b in range(a + 1, n):
                    if found:
                        break
                    if not ((adj[a] >> b) & 1):
                        continue
                    common = adj[a] & adj[b]
                    for c in range(b + 1, n):
                        if found:
                            break
                        if not ((common >> c) & 1):
                            continue
                        dc = common & adj[c]
                        dc &= ~((1 << (c + 1)) - 1)
                        if dc:
                            d = (dc & (-dc)).bit_length() - 1
                            x, y = rng.choice([(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)])
                            adj[x] &= ~(1 << y); adj[y] &= ~(1 << x)
                            found = True
            if not found:
                break
        return not has_k4(adj)

    def turan3():
        adj = [0]*n
        parts = [list(range(0,6)), list(range(6,12)), list(range(12,18))]
        for i in range(3):
            for j in range(i+1,3):
                for a in parts[i]:
                    for b in parts[j]:
                        adj[a] |= (1<<b); adj[b] |= (1<<a)
        return adj

    def rand3partite():
        verts = list(range(n)); rng.shuffle(verts)
        parts = [verts[0:6], verts[6:12], verts[12:18]]
        adj = [0]*n
        cd = 0.55 + rng.random()*0.4
        for i in range(3):
            for j in range(i+1,3):
                for a in parts[i]:
                    for b in parts[j]:
                        if rng.random() < cd:
                            adj[a] |= (1<<b); adj[b] |= (1<<a)
        wd = rng.random()*0.35
        for i in range(3):
            for a in parts[i]:
                for b in parts[i]:
                    if a < b and rng.random() < wd:
                        adj[a] |= (1<<b); adj[b] |= (1<<a)
        return adj

    def rand_dense():
        adj = [0]*n
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < 0.55:
                    adj[i] |= (1<<j); adj[j] |= (1<<i)
        return adj

    cands = [turan3()]
    for _ in range(25):
        if time.time() > DEADLINE - 45:
            break
        a = rand3partite()
        if remove_k4(a):
            cands.append(a)
    for _ in range(20):
        if time.time() > DEADLINE - 45:
            break
        a = rand_dense()
        if remove_k4(a):
            cands.append(a)

    best_adj = None
    best_score = 99
    for a in cands:
        if time.time() > DEADLINE - 25:
            break
        s = max_tf(a, 2.5)
        if s < best_score:
            best_score = s
            best_adj = a[:]

    if best_adj is None:
        best_adj = cands[0]; best_score = max_tf(best_adj, 2.5)

    # Local search: bulk cheap K4-feasible flips, score only on promising moves
    adj = best_adj[:]
    cur = best_score
    best_adj_sa = adj[:]
    best_score_sa = cur
    temp = 1.0
    cool = 0.9975

    for it in range(20000):
        if time.time() > DEADLINE - 8:
            break
        if best_score_sa <= 6:
            break
        i = rng.randrange(n); j = rng.randrange(n)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        old = adj[:]
        if (adj[i] >> j) & 1:
            adj[i] &= ~(1<<j); adj[j] &= ~(1<<i)
        else:
            adj[i] |= (1<<j); adj[j] |= (1<<i)
        if has_k4(adj):
            adj = old
            continue
        s = max_tf(adj, 1.2)
        delta = s - cur
        accept = (delta <= 0) or (rng.random() < math.exp(-delta/max(temp,0.01)))
        if accept:
            cur = s
            if s < best_score_sa:
                best_score_sa = s
                best_adj_sa = adj[:]
        else:
            adj = old
        temp *= cool

    return edges_from_adj(best_adj_sa)
