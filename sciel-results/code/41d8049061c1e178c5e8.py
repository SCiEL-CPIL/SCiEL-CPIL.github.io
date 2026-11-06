import time
import random

def _has_k222(adj):
    n = len(adj)
    # adj[i] is a bitmask of neighbors
    # K2,2,2: parts A={u,v}, B={w,x}, C={y,z}
    # u,v non-adjacent; w,x non-adjacent; y,z non-adjacent
    # all cross edges present
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue  # u,v adjacent, cannot be same part
            common = adj[u] & adj[v]
            verts = []
            mask = common
            while mask:
                lsb = mask & -mask
                verts.append(lsb.bit_length() - 1)
                mask ^= lsb
            m = len(verts)
            for i in range(m):
                w = verts[i]
                for j in range(i + 1, m):
                    x = verts[j]
                    if adj[w] & (1 << x):
                        continue  # w,x adjacent, cannot be same part
                    cand = common & adj[w] & adj[x]
                    cverts = []
                    cmask = cand
                    while cmask:
                        lsb = cmask & -cmask
                        cverts.append(lsb.bit_length() - 1)
                        cmask ^= lsb
                    c = len(cverts)
                    for k in range(c):
                        y = cverts[k]
                        for l in range(k + 1, c):
                            z = cverts[l]
                            if not (adj[y] & (1 << z)):
                                return True
    return False


def _build_adj(colors, n):
    adj = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
            idx += 1
    return adj


def _count_violations(adj):
    # Count number of K2,2,2 copies (for a smoother objective).
    # Same enumeration as _has_k222 but counts instead of early-returning.
    n = len(adj)
    total = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u] & (1 << v):
                continue
            common = adj[u] & adj[v]
            verts = []
            mask = common
            while mask:
                lsb = mask & -mask
                verts.append(lsb.bit_length() - 1)
                mask ^= lsb
            m = len(verts)
            for i in range(m):
                w = verts[i]
                for j in range(i + 1, m):
                    x = verts[j]
                    if adj[w] & (1 << x):
                        continue
                    cand = common & adj[w] & adj[x]
                    cverts = []
                    cmask = cand
                    while cmask:
                        lsb = cmask & -cmask
                        cverts.append(lsb.bit_length() - 1)
                        cmask ^= lsb
                    c = len(cverts)
                    for k in range(c):
                        y = cverts[k]
                        for l in range(k + 1, c):
                            z = cverts[l]
                            if not (adj[y] & (1 << z)):
                                total += 1
    return total


def _paley(n, p):
    if p < n:
        return None
    squares = {x * x % p for x in range(1, p)}
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors


def _is_valid(colors, n):
    if len(colors) != n * (n - 1) // 2:
        return False
    adj1 = _build_adj(colors, n)
    if _has_k222(adj1):
        return False
    # color 0 = complement
    adj0 = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and not (adj1[i] & (1 << j)):
                adj0[i] |= (1 << j)
    if _has_k222(adj0):
        return False
    return True


def _local_search(colors, n, time_limit, seed):
    rng = random.Random(seed)
    start = time.time()
    m = n * (n - 1) // 2
    adj1 = _build_adj(colors, n)
    adj0 = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and not (adj1[i] & (1 << j)):
                adj0[i] |= (1 << j)

    def flip(idx):
        i = idx // (n - 1)
        rem = idx % (n - 1)
        j = i + 1 + rem
        if colors[idx] == 1:
            adj1[i] &= ~(1 << j)
            adj1[j] &= ~(1 << i)
            adj0[i] |= (1 << j)
            adj0[j] |= (1 << i)
        else:
            adj0[i] &= ~(1 << j)
            adj0[j] &= ~(1 << i)
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
        colors[idx] = 1 - colors[idx]

    while time.time() - start < time_limit:
        if not (_has_k222(adj1) or _has_k222(adj0)):
            return True
        # Choose a color class that has a violation, then pick a random edge in it
        # to flip, hoping to break a K2,2,2.
        v1 = _has_k222(adj1)
        v0 = _has_k222(adj0)
        if v1 and v0:
            idx = rng.randrange(m)
        elif v1:
            # random edge that is color 1
            idx = rng.randrange(m)
        elif v0:
            idx = rng.randrange(m)
        else:
            return True
        flip(idx)
    return not (_has_k222(adj1) or _has_k222(adj0))


def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s kill

    best_n = 0
    best_colors = None

    # 1) Try direct Paley constructions for many primes, largest n first.
    primes = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]
    for p in primes:
        if time.time() > deadline:
            break
        for n in range(min(p, 31), 16, -1):
            if time.time() > deadline:
                break
            colors = _paley(n, p)
            if colors is None:
                continue
            if _is_valid(colors, n):
                if n > best_n:
                    best_n = n
                    best_colors = colors[:]
                break  # this p valid at n; try smaller n with same p only if needed

    if best_n >= 30:
        return {'n': best_n, 'colors': best_colors}

    # 2) Multi-restart local search starting from Paley initializations.
    for p in primes:
        if time.time() > deadline:
            break
        for n in range(min(p, 31), best_n, -1):
            if time.time() > deadline:
                break
            base = _paley(n, p)
            if base is None:
                continue
            # A few restarts with random perturbations of the Paley structure
            for attempt in range(8):
                if time.time() > deadline:
                    break
                colors = base[:]
                # Perturb a fraction of edges
                m = n * (n - 1) // 2
                for _ in range(m // 3):
                    idx = random.randrange(m)
                    colors[idx] = 1 - colors[idx]
                tl = min(15, max(1, int(deadline - time.time()) - 10))
                if _local_search(colors, n, tl, seed=attempt * 1000 + p):
                    if n > best_n:
                        best_n = n
                        best_colors = colors[:]
                        if n >= 30:
                            return {'n': best_n, 'colors': best_colors}
                    break  # found valid for this n; move to larger n

    # 3) Pure random restarts for the largest unattained n.
    for n in range(min(best_n + 1, 31), 17, -1):
        if time.time() > deadline:
            break
        for attempt in range(30):
            if time.time() > deadline:
                break
            colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            tl = min(12, max(1, int(deadline - time.time()) - 10))
            if _local_search(colors, n, tl, seed=attempt * 7 + n):
                if n > best_n:
                    best_n = n
                    best_colors = colors[:]
                    if n >= 30:
                        return {'n': best_n, 'colors': best_colors}
                break

    # Fallback to a guaranteed-valid small coloring.
    if best_colors is None:
        best_n = 17
        best_colors = _paley(17, 17)

    return {'n': best_n, 'colors': best_colors}
