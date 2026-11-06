def construct():
    import time, sys
    from math import gcd
    deadline = time.time() + 80  # safety margin before 90s kill
    N = 255
    vals = list(range(2, 257))
    weights = [1.0 / vals[i] for i in range(N)]
    full = (1 << N) - 1

    # Precompute incompatibility bitsets: incomp[i] has bit j iff lcm(vals[i],vals[j]) <= 256
    incomp = [0] * N
    for i in range(N):
        vi = vals[i]
        for j in range(i + 1, N):
            vj = vals[j]
            g = gcd(vi, vj)
            if vi * vj // g <= 256:
                incomp[i] |= (1 << j)
                incomp[j] |= (1 << i)
    compat = [(full & ~(incomp[i] | (1 << i))) for i in range(N)]

    order = sorted(range(N), key=lambda i: -weights[i])  # largest weight first

    def complete(cand):
        """Greedily extend: add every vertex in cand (all compatible with current sel), in high-weight order."""
        added = []
        c = cand
        while c:
            lb = c & (-c)
            i = lb.bit_length() - 1
            added.append(i)
            c &= compat[i]
        return added

    def greedy():
        cand = full
        sel = []
        for i in order:
            if cand & (1 << i):
                sel.append(i)
                cand &= compat[i]
        sel = sel + complete(cand)  # completion pass the seed lacks
        s = sum(weights[i] for i in sel)
        return s, sel

    best = [0.0]
    best_set = [[]]
    gs, gsel = greedy()
    best[0] = gs
    best_set[0] = gsel

    sys.setrecursionlimit(1000000)

    def bit_sum(c):
        s = 0.0
        while c:
            lb = c & (-c)
            s += weights[lb.bit_length() - 1]
            c ^= lb
        return s

    def is_clique(c):
        cc = c
        while cc:
            lb = cc & (-cc)
            i = lb.bit_length() - 1
            if (c & ~(1 << i)) & ~compat[i]:
                return False
            cc ^= lb
        return True

    nodes = [0]

    def solve(cand, cur, sel):
        nodes[0] += 1
        if time.time() > deadline:
            return
        if cand == 0:
            if cur > best[0]:
                best[0] = cur
                best_set[0] = sel[:]
            return
        ub = cur + bit_sum(cand)
        if ub <= best[0] + 1e-12:
            return
        if is_clique(cand):
            v = cur + bit_sum(cand)
            if v > best[0]:
                best[0] = v
                best_set[0] = sel[:] + [i for i in range(N) if cand & (1 << i)]
            return
        # branch on highest-weight candidate
        c = cand
        bi = -1
        bw = -1.0
        while c:
            lb = c & (-c)
            i = lb.bit_length() - 1
            if weights[i] > bw:
                bw = weights[i]
                bi = i
            c ^= lb
        solve(cand & compat[bi], cur + weights[bi], sel + [bi])   # include
        solve(cand & ~(1 << bi), cur, sel)                          # exclude

    solve(full, 0.0, [])

    if not best_set[0]:
        return [2]
    return [vals[i] for i in best_set[0]]
