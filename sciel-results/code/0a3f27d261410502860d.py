import time
from math import lcm
import random

def construct():
    t0 = time.time()
    TIME_LIMIT = 54.0
    vals = list(range(1, 129))
    n = 128
    full = (1 << n) - 1

    # Precompute pairwise LCM matrix.
    L = [[0] * n for _ in range(n)]
    for i in range(n):
        vi = vals[i]
        row = L[i]
        for j in range(i + 1, n):
            v = lcm(vi, vals[j])
            row[j] = v
            L[j][i] = v

    # comp[i][j] = bitset of k such that {i,j,k} is a bad triple
    comp = [[0] * n for _ in range(n)]
    for i in range(n):
        Li = L[i]
        ci = comp[i]
        for j in range(i + 1, n):
            lv = Li[j]
            Lj = L[j]
            mask = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if Li[k] == lv and Lj[k] == lv:
                    mask |= 1 << k
            ci[j] = mask
            comp[j][i] = mask

    # Precompute degrees for branching heuristic
    deg = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            mask = comp[i][j]
            m = mask
            while m:
                k = (m & -m).bit_length() - 1
                m ^= 1 << k
                deg[i] += 1
                deg[j] += 1
                deg[k] += 1

    # Greedy (order-driven) independent-set builder for initial lower bound.
    def greedy(order):
        S = []
        F = 0
        for v in order:
            if (1 << v) & F:
                continue
            S.append(v)
            nf = F
            for a in S[:-1]:
                nf |= comp[v][a]
            F = nf
        return S

    rng = random.Random(12345)
    base = list(range(n))
    best_len = 0
    best_set = []
    for _ in range(200):
        order = base[:]
        rng.shuffle(order)
        S = greedy(order)
        if len(S) > best_len:
            best_len = len(S)
            best_set = S[:]
        if time.time() - t0 > TIME_LIMIT * 0.1:
            break
    if not best_set:
        best_set = greedy(base)
        best_len = len(best_set)

    best = [best_len]
    best_sol = best_set[:]
    nodes = [0]

    def bb(S, F, R):
        nodes[0] += 1
        if time.time() - t0 > TIME_LIMIT:
            return
        avail = R & ~F
        if avail == 0:
            if len(S) > best[0]:
                best[0] = len(S)
                best_sol[:] = S[:]
            return
        
        # Trivial bound
        if len(S) + avail.bit_count() <= best[0]:
            return

        # Branching variable selection
        bestv = -1
        best_gain = -1
        
        if S:
            av = avail
            while av:
                v = (av & -av).bit_length() - 1
                av ^= 1 << v
                nf = F
                for a in S:
                    nf |= comp[v][a]
                gain = (avail & nf).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    bestv = v
        else:
            av = avail
            bestdeg = -1
            while av:
                v = (av & -av).bit_length() - 1
                av ^= 1 << v
                if deg[v] > bestdeg:
                    bestdeg = deg[v]
                    bestv = v
            best_gain = 0

        v = bestv
        bit_v = 1 << v
        
        # Include branch
        if S:
            nf = F
            for a in S:
                nf |= comp[v][a]
        else:
            nf = F
            
        S.append(v)
        bb(S, nf, R & ~bit_v)
        S.pop()
        
        # Exclude branch
        bb(S, F, R & ~bit_v)

    bb([], 0, full)

    res = sorted(vals[i] for i in best_sol)
    if not res:
        res = [1]
    return res
