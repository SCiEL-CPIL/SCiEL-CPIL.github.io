import time
import random
from math import gcd

MAXM = 128

def construct():
    t0 = time.time()
    DEADLINE = 80.0

    candidates = [m for m in range(2, MAXM + 1) if m % 2 == 0]
    n = len(candidates)
    moduli = candidates
    residues = [m // 2 for m in moduli]

    G = [[0] * n for _ in range(n)]
    for i in range(n):
        mi = moduli[i]
        for j in range(i + 1, n):
            g = gcd(mi, moduli[j])
            G[i][j] = G[j][i] = g

    def conflict_count():
        c = 0
        for i in range(n):
            ri = residues[i]
            for j in range(i + 1, n):
                if (ri - residues[j]) % G[i][j] == 0:
                    c += 1
        return c

    def best_flip(i):
        ri = residues[i]
        mi = moduli[i]
        base = conflict_count()
        best_r = ri
        best_c = base
        trial = [ri, mi // 2, 1, mi - 1, (ri + 1) % mi, (ri - 1) % mi]
        for _ in range(6):
            trial.append(random.randrange(mi))
        for r in set(t % mi for t in trial):
            if r == ri:
                continue
            residues[i] = r
            c = conflict_count()
            if c < best_c:
                best_c = c
                best_r = r
        residues[i] = best_r
        return best_r

    improved = True
    while improved and time.time() < DEADLINE - 8.0:
        improved = False
        order = list(range(n))
        random.shuffle(order)
        for i in order:
            old = residues[i]
            new = best_flip(i)
            if new != old:
                improved = True

    # conflict graph
    conf = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if (residues[i] - residues[j]) % G[i][j] == 0:
                conf[i][j] = conf[j][i] = True

    # greedy max independent set (min-degree pick)
    remaining = set(range(n))
    chosen = []
    while remaining:
        best = None
        bestdeg = None
        for v in remaining:
            deg = 0
            for u in remaining:
                if u != v and conf[v][u]:
                    deg += 1
            if bestdeg is None or deg < bestdeg:
                bestdeg = deg
                best = v
        chosen.append(best)
        remaining.discard(best)

    # greedy add pass (already independent by construction) - ensure validity
    def is_valid(S):
        for a in range(len(S)):
            for b in range(a + 1, len(S)):
                i, j = S[a], S[b]
                if (residues[i] - residues[j]) % G[i][j] == 0:
                    return False
        return True

    if not is_valid(chosen):
        # fallback: rebuild safely
        chosen = []
        for v in range(n):
            if all((residues[v] - residues[u]) % G[v][u] != 0 for u in chosen):
                chosen.append(v)

    result = [[residues[i], moduli[i]] for i in chosen]
    if not result:
        return [[1, 2]]
    return result
