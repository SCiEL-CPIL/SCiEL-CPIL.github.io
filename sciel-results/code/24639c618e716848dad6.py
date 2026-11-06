def construct():
    import time
    from math import gcd
    from itertools import combinations
    import random

    deadline = time.time() + 85
    N = 128
    universe = list(range(1, N + 1))

    g = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g[i][j] = gcd(i, j); g[j][i] = g[i][j]

    bad_triples = []
    for a, b, c in combinations(universe, 3):
        if g[a][b] == g[a][c] == g[b][c]:
            bad_triples.append((a, b, c))
    T = len(bad_triples)

    # elem -> list of (triple_idx, other1, other2)
    elem_triples = {x: [] for x in universe}
    for idx, (a, b, c) in enumerate(bad_triples):
        elem_triples[a].append((idx, b, c))
        elem_triples[b].append((idx, a, c))
        elem_triples[c].append((idx, a, b))

    def greedy_removal(order):
        S = set(universe)
        active = [True]*T  # triple active if all 3 in S
        deg = [0]*(N+1)
        # initialize deg
        for idx in range(T):
            a, b, c = bad_triples[idx]
            deg[a]+=1; deg[b]+=1; deg[c]+=1
        pos = {x: i for i, x in enumerate(order)}
        while True:
            # find element in S with max deg
            best=None; maxd=-1
            for x in S:
                if deg[x] > maxd or (deg[x]==maxd and (best is None or pos[x]<pos[best])):
                    maxd=deg[x]; best=x
            if maxd==0:
                break
            # remove best
            S.remove(best)
            for (idx, y, z) in elem_triples[best]:
                if active[idx]:
                    active[idx]=False
                    a,b,c=bad_triples[idx]
                    deg[a]-=1; deg[b]-=1; deg[c]-=1
        return S

    def try_add(S):
        # maintain active set of bad triples within S
        active = set()
        for idx,(a,b,c) in enumerate(bad_triples):
            if a in S and b in S and c in S:
                active.add(idx)
        changed=True
        while changed and time.time()<deadline:
            changed=False
            for x in sorted(set(universe)-S):
                # check if adding x creates any bad triple: need a triple containing x with other two in S
                ok=True
                for (idx,y,z) in elem_triples[x]:
                    if y in S and z in S:
                        ok=False; break
                if ok:
                    S.add(x)
                    # update active: add triples containing x now fully in S
                    for (idx,y,z) in elem_triples[x]:
                        if y in S and z in S:
                            active.add(idx)
                    changed=True
                    break
        return S

    best=None; best_size=-1
    rng=random.Random(7)
    orders=[list(range(1,N+1)), list(range(N,0,-1))]
    for _ in range(10):
        o=list(universe); rng.shuffle(o); orders.append(o)
    for order in orders:
        if time.time()>deadline-4: break
        S=greedy_removal(order)
        S=try_add(S)
        if len(S)>best_size:
            best_size=len(S); best=S
    if not best: best={1}
    # verify
    for (a,b,c) in bad_triples:
        if a in best and b in best and c in best:
            best={1}; break
    return sorted(best)
