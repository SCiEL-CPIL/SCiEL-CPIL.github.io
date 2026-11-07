"""Rewritten program: pack edge-disjoint Hamiltonian cycles on a shared vertex set."""
import time
import random
import sys
import itertools
sys.setrecursionlimit(1000000)

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}


def construct():
    edges = P['edges']
    n = P['n']
    all_vertices = list(range(n))
    edge_set = set((min(a, b), max(a, b)) for a, b in edges)
    adj = {v: set() for v in all_vertices}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    random.seed(12345)
    time_limit = time.time() + 54.0

    def find_one_cycle(S, Sset, used_edges, start, max_steps):
        nS = len(S)
        found = [None]

        def avail(v):
            out = []
            for w in adj[v]:
                if w in Sset and (min(v, w), max(v, w)) not in used_edges:
                    out.append(w)
            return out

        path = [start]
        visited = {start}
        steps = [0]

        def dfs():
            if found[0] is not None or steps[0] > max_steps:
                return
            steps[0] += 1
            if len(path) == nS:
                last = path[-1]
                if (min(last, start), max(last, start)) not in used_edges:
                    found[0] = list(path)
                return
            v = path[-1]
            cands = [w for w in avail(v) if w not in visited]
            # Warnsdorff heuristic: prefer vertices with fewest onward options
            cands.sort(key=lambda w: len([x for x in avail(w) if x not in visited and x != start]))
            for w in cands:
                path.append(w)
                visited.add(w)
                dfs()
                if found[0] is not None:
                    return
                path.pop()
                visited.remove(w)

        dfs()
        return found[0]

    def try_pack(S, k, time_limit, first_restarts=25, rest_restarts=40):
        S = sorted(S)
        Sset = set(S)
        used = set()
        cycles = []
        for ci in range(k):
            if time.time() > time_limit:
                break
            restarts = first_restarts if ci == 0 else rest_restarts
            found = None
            for _ in range(restarts):
                if time.time() > time_limit:
                    break
                start = random.choice(S)
                c = find_one_cycle(S, Sset, used, start, max_steps=150000)
                if c is not None:
                    found = c
                    break
            if found is None:
                break
            cyc = found
            for i in range(len(cyc)):
                a = cyc[i]
                b = cyc[(i + 1) % len(cyc)]
                used.add((min(a, b), max(a, b)))
            cycles.append(cyc)
        return cycles

    def valid(cycles):
        if not cycles:
            return False
        Sset = set(cycles[0])
        if len(Sset) < 3:
            return False
        used = set()
        for c in cycles:
            if set(c) != Sset:
                return False
            if len(set(c)) != len(c):
                return False
            L = len(c)
            if L < 3:
                return False
            for i in range(L):
                a = c[i]
                b = c[(i + 1) % L]
                e = (min(a, b), max(a, b))
                if e in used or e not in edge_set:
                    return False
                used.add(e)
        return True

    best = []

    # Primary: full vertex set, try 3 then 2
    for k in (3, 2):
        c = try_pack(all_vertices, k, time_limit)
        if len(c) > len(best):
            best = c
        if len(best) == 3:
            break

    # Fallback: try smaller vertex subsets (removing 1-3 vertices) to reach 3
    if len(best) < 3:
        for rem in range(1, 4):
            if time.time() > time_limit:
                break
            for S in itertools.combinations(all_vertices, n - rem):
                if time.time() > time_limit:
                    break
                S = list(S)
                c = try_pack(S, 3, time_limit, first_restarts=12, rest_restarts=20)
                if len(c) > len(best):
                    best = c
                    if len(best) == 3:
                        break
            if len(best) == 3:
                break

    if not valid(best):
        best = [list(range(n))]

    out = [ [int(v) for v in c] for c in best ]
    return out
