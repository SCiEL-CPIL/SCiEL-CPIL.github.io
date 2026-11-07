"""Rewritten program: minimum clique cover of three split graphs."""
# EVOLVE-BLOCK-START
import time
import heapq
from itertools import combinations

def construct():
    DEADLINE = time.time() + 130.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    # ---- Bron–Kerbosch maximal clique enumeration with pivot ----
    def maximal_cliques(adj):
        n = len(adj)
        best = []

        def bk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    best.append(tuple(sorted(R)))
                return
            # pivot: choose u in P ∪ X maximizing |P ∩ N(u)|
            U = list(P) + list(X)
            if not U:
                return
            # pick pivot with max |P ∩ N(u)|
            pivot = max(U, key=lambda u: sum(1 for w in P if w in adj[u]))
            cands = [v for v in P if v not in adj[pivot]]
            for v in cands:
                Nv = adj[v]
                bk(R | {v}, P & Nv, X & Nv)
                P = P - {v}
                X = X | {v}

        bk(frozenset(), frozenset(range(n)), frozenset())
        return best

    def solve_one(edges):
        if not edges:
            return []
        verts = set()
        for u, v in edges:
            verts.add(u)
            verts.add(v)
        verts = sorted(verts)
        idx = {v: i for i, v in enumerate(verts)}
        n = len(verts)
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            i, j = idx[u], idx[v]
            adj[i].add(j)
            adj[j].add(i)
            edge_set.add((min(i, j), max(i, j)))

        mcs = maximal_cliques(adj)
        # clique -> set of edge-indices it covers
        edge_list = list(edge_set)
        eindex = {e: i for i, e in enumerate(edge_list)}
        E = len(edge_list)

        def clique_edges(cl):
            s = set()
            lst = list(cl)
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    x, y = lst[a], lst[b]
                    s.add((min(x, y), max(x, y)))
            return s

        cover_sets = [clique_edges(c) for c in mcs]
        # remove cliques that are subsets of another (redundant)
        # keep all for now; greedy handles redundancy
        # greedy set cover
        uncovered = set(range(E))
        chosen = []
        # for speed, precompute each clique's edge list
        cliq_el = [sorted(s) for s in cover_sets]
        while uncovered and time.time() < DEADLINE - 20:
            best_c = -1
            best_cnt = -1
            for ci, el in enumerate(cliq_el):
                cnt = 0
                for e in el:
                    if e in uncovered:
                        cnt += 1
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = ci
            if best_c < 0:
                break
            chosen.append(best_c)
            for e in cliq_el[best_c]:
                uncovered.discard(e)
            if best_cnt == 0:
                break

        # local search: try removing a clique if its edges are covered by others
        def is_covered(ci, chosen_set):
            for e in cliq_el[ci]:
                ok = False
                for cj in chosen_set:
                    if cj != ci and e in cover_sets[cj]:
                        ok = True
                        break
                if not ok:
                    return False
            return True

        # a few refinement passes
        for _ in range(3):
            if time.time() >= DEADLINE - 20:
                break
            changed = True
            while changed:
                changed = False
                chosen_set = set(chosen)
                for ci in list(chosen):
                    if is_covered(ci, chosen_set):
                        chosen.remove(ci)
                        chosen_set.discard(ci)
                        changed = True
            # try single swaps: remove a, add b if reduces count (only useful if -1+1=-1? no) skip
        # convert to vertex lists
        result = []
        for ci in chosen:
            result.append([verts[v] for v in sorted(mcs[ci])])
        return result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    out = []
    for spec in specs:
        if time.time() >= DEADLINE - 10:
            # fallback: list edges as cliques (valid, though large)
            out.append([[int(u), int(v)] for (u, v) in instance(*spec)])
            continue
        edges = instance(*spec)
        out.append(solve_one(edges))
    # final safety: if we ran out of time, fill with edge-cliques
    for i, spec in enumerate(specs):
        if i >= len(out):
            out.append([[int(u), int(v)] for (u, v) in instance(*spec)])
    # ensure types are Python int and lists
    final = []
    for inst in out:
        ci = []
        for cl in inst:
            ci.append([int(x) for x in cl])
        final.append(ci)
    return final
# EVOLVE-BLOCK-END
