import time
import sys
from random import Random
from collections import defaultdict


def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)

        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)

        internal_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                if v in adj[u]:
                    internal_edges.add((u, v))

        i_c_edges = {}
        for v in I:
            nbrs = adj[v] & C_set
            if nbrs:
                i_c_edges[v] = nbrs

        groups = defaultdict(list)
        for v in I:
            if i_c_edges[v]:
                groups[frozenset(i_c_edges[v])].append(v)

        best_partition = None
        best_count = float('inf')

        for strategy in range(2):
            partition = []
            used_c_edges = set()

            if strategy == 0:
                group_order = sorted(groups.items(), key=lambda x: -len(x[0]))
            else:
                group_order = sorted(groups.items(), key=lambda x: len(x[0]))

            for shadow, i_vertices in group_order:
                if not i_vertices:
                    continue

                avail = [u for u in shadow]

                def unused_degree(u):
                    cnt = 0
                    for w in avail:
                        if u != w:
                            e = (min(u, w), max(u, w))
                            if e in internal_edges and e not in used_c_edges:
                                cnt += 1
                    return cnt

                avail.sort(key=lambda u: -unused_degree(u))

                clique = []
                for u in avail:
                    ok = True
                    for w in clique:
                        e = (min(u, w), max(u, w))
                        if e not in internal_edges or e in used_c_edges:
                            ok = False
                            break
                    if ok:
                        clique.append(u)

                if len(clique) >= 1:
                    partition.append(sorted(i_vertices + clique))
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            used_c_edges.add((min(clique[i], clique[j]), max(clique[i], clique[j])))

                for v in i_vertices:
                    remaining_nbrs = shadow - set(clique)
                    for u in remaining_nbrs:
                        partition.append(sorted([v, u]))

            rem_set = set(internal_edges - used_c_edges)

            while rem_set:
                rem_adj = {u: set() for u in range(k)}
                for (a, b) in rem_set:
                    rem_adj[a].add(b)
                    rem_adj[b].add(a)

                max_deg = -1
                start_v = None
                for u in range(k):
                    if len(rem_adj[u]) > max_deg:
                        max_deg = len(rem_adj[u])
                        start_v = u
                if start_v is None:
                    break

                clique = [start_v]
                candidates = sorted(rem_adj[start_v], key=lambda x: -len(rem_adj[x]))
                for w in candidates:
                    ok = True
                    for u in clique:
                        if w not in rem_adj[u]:
                            ok = False
                            break
                    if ok:
                        clique.append(w)

                if len(clique) >= 2:
                    partition.append(sorted(clique))
                    for i in range(len(clique)):
                        for j in range(i + 1, len(clique)):
                            rem_set.discard((min(clique[i], clique[j]), max(clique[i], clique[j])))

            cnt = len(partition)
            if cnt < best_count:
                best_count = cnt
                best_partition = partition

        if best_partition is None:
            return []

        cleaned = []
        for clique in best_partition:
            cleaned.append([int(x) for x in clique])
        return cleaned

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        results.append(res)

    return results
