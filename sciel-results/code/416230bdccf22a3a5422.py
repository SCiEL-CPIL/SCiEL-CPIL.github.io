<<<<<<< SEARCH
def _try_reduce(coloring, adj, deadline):
    n = len(coloring)
    valid, color_pairs, pair_edges = _fast_check(coloring)
    if not valid:
        return coloring

    # adjacency set for the quick "adjacent" pre-filter
    adj_set = [set(a) for a in adj]

    def rebuild(coloring):
        for u, v in P['edges']:
            if coloring[u] == coloring[v]:
                return False, None, None
        pe = defaultdict(list)
        for u, v in P['edges']:
            cu, cv = coloring[u], coloring[v]
            key = (cu, cv) if cu < cv else (cv, cu)
            pe[key].append((u, v))
        for (c1, c2), edges in pe.items():
            parent = list(range(n))
            rank = [0] * n
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for u, v in edges:
                ru, rv = find(u), find(v)
                if ru == rv:
                    return False, None, None
                if rank[ru] < rank[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                if rank[ru] == rank[rv]:
                    rank[ru] += 1
        return True, list(pe.keys()), pe

    changed = True
    while changed and time.time() < deadline - 3:
        changed = False
        distinct = sorted(set(coloring))
        if len(distinct) <= 2:
            break
        counts = Counter(coloring)
        ordered = sorted(distinct, key=lambda c: (-counts[c], c))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                ca, cb = ordered[i], ordered[j]
                # Quick filter: no edge directly between ca and cb
                if (ca, cb) not in pair_edges and (cb, ca) not in pair_edges:
                    new_coloring = [ca if c == cb else c for c in coloring]
                    ok, cp, pe = rebuild(new_coloring)
                    if ok:
                        coloring = new_coloring
                        color_pairs, pair_edges = cp, pe
                        changed = True
                        break
            if changed:
                break
    return coloring
=======
def _try_reduce(coloring, adj, deadline):
    n = len(coloring)
    if not _fast_check(coloring):
        return coloring

    def rebuild(coloring):
        for u, v in P['edges']:
            if coloring[u] == coloring[v]:
                return False
        pe = defaultdict(list)
        for u, v in P['edges']:
            cu, cv = coloring[u], coloring[v]
            key = (cu, cv) if cu < cv else (cv, cu)
            pe[key].append((u, v))
        for (c1, c2), edges in pe.items():
            parent = list(range(n))
            rank = [0] * n
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for u, v in edges:
                ru, rv = find(u), find(v)
                if ru == rv:
                    return False
                if rank[ru] < rank[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                if rank[ru] == rank[rv]:
                    rank[ru] += 1
        return True

    changed = True
    while changed and time.time() < deadline - 3:
        changed = False
        distinct = sorted(set(coloring))
        if len(distinct) <= 2:
            break
        counts = Counter(coloring)
        ordered = sorted(distinct, key=lambda c: (-counts[c], c))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                ca, cb = ordered[i], ordered[j]
                new_coloring = [ca if c == cb else c for c in coloring]
                if rebuild(new_coloring):
                    coloring = new_coloring
                    changed = True
                    break
            if changed:
                break
    return coloring
>>>>>>> REPLACE

