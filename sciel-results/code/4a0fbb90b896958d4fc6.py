import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55.0

    def check_edge_valid(u, v, deg, adj):
        """Check if edge (u,v) is valid given current deg and adj."""
        nu = [s for s in adj[u] if s != v]
        nv = [t for t in adj[v] if t != u]
        deg_map = {}
        for t in nv:
            d = deg[t]
            if d not in deg_map:
                deg_map[d] = []
            deg_map[d].append(t)
        for s in nu:
            ds = deg[s]
            if ds in deg_map:
                lst = deg_map[ds]
                if len(lst) > 1:
                    return False
                elif len(lst) == 1:
                    if lst[0] != s:
                        return False
        return True

    def is_fully_valid(edge_set, n):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        for (u, v) in edge_set:
            if not check_edge_valid(u, v, deg, adj):
                return False
        return True

    # Known valid solution: K_{7,8}
    A = list(range(7))
    B = list(range(7, 15))
    base_edges = [(a, b) for a in A for b in B]
    base_set = set(tuple(sorted(e)) for e in base_edges)
    best_set = base_set
    best_count = len(base_set)

    # Candidate edges: within A and within B
    candidates = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            candidates.append((A[i], A[j]))
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            candidates.append((B[i], B[j]))

    for restart in range(40):
        if time.time() > deadline:
            break

        current_set = set(base_set)
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in current_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)

        not_in = [e for e in candidates if tuple(sorted(e)) not in current_set]
        random.shuffle(not_in)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx, e in enumerate(not_in):
                if time.time() > deadline:
                    break
                u, v = e
                su = tuple(sorted(e))
                if su in current_set:
                    continue

                old_deg_u = deg[u]
                old_deg_v = deg[v]
                deg[u] += 1
                deg[v] += 1
                adj[u].add(v)
                adj[v].add(u)

                all_ok = check_edge_valid(u, v, deg, adj)
                if all_ok:
                    for w in list(adj[u]):
                        if w == v:
                            continue
                        if not check_edge_valid(u, w, deg, adj):
                            all_ok = False
                            break
                if all_ok:
                    for w in list(adj[v]):
                        if w == u:
                            continue
                        if not check_edge_valid(v, w, deg, adj):
                            all_ok = False
                            break

                if all_ok:
                    current_set.add(su)
                    not_in.pop(idx)
                    improved = True
                    if len(current_set) > best_count:
                        best_set = set(current_set)
                        best_count = len(current_set)
                    break
                else:
                    deg[u] = old_deg_u
                    deg[v] = old_deg_v
                    adj[u].discard(v)
                    adj[v].discard(u)

        if len(current_set) > best_count:
            best_set = set(current_set)
            best_count = len(current_set)

    if not is_fully_valid(best_set, n):
        best_set = base_set
        best_count = len(base_set)

    result = [list(e) for e in best_set]
    result.sort()
    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])

    return result
