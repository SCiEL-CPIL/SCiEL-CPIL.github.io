import time
import random

def construct():
    n = 15
    deadline = time.time() + 87.0
    best_edges = None
    best_count = -1

    def build_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def check_adj(adj, deg):
        """Check validity using pre-built adj and deg."""
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    continue
                for s in range(n):
                    if s == u or s == v:
                        continue
                    if not (adj[u] >> s) & 1:
                        continue
                    for t in range(n):
                        if t == u or t == v or t == s:
                            continue
                        if not (adj[v] >> t) & 1:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True

    def check(edges):
        adj = build_adj(edges)
        deg = [bin(adj[i]).count('1') for i in range(n)]
        return check_adj(adj, deg)

    def try_improve(start_edges):
        edges = list(start_edges)
        adj = build_adj(edges)
        deg = [bin(adj[i]).count('1') for i in range(n)]
        present = set()
        for u, v in edges:
            a, b = (u, v) if u < v else (v, u)
            present.add((a, b))

        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for u in range(n):
                if time.time() > deadline - 2:
                    break
                for v in range(u + 1, n):
                    if (u, v) in present:
                        continue
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    deg[u] += 1
                    deg[v] += 1
                    if check_adj(adj, deg):
                        present.add((u, v))
                        edges.append([u, v])
                        improved = True
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                        deg[u] -= 1
                        deg[v] -= 1
        return edges

    def local_search(start_edges):
        edges = list(start_edges)
        adj = build_adj(edges)
        deg = [bin(adj[i]).count('1') for i in range(n)]
        present = set()
        for u, v in edges:
            a, b = (u, v) if u < v else (v, u)
            present.add((a, b))

        best = len(edges)
        best_copy = [e[:] for e in edges]

        # Greedy add
        for u in range(n):
            if time.time() > deadline - 3:
                return best_copy
            for v in range(u + 1, n):
                if (u, v) in present:
                    continue
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                deg[u] += 1
                deg[v] += 1
                if check_adj(adj, deg):
                    present.add((u, v))
                    edges.append([u, v])
                    if len(edges) > best:
                        best = len(edges)
                        best_copy = [e[:] for e in edges]
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    deg[u] -= 1
                    deg[v] -= 1

        # 2-opt: remove one edge, try to add back greedily
        if time.time() < deadline - 8:
            edges2 = [e[:] for e in best_copy]
            adj2 = build_adj(edges2)
            deg2 = [bin(adj2[i]).count('1') for i in range(n)]
            present2 = set()
            for u, v in edges2:
                a, b = (u, v) if u < v else (v, u)
                present2.add((a, b))

            for _ in range(5):
                if time.time() > deadline - 4:
                    break
                random.seed(_ + 200)
                if not edges2:
                    break
                idx = random.randint(0, len(edges2) - 1)
                ru, rv = edges2[idx]
                edges2.pop(idx)
                adj2[ru] &= ~(1 << rv)
                adj2[rv] &= ~(1 << ru)
                deg2[ru] -= 1
                deg2[rv] -= 1
                present2.discard((ru, rv) if ru < rv else (rv, ru))

                # Now greedily add
                improved2 = True
                while improved2 and time.time() < deadline - 3:
                    improved2 = False
                    for a in range(n):
                        if time.time() > deadline - 3:
                            break
                        for b in range(a + 1, n):
                            if (a, b) in present2:
                                continue
                            adj2[a] |= (1 << b)
                            adj2[b] |= (1 << a)
                            deg2[a] += 1
                            deg2[b] += 1
                            if check_adj(adj2, deg2):
                                present2.add((a, b))
                                edges2.append([a, b])
                                improved2 = True
                            else:
                                adj2[a] &= ~(1 << b)
                                adj2[b] &= ~(1 << a)
                                deg2[a] -= 1
                                deg2[b] -= 1
                if len(edges2) > best:
                    best = len(edges2)
                    best_copy = [e[:] for e in edges2]

        return best_copy

    # Generate candidates
    candidates = []

    # Bipartite graphs
    for a in range(1, 15):
        edges = []
        A = list(range(a))
        B = list(range(a, 15))
        for x in A:
            for y in B:
                edges.append([x, y])
        candidates.append(edges)

    # Complete graph
    complete = [[u, v] for u in range(15) for v in range(u + 1, 15)]
    candidates.append(complete)

    # Empty
    candidates.append([])

    # Random graphs with varying densities
    for seed in range(15):
        random.seed(seed)
        density = 0.2 + 0.3 * (seed / 15.0)
        edges = []
        for u in range(15):
            for v in range(u + 1, 15):
                if random.random() < density:
                    edges.append([u, v])
        candidates.append(edges)

    # Near-bipartite: bipartite + a few cross edges within partitions
    for seed in range(5):
        random.seed(1000 + seed)
        a = random.randint(5, 10)
        A = list(range(a))
        B = list(range(a, 15))
        edges = []
        for x in A:
            for y in B:
                edges.append([x, y])
        # Add some edges within A
        for i in range(len(A)):
            for j in range(i + 1, len(A)):
                if random.random() < 0.3:
                    edges.append([A[i], A[j]])
        # Add some edges within B
        for i in range(len(B)):
            for j in range(i + 1, len(B)):
                if random.random() < 0.3:
                    edges.append([B[i], B[j]])
        candidates.append(edges)

    def fix_invalid(edges):
        if check(edges):
            return edges
        adj = build_adj(edges)
        deg = [bin(adj[i]).count('1') for i in range(n)]
        present = set()
        for u, v in edges:
            a, b = (u, v) if u < v else (v, u)
            present.add((a, b))

        changed = True
        while changed and time.time() < deadline - 5:
            changed = False
            if check_adj(adj, deg):
                break
            # Find all violations and count which edge appears most
            edge_violation_count = {}
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] >> v) & 1:
                        continue
                    for s in range(n):
                        if s == u or s == v:
                            continue
                        if not (adj[u] >> s) & 1:
                            continue
                        for t in range(n):
                            if t == u or t == v or t == s:
                                continue
                            if not (adj[v] >> t) & 1:
                                continue
                            if deg[s] == deg[t]:
                                edge_violation_count[(u, v)] = edge_violation_count.get((u, v), 0) + 1
            if not edge_violation_count:
                break
            # Remove the edge with most violations
            worst_edge = max(edge_violation_count, key=lambda e: edge_violation_count[e])
            edges = [e for e in edges if not ((e[0] == worst_edge[0] and e[1] == worst_edge[1]) or
                                              (e[0] == worst_edge[1] and e[1] == worst_edge[0]))]
            adj = build_adj(edges)
            deg = [bin(adj[i]).count('1') for i in range(n)]
            changed = True
        return edges

    for cand in candidates:
        if time.time() > deadline - 3:
            break
        cand = fix_invalid(cand)
        improved = try_improve(cand)
        if check(improved):
            cnt = len(improved)
            if cnt > best_count:
                best_count = cnt
                best_edges = improved

        if best_edges is not None and time.time() < deadline - 8:
            ls_result = local_search(best_edges)
            if check(ls_result):
                cnt = len(ls_result)
                if cnt > best_count:
                    best_count = cnt
                    best_edges = ls_result

    if best_edges is None:
        return [[0, v] for v in range(1, 15)]

    out = [[int(u), int(v)] for u, v in best_edges]
    return out
