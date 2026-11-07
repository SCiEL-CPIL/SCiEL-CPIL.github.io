import time
import random

# EVOLVE-BLOCK-START
P = {'n': 15}


def construct():
    n = P['n']
    start = time.time()
    deadline = start + 80.0

    def _check_valid(adj, n):
        deg = [len(adj[i]) for i in range(n)]
        for u in range(n):
            for v in adj[u]:
                if u >= v:
                    continue
                for s in adj[u]:
                    if s == v:
                        continue
                    for t in adj[v]:
                        if t == u or t == s:
                            continue
                        if deg[s] == deg[t]:
                            return False
        return True

    def _edge_count(adj):
        return sum(len(s) for s in adj) // 2

    best_adj = [set() for _ in range(n)]
    best_count = 0

    # Strategy: start with a structured graph, then optimize
    # Try multiple random seeds and edge orderings

    for trial in range(50):
        if time.time() > deadline - 3.0:
            break

        adj = [set() for _ in range(n)]
        rng = random.Random(42 + trial)

        # Random edge order
        all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_edges)

        # Greedy add
        for (i, j) in all_edges:
            if time.time() > deadline - 1.0:
                break
            adj[i].add(j)
            adj[j].add(i)
            if not _check_valid(adj, n):
                adj[i].discard(j)
                adj[j].discard(i)

        cnt = _edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]

        # Local search with multiple passes
        for ls_pass in range(50):
            if time.time() > deadline - 1.0:
                break

            improved = False
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) if j not in adj[i]]
            existing = [(i, j) for i in range(n) for j in adj[i] if i < j]

            rng.shuffle(missing)
            rng.shuffle(existing)

            # Try adding an edge
            for (i, j) in missing:
                if time.time() > deadline - 1.0:
                    break
                adj[i].add(j)
                adj[j].add(i)
                if _check_valid(adj, n):
                    improved = True
                    break
                else:
                    adj[i].discard(j)
                    adj[j].discard(i)

            if not improved:
                # Try a swap: remove one edge, add another
                for (a, b) in existing:
                    if time.time() > deadline - 1.0:
                        break
                    for (c, d) in missing:
                        if (c, d) == (a, b) or (c, d) == (b, a):
                            continue
                        adj[a].discard(b)
                        adj[b].discard(a)
                        adj[c].add(d)
                        adj[d].add(c)
                        if _check_valid(adj, n):
                            improved = True
                            break
                        else:
                            adj[c].discard(d)
                            adj[d].discard(c)
                            adj[a].add(b)
                            adj[b].add(a)
                    if improved:
                        break

            if not improved:
                break

        cnt = _edge_count(adj)
        if cnt > best_count:
            best_count = cnt
            best_adj = [set(s) for s in adj]

    edges = []
    for i in range(n):
        for j in best_adj[i]:
            if i < j:
                edges.append([int(i), int(j)])
    return edges

# EVOLVE-BLOCK-END
