import time
import random
from collections import defaultdict


def construct():
    n = 15
    start = time.time()
    deadline = start + 80.0          # safety margin before the 90s kill
    rng = random.Random(20240601)

    def star_adj():
        a = [set() for _ in range(n)]
        for v in range(1, n):
            a[0].add(v)
            a[v].add(0)
        return a

    def has_3path(adj, s, t):
        # simple 3-edge path s-u-v-t with s,u,v,t distinct
        for u in adj[s]:
            if u == t:
                continue
            for v in adj[u]:
                if v == s or v == u:
                    continue
                if v in adj[t]:
                    return True
        return False

    def valid(adj):
        deg = [len(adj[i]) for i in range(n)]
        groups = defaultdict(list)
        for i, d in enumerate(deg):
            groups[d].append(i)
        for d, vs in groups.items():
            if len(vs) < 2:
                continue
            for a in range(len(vs)):
                for b in range(a + 1, len(vs)):
                    if has_3path(adj, vs[a], vs[b]):
                        return False
        return True

    def edge_list(adj):
        out = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    out.append([int(i), int(j)])
        return out

    def count(adj):
        return sum(len(s) for s in adj) // 2

    def all_edges(adj):
        return [(i, j) for i in range(n) for j in range(i + 1, n) if j in adj[i]]

    def all_nonedges(adj):
        return [(i, j) for i in range(n) for j in range(i + 1, n) if j not in adj[i]]

    def hill_climb(adj, t_end):
        while time.time() < t_end:
            nonedges = all_nonedges(adj)
            rng.shuffle(nonedges)
            progress = False
            for (u, v) in nonedges:
                if time.time() > t_end:
                    break
                if v in adj[u]:
                    continue
                adj[u].add(v)
                adj[v].add(u)
                if valid(adj):
                    progress = True
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)

            if progress:
                continue

            # no single add works: try edge swaps to rearrange
            edges = all_edges(adj)
            if not edges:
                break
            swapped = False
            for _ in range(250):
                if time.time() > t_end:
                    break
                e1 = rng.choice(edges)
                ne = all_nonedges(adj)
                if not ne:
                    break
                e2 = rng.choice(ne)
                adj[e1[0]].discard(e1[1])
                adj[e1[1]].discard(e1[0])
                adj[e2[0]].add(e2[1])
                adj[e2[1]].add(e2[0])
                if valid(adj):
                    swapped = True
                    edges = all_edges(adj)
                else:
                    adj[e1[0]].add(e1[1])
                    adj[e1[1]].add(e1[0])
                    adj[e2[0]].discard(e2[1])
                    adj[e2[1]].discard(e2[0])

            if not swapped:
                # escape local optimum: drop a few random edges, then retry adds
                for _ in range(rng.randint(1, 3)):
                    if not edges:
                        break
                    e1 = rng.choice(edges)
                    adj[e1[0]].discard(e1[1])
                    adj[e1[1]].discard(e1[0])
                edges = all_edges(adj)
                if not edges:
                    break
        return adj

    best = star_adj()
    best_c = count(best)

    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        if attempts == 1:
            adj = star_adj()
        else:
            adj = [set() for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    if rng.random() < 0.5:
                        adj[i].add(j)
                        adj[j].add(i)
            if not valid(adj):
                adj = star_adj()
        t_end = min(deadline - 5.0, time.time() + 12.0)
        adj = hill_climb(adj, t_end)
        if valid(adj):
            c = count(adj)
            if c > best_c:
                best_c = c
                best = [set(s) for s in adj]

    if not valid(best):
        best = star_adj()

    return edge_list(best)
