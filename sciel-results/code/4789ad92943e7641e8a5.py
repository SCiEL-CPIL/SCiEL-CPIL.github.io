import time
import random

def _has_c5(adj, n):
    for v0 in range(n):
        for v4 in adj[v0]:
            if v4 <= v0:
                continue
            nv0 = adj[v0] - {v4}
            nv4 = adj[v4] - {v0}
            for v1 in nv0:
                for v3 in nv4:
                    if v1 == v3:
                        continue
                    c = adj[v1] & adj[v3]
                    c -= {v0, v1, v3, v4}
                    if c:
                        return True
    return False

def _alpha_ge_9(adj, n):
    comp = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                comp[i].add(j)
                comp[j].add(i)
    found = [False]
    def search(sz, cand):
        if found[0]:
            return
        if sz >= 9:
            found[0] = True
            return
        if not cand:
            return
        if sz + len(cand) < 9:
            return
        v = max(cand, key=lambda x: len(comp[x] & cand))
        search(sz + 1, cand & comp[v])
        if found[0]:
            return
        search(sz, cand - {v})
    search(0, frozenset(range(n)))
    return found[0]

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    # Fallback: 8 disjoint K4s (n=32)
    edges32 = []
    for b in range(8):
        for i in range(4):
            for j in range(i + 1, 4):
                edges32.append([b * 4 + i, b * 4 + j])
    adj32 = [set() for _ in range(32)]
    for u, v in edges32:
        adj32[u].add(v)
        adj32[v].add(u)
    if _has_c5(adj32, 32) or _alpha_ge_9(adj32, 32):
        return {'n': 0, 'edges': []}
    best_n = 32
    best_edges = edges32

    # Search for n = 33, 34, 35
    for tn in [33, 34, 35]:
        if time.time() > deadline - 35.0:
            break
        found = False
        for attempt in range(500):
            if time.time() > deadline - 10.0:
                break
            if found:
                break
            adj = [set() for _ in range(tn)]
            for i in range(tn):
                for j in range(i + 1, tn):
                    if random.random() < 0.55:
                        adj[i].add(j)
                        adj[j].add(i)
            for _ in range(2000):
                if time.time() > deadline - 5.0:
                    break
                if not _has_c5(adj, tn):
                    break
                all_e = []
                for u in range(tn):
                    for v in adj[u]:
                        if u < v:
                            all_e.append((u, v))
                if not all_e:
                    break
                u, v = random.choice(all_e)
                adj[u].discard(v)
                adj[v].discard(u)
            if _has_c5(adj, tn):
                continue
            if not _alpha_ge_9(adj, tn):
                ed = [[u, v] for u in range(tn) for v in adj[u] if u < v]
                best_n = tn
                best_edges = ed
                found = True
                break

    return {'n': best_n, 'edges': best_edges}
