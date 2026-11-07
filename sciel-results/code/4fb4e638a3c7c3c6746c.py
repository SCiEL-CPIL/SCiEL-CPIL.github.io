"""Rewritten program: compatibility-matrix based dense graph construction."""
# EVOLVE-BLOCK-START
import time
import random

N = 15

def _is_valid_fast(adj, deg):
    """Fast validity check using degree-count method per edge."""
    for u in range(N):
        for v in range(u + 1, N):
            if not adj[u][v]:
                continue
            # A = N(u) \ {u,v}, B = N(v) \ {u,v}
            deg_a = {}
            deg_b = {}
            for s in range(N):
                if s == u or s == v:
                    continue
                if adj[s][u]:
                    d = deg[s]
                    deg_a[d] = deg_a.get(d, 0) + 1
                if adj[s][v]:
                    d = deg[s]
                    deg_b[d] = deg_b.get(d, 0) + 1
            # Check for violations: exists s in A, t in B, s!=t, deg[s]==deg[t]
            for d in deg_a:
                if d in deg_b:
                    ca = deg_a[d]
                    cb = deg_b[d]
                    if ca + cb >= 2:
                        if ca == 1 and cb == 1:
                            # Find the unique vertex in A with degree d
                            s_val = -1
                            for s in range(N):
                                if s != u and s != v and adj[s][u] and deg[s] == d:
                                    s_val = s
                                    break
                            # If s_val is also in B (adj to v), same vertex, no violation
                            if s_val != -1 and adj[s_val][v]:
                                continue
                        return False
    return True

def _try_add_edge(adj, deg, u, v):
    """Try adding edge (u,v); return (new_adj, new_deg) if valid, else None."""
    if u == v or adj[u][v]:
        return None
    new_adj = [row[:] for row in adj]
    new_adj[u][v] = new_adj[v][u] = 1
    new_deg = deg[:]
    new_deg[u] += 1
    new_deg[v] += 1
    if _is_valid_fast(new_adj, new_deg):
        return (new_adj, new_deg)
    return None

def _try_swap(adj, deg, remove, add):
    """Remove edge `remove`, add edge `add`. Return (new_adj, new_deg) if valid."""
    (ru, rv) = remove
    (au, av) = add
    if au == av or adj[au][av]:
        return None
    if not adj[ru][rv]:
        return None
    new_adj = [row[:] for row in adj]
    new_adj[ru][rv] = new_adj[rv][ru] = 0
    new_adj[au][av] = new_adj[av][au] = 1
    new_deg = deg[:]
    new_deg[ru] -= 1
    new_deg[rv] -= 1
    new_deg[au] += 1
    new_deg[av] += 1
    if _is_valid_fast(new_adj, new_deg):
        return (new_adj, new_deg)
    return None

def _edge_count(adj):
    return sum(sum(row) for row in adj) // 2

def construct():
    start = time.time()
    deadline = start + 85  # 5s safety margin before 90s kill

    # Seed: K7,8 bipartite
    adj = [[0]*N for _ in range(N)]
    deg = [0]*N
    for a in range(7):
        for b in range(7, 15):
            adj[a][b] = adj[b][a] = 1
    for a in range(7):
        deg[a] = 8
    for b in range(7, 15):
        deg[b] = 7
    assert _is_valid_fast(adj, deg), "Seed K7,8 should be valid"

    best_adj = [row[:] for row in adj]
    best_deg = deg[:]
    best_count = _edge_count(best_adj)

    random.seed(42)

    # Precompute all possible edges
    all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
    total_edges = len(all_edges)

    # Phase 1: greedily try to add any missing edge
    improved = True
    while improved and (time.time() < deadline - 15):
        improved = False
        for u in range(N):
            for v in range(u+1, N):
                if adj[u][v]:
                    continue
                res = _try_add_edge(adj, deg, u, v)
                if res is not None:
                    adj, deg = res
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                    best_count = _edge_count(best_adj)
                    improved = True
                    break
            if improved:
                break

    # Phase 2: fast 2-swap local search with precomputed edge lists
    # Maintain running lists of present/absent edges
    present_edges = [(u, v) for (u, v) in all_edges if adj[u][v]]
    absent_edges = [(u, v) for (u, v) in all_edges if not adj[u][v]]

    for iteration in range(2000000):
        if time.time() > deadline - 10:
            break
        if not absent_edges or not present_edges:
            break
        add = random.choice(absent_edges)
        remove = random.choice(present_edges)
        res = _try_swap(adj, deg, remove, add)
        if res is not None:
            adj, deg = res
            # Update edge lists
            present_edges.remove(remove)
            absent_edges.remove(add)
            present_edges.append(add)
            absent_edges.append(remove)
            cnt = _edge_count(adj)
            if cnt > best_count:
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
                best_count = cnt

    # Phase 3: try adding edges again
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        for u in range(N):
            for v in range(u+1, N):
                if adj[u][v]:
                    continue
                res = _try_add_edge(adj, deg, u, v)
                if res is not None:
                    adj, deg = res
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                    best_count = _edge_count(best_adj)
                    improved = True
                    break
            if improved:
                break

    # Phase 4: final polish - exhaustive single-edge add attempts
    if time.time() < deadline - 2:
        for u in range(N):
            for v in range(u+1, N):
                if adj[u][v]:
                    continue
                res = _try_add_edge(adj, deg, u, v)
                if res is not None:
                    adj, deg = res
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                    best_count = _edge_count(best_adj)

    # Phase 5: one more round of swaps with different random seed
    if time.time() < deadline - 1:
        random.seed(123)
        present_edges = [(u, v) for (u, v) in all_edges if adj[u][v]]
        absent_edges = [(u, v) for (u, v) in all_edges if not adj[u][v]]
        for iteration in range(500000):
            if time.time() > deadline - 0.5:
                break
            if not absent_edges or not present_edges:
                break
            add = random.choice(absent_edges)
            remove = random.choice(present_edges)
            res = _try_swap(adj, deg, remove, add)
            if res is not None:
                adj, deg = res
                present_edges.remove(remove)
                absent_edges.remove(add)
                present_edges.append(add)
                absent_edges.append(remove)
                cnt = _edge_count(adj)
                if cnt > best_count:
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                    best_count = cnt

    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if best_adj[u][v]:
                edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
