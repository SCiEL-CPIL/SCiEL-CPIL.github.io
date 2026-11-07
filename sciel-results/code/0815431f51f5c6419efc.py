"""Rewritten program: seed with K7,8, then local-search to add intra-part edges."""
# EVOLVE-BLOCK-START
import time
import random

N = 15

def _is_valid(adj, deg):
    """Check if graph (adjacency matrix) has no violating 3-path s-u-v-t."""
    for u in range(N):
        for v in range(u + 1, N):
            if not adj[u][v]:
                continue
            for s in range(N):
                if s == u or s == v or not adj[s][u]:
                    continue
                for t in range(N):
                    if t == u or t == v or t == s or not adj[t][v]:
                        continue
                    if deg[s] == deg[t]:
                        return False
    return True

def _try_add_edge_fast(adj, deg, u, v):
    """Try adding edge (u,v); return (new_adj, new_deg) if valid, else None."""
    if u == v or adj[u][v]:
        return None
    
    nbr_u_mask = 0
    nbr_v_mask = 0
    for s in range(N):
        if s != v and adj[s][u]:
            nbr_u_mask |= (1 << s)
        if s != u and adj[s][v]:
            nbr_v_mask |= (1 << s)
    
    for s in range(N):
        if not (nbr_u_mask & (1 << s)):
            continue
        ds = deg[s]
        for t in range(N):
            if t == s or not (nbr_v_mask & (1 << t)):
                continue
            if ds == deg[t]:
                return None
    
    for s in range(N):
        if not (nbr_v_mask & (1 << s)):
            continue
        ds = deg[s]
        for t in range(N):
            if t == s or not (nbr_u_mask & (1 << t)):
                continue
            if ds == deg[t]:
                return None
    
    new_adj = [row[:] for row in adj]
    new_adj[u][v] = new_adj[v][u] = 1
    new_deg = deg[:]
    new_deg[u] += 1
    new_deg[v] += 1
    
    if _is_valid(new_adj, new_deg):
        return (new_adj, new_deg)
    return None

def _try_swap_fast(adj, deg, remove, add):
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
    
    if _is_valid(new_adj, new_deg):
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
    assert _is_valid(adj, deg), "Seed K7,8 should be valid"
    
    best_adj = [row[:] for row in adj]
    best_deg = deg[:]
    best_count = _edge_count(best_adj)
    
    random.seed(42)
    
    # Phase 1: greedily try to add any missing edge
    improved = True
    while improved and (time.time() < deadline - 5):
        improved = False
        for u in range(N):
            for v in range(u+1, N):
                if adj[u][v]:
                    continue
                res = _try_add_edge_fast(adj, deg, u, v)
                if res is not None:
                    adj, deg = res
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                    best_count = _edge_count(best_adj)
                    improved = True
                    break
            if improved:
                break
    
    # Phase 2: systematic 1-swap
    all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
    non_edges = [(u,v) for (u,v) in all_edges if not adj[u][v]]
    edges_present = [(u,v) for (u,v) in all_edges if adj[u][v]]
    
    for add in non_edges:
        if time.time() > deadline - 5:
            break
        for remove in edges_present:
            if remove == add:
                continue
            res = _try_swap_fast(adj, deg, remove, add)
            if res is not None:
                adj, deg = res
                cnt = _edge_count(adj)
                if cnt > best_count:
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                    best_count = cnt
                non_edges = [(u,v) for (u,v) in all_edges if not adj[u][v]]
                edges_present = [(u,v) for (u,v) in all_edges if adj[u][v]]
                break
    
    # Phase 3: fast random swaps with cached edge lists
    non_edges = [(u,v) for (u,v) in all_edges if not adj[u][v]]
    edges_present = [(u,v) for (u,v) in all_edges if adj[u][v]]
    
    for iteration in range(1000000):
        if time.time() > deadline - 3:
            break
        if not non_edges or not edges_present:
            break
        add = random.choice(non_edges)
        remove = random.choice(edges_present)
        if remove == add:
            continue
        res = _try_swap_fast(adj, deg, remove, add)
        if res is not None:
            adj, deg = res
            cnt = _edge_count(adj)
            if cnt > best_count:
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
                best_count = cnt
            # Update cached lists
            non_edges = [(u,v) for (u,v) in all_edges if not adj[u][v]]
            edges_present = [(u,v) for (u,v) in all_edges if adj[u][v]]
    
    # Phase 4: final greedy add pass
    improved = True
    while improved and (time.time() < deadline - 1):
        improved = False
        for u in range(N):
            for v in range(u+1, N):
                if adj[u][v]:
                    continue
                res = _try_add_edge_fast(adj, deg, u, v)
                if res is not None:
                    adj, deg = res
                    best_adj = [row[:] for row in adj]
                    best_deg = deg[:]
                    best_count = _edge_count(best_adj)
                    improved = True
                    break
            if improved:
                break
    
    # Convert to edge list
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if best_adj[u][v]:
                edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
