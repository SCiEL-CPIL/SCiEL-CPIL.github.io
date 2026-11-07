"""Rewritten program: multi-start local search with diverse seeds."""
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

def _greedy_add(adj, deg, deadline):
    """Repeatedly try to add edges until no more can be added or time runs out."""
    improved = True
    while improved and (time.time() < deadline):
        improved = False
        for u in range(N):
            for v in range(u+1, N):
                if adj[u][v]:
                    continue
                res = _try_add_edge(adj, deg, u, v)
                if res is not None:
                    adj, deg = res
                    improved = True
                    break
            if improved:
                break
    return adj, deg

def _local_search(adj, deg, deadline, rng, max_iter=200000):
    """Random 2-swap local search. Returns (adj, deg, count)."""
    all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
    count = _edge_count(adj)
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        non_edges = [(u,v) for (u,v) in all_edges if not adj[u][v]]
        if not non_edges:
            break
        edges_present = [(u,v) for (u,v) in all_edges if adj[u][v]]
        if not edges_present:
            break
        add = rng.choice(non_edges)
        remove = rng.choice(edges_present)
        res = _try_swap(adj, deg, remove, add)
        if res is not None:
            adj, deg = res
            new_count = _edge_count(adj)
            if new_count > count:
                count = new_count
    return adj, deg, count

def _make_seed_k78():
    """Create K7,8 bipartite seed."""
    adj = [[0]*N for _ in range(N)]
    deg = [0]*N
    for a in range(7):
        for b in range(7, 15):
            adj[a][b] = adj[b][a] = 1
    for a in range(7):
        deg[a] = 8
    for b in range(7, 15):
        deg[b] = 7
    return adj, deg

def _make_seed_random(rng, density=0.4):
    """Create a random valid seed by greedily adding edges."""
    adj = [[0]*N for _ in range(N)]
    deg = [0]*N
    # Start with a few random edges
    for _ in range(5):
        u, v = rng.sample(range(N), 2)
        if not adj[u][v]:
            adj[u][v] = adj[v][u] = 1
            deg[u] += 1
            deg[v] += 1
    # Greedily add more edges
    for u in range(N):
        for v in range(u+1, N):
            if adj[u][v]:
                continue
            res = _try_add_edge(adj, deg, u, v)
            if res is not None:
                adj, deg = res
    return adj, deg

def construct():
    start = time.time()
    deadline = start + 82  # 8s safety margin before 90s kill
    
    best_adj = None
    best_deg = None
    best_count = 0
    
    # Multi-start with different seeds
    seeds = []
    
    # Seed 1: K7,8 bipartite
    seeds.append(("k78", _make_seed_k78()))
    
    # Seed 2-5: Random seeds with different densities
    rng_init = random.Random(42)
    for i in range(4):
        adj, deg = _make_seed_random(rng_init, density=0.3 + 0.1*i)
        seeds.append((f"random_{i}", (adj, deg)))
    
    # Process each seed
    num_seeds = len(seeds)
    time_per_seed = (deadline - start) / num_seeds
    
    for idx, (name, (adj, deg)) in enumerate(seeds):
        seed_deadline = start + (idx + 1) * time_per_seed
        if time.time() > seed_deadline - 2:
            break
        
        # Greedy add phase
        adj, deg = _greedy_add(adj, deg, seed_deadline - 3)
        count = _edge_count(adj)
        
        # Local search phase
        rng = random.Random(42 + idx * 1000)
        adj, deg, count = _local_search(adj, deg, seed_deadline - 1, rng, max_iter=100000)
        
        # Final greedy add
        adj, deg = _greedy_add(adj, deg, seed_deadline - 0.5)
        count = _edge_count(adj)
        
        if count > best_count:
            best_count = count
            best_adj = [row[:] for row in adj]
            best_deg = deg[:]
    
    # If we have time left, do more local search on the best solution
    if best_adj is not None and time.time() < deadline - 3:
        remaining = deadline - 3 - time.time()
        if remaining > 1:
            rng = random.Random(999)
            adj, deg, count = _local_search(best_adj, best_deg, deadline - 1, rng, max_iter=300000)
            adj, deg = _greedy_add(adj, deg, deadline - 0.5)
            count = _edge_count(adj)
            if count > best_count:
                best_count = count
                best_adj = [row[:] for row in adj]
                best_deg = deg[:]
    
    # Convert to edge list
    if best_adj is None:
        # Fallback: return K7,8
        adj, deg = _make_seed_k78()
        best_adj = adj
        best_deg = deg
    
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if best_adj[u][v]:
                edges.append([int(u), int(v)])
    return edges
# EVOLVE-BLOCK-END
