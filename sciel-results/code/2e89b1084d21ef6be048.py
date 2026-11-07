"""Structural approach: target degree sequences with distinct degrees, fast validity via degree groups."""
# EVOLVE-BLOCK-START
import time
import random

def _fast_check_validity(adj, deg, n, edges):
    """
    Fast validity check: for each edge (u,v), check that no s in N(u)\{v} and t in N(v)\{u}
    with s!=t have deg[s]==deg[t].
    Group by degree to speed up.
    """
    # Build degree groups: list of vertices for each degree value
    deg_groups = {}
    for i in range(n):
        d = deg[i]
        if d not in deg_groups:
            deg_groups[d] = set()
        deg_groups[d].add(i)
    
    for (u, v) in edges:
        # For each s in N(u) \ {v}, for each t in N(v) \ {u}, s != t:
        # need deg[s] != deg[t]
        # Equivalent: for each degree d, count how many s in N(u)\{v} have deg[s]==d,
        # and how many t in N(v)\{u} have deg[t]==d.
        # If both counts >= 1 and (count_s + count_t >= 2 or (count_s >= 1 and count_t >= 1 and not (the single s == the single t))),
        # actually simpler: if there exists s in N(u)\{v} and t in N(v)\{u} with s!=t and deg[s]==deg[t], invalid.
        # So: for each degree d, let S_d = {s in N(u)\{v} : deg[s]==d}, T_d = {t in N(v)\{u} : deg[t]==d}.
        # Invalid if |S_d| >= 1 and |T_d| >= 1 and (|S_d| + |T_d| >= 2 or (|S_d|==1 and |T_d|==1 and the elements differ)).
        # Actually: invalid iff there exist s in S_d, t in T_d with s != t.
        # This is true iff |S_d| >= 1 and |T_d| >= 1 and (S_d != T_d or |S_d| > 1 or |T_d| > 1).
        # Simpler: just iterate.
        nu = adj[u]
        nv = adj[v]
        # Collect degrees of neighbors
        for s in nu:
            if s == v:
                continue
            ds = deg[s]
            for t in nv:
                if t == u:
                    continue
                if s == t:
                    continue
                if deg[t] == ds:
                    return False
    return True

def _check_validity(adj, deg, n, edges):
    """Check no 3-edge path s-u-v-t (s!=t) has deg(s)==deg(t)."""
    return _fast_check_validity(adj, deg, n, edges)

def _try_add_edge(adj, deg, n, edges, edge_set, u, v):
    """Try to add edge (u,v). Returns True if added, False if invalid."""
    new_edges = edges + [[u, v]]
    new_adj = [set() for _ in range(n)]
    new_deg = [0] * n
    for (a, b) in new_edges:
        new_adj[a].add(b)
        new_adj[b].add(a)
        new_deg[a] += 1
        new_deg[b] += 1
    
    if _check_validity(new_adj, new_deg, n, new_edges):
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        edges.append([u, v])
        edge_set.add((u, v))
        return True
    return False

def _try_remove_edge(adj, deg, n, edges, edge_set, u, v):
    """Remove edge (u,v)."""
    adj[u].discard(v)
    adj[v].discard(u)
    deg[u] -= 1
    deg[v] -= 1
    edges.remove([u, v])
    edge_set.discard((u, v))

def construct():
    t0 = time.time()
    deadline = t0 + 85.0  # Safety margin before 90s kill
    n = 15
    
    best_edges = []
    best_count = 0
    
    # Strategy: simulated annealing on edge set
    # State: set of edges. Move: add/remove/swap an edge.
    # Objective: maximize edge count subject to validity.
    
    # Start with a good initial graph
    # Try multiple initial constructions
    
    for restart in range(200):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(restart * 104729)
        
        # Initialize: start with a random subset of edges, then greedily add
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        edge_set = set()
        
        # Greedy initialization: try all edges in random order, add if valid
        candidates = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(candidates)
        
        for (u, v) in candidates:
            if time.time() > deadline - 5:
                break
            if (u, v) not in edge_set:
                _try_add_edge(adj, deg, n, edges, edge_set, u, v)
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [list(e) for e in edges]
        
        # Simulated annealing: try to improve by adding/removing/swapping edges
        # Temperature schedule
        T = 1.0
        cooling = 0.995
        min_T = 0.01
        
        current_edges = [list(e) for e in edges]
        current_count = len(edges)
        
        # Build current adj/deg for fast evaluation
        cur_adj = [set() for _ in range(n)]
        cur_deg = [0] * n
        cur_edge_set = set()
        for (a, b) in current_edges:
            cur_adj[a].add(b)
            cur_adj[b].add(a)
            cur_deg[a] += 1
            cur_deg[b] += 1
            cur_edge_set.add((a, b))
        
        # All possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        
        steps = 0
        max_steps = 5000
        while T > min_T and steps < max_steps:
            if time.time() > deadline - 5:
                break
            
            steps += 1
            T *= cooling
            
            # Choose a random move
            move_type = rng.random()
            if move_type < 0.4:
                # Try to add a random missing edge
                missing = [(u, v) for (u, v) in all_edges if (u, v) not in cur_edge_set]
                if not missing:
                    continue
                u, v = rng.choice(missing)
                
                # Test: add edge
                test_adj = [set(s) for s in cur_adj]
                test_deg = list(cur_deg)
                test_edges = current_edges + [[u, v]]
                test_edge_set = set(cur_edge_set) | {(u, v)}
                
                test_adj[u].add(v)
                test_adj[v].add(u)
                test_deg[u] += 1
                test_deg[v] += 1
                
                if _check_validity(test_adj, test_deg, n, test_edges):
                    # Accept
                    current_edges = test_edges
                    current_count += 1
                    cur_adj = test_adj
                    cur_deg = test_deg
                    cur_edge_set = test_edge_set
                    if current_count > best_count:
                        best_count = current_count
                        best_edges = [list(e) for e in current_edges]
                # If invalid, reject
            elif move_type < 0.7:
                # Try to remove a random edge
                if not current_edges:
                    continue
                idx = rng.randrange(len(current_edges))
                u, v = current_edges[idx]
                
                test_adj = [set(s) for s in cur_adj]
                test_deg = list(cur_deg)
                test_edges = [e for i, e in enumerate(current_edges) if i != idx]
                test_edge_set = set(cur_edge_set) - {(u, v)}
                
                test_adj[u].discard(v)
                test_adj[v].discard(u)
                test_deg[u] -= 1
                test_deg[v] -= 1
                
                # Removal is always valid (fewer constraints)
                current_edges = test_edges
                current_count -= 1
                cur_adj = test_adj
                cur_deg = test_deg
                cur_edge_set = test_edge_set
            else:
                # Swap: remove one edge, add another
                if not current_edges:
                    continue
                missing = [(u, v) for (u, v) in all_edges if (u, v) not in cur_edge_set]
                if not missing:
                    continue
                
                idx = rng.randrange(len(current_edges))
                u1, v1 = current_edges[idx]
                u2, v2 = rng.choice(missing)
                
                test_adj = [set(s) for s in cur_adj]
                test_deg = list(cur_deg)
                test_edges = [e for i, e in enumerate(current_edges) if i != idx] + [[u2, v2]]
                test_edge_set = set(cur_edge_set) - {(u1, v1)} | {(u2, v2)}
                
                test_adj[u1].discard(v1)
                test_adj[v1].discard(u1)
                test_deg[u1] -= 1
                test_deg[v1] -= 1
                test_adj[u2].add(v2)
                test_adj[v2].add(u2)
                test_deg[u2] += 1
                test_deg[v2] += 1
                
                if _check_validity(test_adj, test_deg, n, test_edges):
                    current_edges = test_edges
                    cur_adj = test_adj
                    cur_deg = test_deg
                    cur_edge_set = test_edge_set
                    if current_count > best_count:
                        best_count = current_count
                        best_edges = [list(e) for e in current_edges]
    
    # Final validation
    if best_edges:
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for (u, v) in best_edges:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        if not _check_validity(adj, deg, n, best_edges):
            best_edges = _fallback_valid(n)
    
    result = [[int(u), int(v)] for (u, v) in best_edges]
    return result

def _fallback_valid(n):
    """Return a simple valid graph as fallback."""
    edges = []
    for v in range(1, n):
        edges.append([0, v])
    return edges
# EVOLVE-BLOCK-END
