import time
import random
import sys

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_long_cycle(adj, n, max_attempts=20):
    """
    Find a long simple cycle in the graph represented by adj.
    Uses a greedy randomized strategy: extend path as far as possible, 
    only close if no other options.
    Returns the cycle as a list of vertices or None.
    """
    best_cycle = None
    best_len = 0
    
    # Try starting from each vertex
    for start in range(n):
        if not adj[start]:
            continue
            
        # Run multiple random attempts from this start
        for _ in range(max_attempts):
            path = [start]
            visited = {start}
            current = start
            
            while True:
                neighbors = list(adj[current])
                # Filter out visited neighbors
                unvisited_neighbors = [v for v in neighbors if v not in visited]
                
                # Check if we can close the cycle
                can_close = start in neighbors and len(path) >= 3
                
                if not unvisited_neighbors:
                    # Stuck. Can we close?
                    if can_close:
                        # Found a cycle
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    break
                else:
                    # We have unvisited neighbors.
                    # Do we want to close now?
                    # Only if we have no other choice? 
                    # The strategy is to extend as much as possible.
                    # So we only close if there are NO unvisited neighbors.
                    # This is handled above.
                    # So we must pick an unvisited neighbor.
                    
                    # Heuristic: Pick a neighbor that is not the start (to keep extending)
                    # If start is in unvisited_neighbors, we ignore it for now 
                    # unless it's the only option (which is handled by the 
                    # "no unvisited" case if we consider start as visited? 
                    # Wait, start is in visited. So start is NOT in unvisited_neighbors.
                    # So we will never pick start until we are stuck?
                    # No, if we are stuck, we check if start is a neighbor.
                    
                    # So we just pick a random unvisited neighbor.
                    next_node = random.choice(unvisited_neighbors)
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node

    return best_cycle

def _decompose(edges, n, deadline, rnd):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    pieces = []
    remaining = len(edges)
    
    while remaining > 0:
        if time.time() > deadline:
            break
            
        # Find a long cycle
        cyc = _find_long_cycle(adj, n, max_attempts=10)
        
        if cyc:
            # Remove cycle edges
            m = len(cyc)
            # The cycle is [v0, v1, ..., v_{m-1}]
            # Edges are (v0, v1), (v1, v2), ..., (v_{m-1}, v0)
            for i in range(m):
                a = cyc[i]
                b = cyc[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(cyc)
        else:
            # No cycle found, remove an edge
            # Find an edge
            edge_removed = False
            for u in range(n):
                for v in list(adj[u]):
                    if u < v:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        remaining -= 1
                        pieces.append([u, v])
                        edge_removed = True
                        break
                if edge_removed:
                    break
            if not edge_removed:
                break

    return pieces

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    
    # Use multiple random seeds for the internal randomization
    # The _find_long_cycle uses global random or we should pass a seed?
    # Currently it uses global random. We should make it deterministic per attempt 
    # or use a local RNG.
    # Let's modify _find_long_cycle to accept a random instance.
    
    # Actually, let's just re-run the whole decomposition with different 
    # global random states or just rely on the non-determinism of 
    # random.choice if we seed it differently outside?
    # Better: Pass a random object to _find_long_cycle.
    
    # Let's refactor _find_long_cycle to take a rng.
    
    # Wait, I can't easily refactor in the thought process, let's just 
    # use a local random instance in _decompose and pass it down?
    # Or just seed the global random differently for each attempt.
    
    # Since _find_long_cycle uses `random.choice`, it uses the global RNG.
    # We can seed it before each attempt.
    
    num_attempts = 50 # Adjust based on time
    base_seed = 42
    
    for i in range(num_attempts):
        if time.time() > deadline - 1.0:
            break
            
        random.seed(base_seed + i)
        pieces = _decompose(edges, n, deadline, None)
        cnt = len(pieces)
        
        if cnt < best_count:
            best_count = cnt
            best = pieces
            
        # If we found a very good solution (e.g., few edges left as singles), stop early?
        # Hard to know the optimum.
        
    return best

def construct():
    start_time = time.time()
    total_budget = 140.0 # Safety margin for 150s limit
    per_graph_budget = total_budget / 3.0
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        # Allocate time for this graph, ensuring we don't exceed total budget
        graph_deadline = start_time + per_graph_budget * (i + 1)
        
        # Ensure we leave at least 2 seconds for the rest
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0
            
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        
        # If we are running out of time, fill remaining with trivial solutions
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
            
    return results
