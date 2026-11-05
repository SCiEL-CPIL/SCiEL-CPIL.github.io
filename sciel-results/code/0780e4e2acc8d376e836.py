import time
import random
import sys
import heapq

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_long_cycle(adj, n, rng, max_attempts=50):
    """
    Find a long simple cycle in the graph represented by adj.
    Uses a randomized greedy strategy: extend path as far as possible.
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
                    # Heuristic: Pick a neighbor that is not the start (to keep extending)
                    # If start is in unvisited_neighbors, we ignore it for now unless it's the only option.
                    # But start is in visited, so it's not in unvisited_neighbors.
                    # So we just pick a random unvisited neighbor.
                    
                    # Better heuristic: pick the neighbor with the fewest unvisited neighbors (Warnsdorff-like)
                    # to try to force a longer path or a cycle.
                    def score(v):
                        return sum(1 for w in adj[v] if w not in visited)
                    
                    # Sort by score, with some randomness
                    unvisited_neighbors.sort(key=lambda v: (score(v), rng.random()))
                    next_node = unvisited_neighbors[0]
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node

    return best_cycle

def _decompose_greedy(adj, n, deadline, rng):
    """
    Greedily extract cycles from the graph.
    Returns a list of pieces (cycles or edges).
    """
    pieces = []
    remaining = sum(len(s) for s in adj) // 2
    
    while remaining > 0:
        if time.time() > deadline:
            break
            
        # Find a long cycle
        cyc = _find_long_cycle(adj, n, rng, max_attempts=20)
        
        if cyc:
            # Remove cycle edges
            m = len(cyc)
            for i in range(m):
                a = cyc[i]
                b = cyc[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(cyc)
        else:
            # No cycle found, remove an edge
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

def _decompose_flow_based(edges, n, deadline):
    """
    Decompose edges into cycles and edges using a flow-based approach to find a 2-factor first.
    """
    # For now, fall back to greedy. A full flow-based implementation is complex.
    adj = _build_adj(edges, n)
    rng = random.Random(12345)
    return _decompose_greedy(adj, n, deadline, rng)

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    
    # Use multiple random seeds for the internal randomization
    num_attempts = 30 # Adjust based on time
    base_seed = 42
    
    for i in range(num_attempts):
        if time.time() > deadline - 1.0:
            break
            
        rng = random.Random(base_seed + i)
        adj = _build_adj(edges, n)
        pieces = _decompose_greedy(adj, n, deadline, rng)
        cnt = len(pieces)
        
        if cnt < best_count:
            best_count = cnt
            best = pieces
            
        # If we found a very good solution, stop early?
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
