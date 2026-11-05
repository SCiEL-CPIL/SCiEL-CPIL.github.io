import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_long_cycle(adj, n, max_attempts=20, deadline=None):
    """
    Find a long simple cycle in the graph represented by adj.
    Uses a greedy randomized strategy with degree-based heuristics.
    Returns the cycle as a list of vertices or None.
    """
    best_cycle = None
    best_len = 0
    
    # Try starting from each vertex, prioritizing high-degree vertices
    candidates = [v for v in range(n) if adj[v]]
    if not candidates:
        return None
    
    # Sort by degree descending, but add some randomness
    candidates.sort(key=lambda v: len(adj[v]), reverse=True)
    # Shuffle top candidates to add diversity
    rng_local = random.Random()
    
    for start in candidates:
        if deadline and time.time() > deadline:
            break
        if not adj[start]:
            continue
            
        # Run multiple random attempts from this start
        for attempt in range(max_attempts):
            if deadline and time.time() > deadline:
                break
                
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
                    # Use degree-based heuristic: prefer neighbors with higher degree
                    # to keep more options open
                    scored = [(len(adj[v]), v) for v in unvisited_neighbors]
                    scored.sort(reverse=True)
                    
                    # With some probability, pick randomly for diversity
                    if rng_local.random() < 0.3:
                        next_node = rng_local.choice(unvisited_neighbors)
                    else:
                        # Pick from top 3 highest-degree neighbors
                        top_k = scored[:min(3, len(scored))]
                        next_node = rng_local.choice([v for _, v in top_k])
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node

    return best_cycle

def _decompose(edges, n, deadline, rnd=None):
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
        cyc = _find_long_cycle(adj, n, max_attempts=10, deadline=deadline)
        
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

def _local_decompose(edges, n, deadline):
    """
    Decompose a subset of edges.
    """
    return _decompose(edges, n, deadline)

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    
    # Use multiple random seeds for the internal randomization
    num_attempts = 100 # Increased attempts
    base_seed = 42
    
    for i in range(num_attempts):
        if time.time() > deadline - 1.0:
            break
            
        random.seed(base_seed + i)
        
        # 1. Initial Greedy
        pieces = _decompose(edges, n, deadline, None)
        cnt = len(pieces)
        
        if cnt < best_count:
            best_count = cnt
            best = pieces
            
        # 2. Local Search (Iterated Local Search)
        # Try to improve by splicing cycles and re-decomposing local parts
        # We only do this if we have time and if the solution has cycles
        if time.time() < deadline - 5.0 and cnt > 2:
            local_deadline = min(deadline - 2.0, time.time() + 2.0)
            current_pieces = pieces[:]
            current_count = cnt
            
            # Map piece index to set of edges for quick lookup
            # And map vertex to pieces containing it
            
            for ls_iter in range(50):
                if time.time() > local_deadline:
                    break
                
                # Find cycles in current_pieces
                cycles = [p for p in current_pieces if len(p) >= 3]
                if len(cycles) < 2:
                    break
                
                # Pick 2 random cycles
                c1_idx, c2_idx = random.sample(range(len(cycles)), 2)
                c1 = cycles[c1_idx]
                c2 = cycles[c2_idx]
                
                # Check if they share at least 2 vertices
                v1 = set(c1)
                v2 = set(c2)
                shared = v1 & v2
                if len(shared) < 2:
                    continue
                
                # Extract edges of c1 and c2
                e1 = set()
                for i in range(len(c1)):
                    u, v = c1[i], c1[(i+1)%len(c1)]
                    e1.add(tuple(sorted((u, v))))
                
                e2 = set()
                for i in range(len(c2)):
                    u, v = c2[i], c2[(i+1)%len(c2)]
                    e2.add(tuple(sorted((u, v))))
                
                # The local subgraph is e1 | e2
                local_edges = list(e1 | e2)
                
                # Re-decompose local_edges
                new_local_pieces = _decompose(local_edges, n, local_deadline)
                
                # Construct new full solution
                # Remove c1 and c2 from current_pieces
                # Note: c1 and c2 are references to the lists in current_pieces
                # We need to find their indices in current_pieces
                
                # Since current_pieces contains both cycles and edges,
                # and c1, c2 are specific lists, we can remove them by value
                # But lists are not hashable. We use identity or index.
                # Easier: Build a new list.
                
                new_pieces = []
                for p in current_pieces:
                    if p is c1 or p is c2:
                        continue
                    new_pieces.append(p)
                
                new_pieces.extend(new_local_pieces)
                
                new_count = len(new_pieces)
                
                # Accept if better, or with some probability (annealing)
                # Here we are just hill climbing on piece count
                if new_count < current_count:
                    current_pieces = new_pieces
                    current_count = new_count
                    if current_count < best_count:
                        best_count = current_count
                        best = current_pieces[:]
                elif new_count == current_count and random.random() < 0.1:
                    # Accept neutral move to escape local minima
                    current_pieces = new_pieces
                    current_count = new_count
                # If worse, reject

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
