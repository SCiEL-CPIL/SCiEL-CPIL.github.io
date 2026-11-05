import time
import random
import sys
from collections import defaultdict

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_long_cycle(adj, n, rng, max_restarts=10):
    """
    Find a long simple cycle in the graph represented by adj.
    Uses a randomized greedy strategy with scoring to prefer high-degree unvisited nodes.
    """
    best_cycle = None
    best_len = 0

    # Try starting from each vertex
    # To save time, we can sample starts if n is large, but n is small (20-30)
    starts = list(range(n))
    rng.shuffle(starts)
    
    for start in starts:
        if not adj[start]:
            continue

        for _ in range(max_restarts):
            path = [start]
            visited = {start}
            current = start

            while True:
                neighbors = list(adj[current])
                unvisited = [v for v in neighbors if v not in visited]
                
                # Check if we can close the cycle to start
                if start in adj[current] and len(path) >= 3:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                    # If we found a very long cycle, we might stop early? 
                    # No, let's keep searching for the absolute best in this restart.
                    # But usually, closing as soon as possible after reaching a decent length 
                    # is a good heuristic to avoid dead ends.
                    # Let's close with some probability if path is long enough.
                    if len(path) > 5 and rng.random() < 0.2:
                        break
                    # Otherwise, if no other unvisited neighbors, we must close or break
                    if not unvisited:
                        break

                if not unvisited:
                    # Dead end, cannot extend, and couldn't close (or chose not to)
                    break

                # Score unvisited neighbors
                # Heuristic: prefer neighbors with more unvisited neighbors (greedy expansion)
                scored = []
                for v in unvisited:
                    # Count how many neighbors of v are not visited
                    # This is O(deg(v))
                    unvisited_count = sum(1 for w in adj[v] if w not in visited)
                    # Add a small random noise to break ties and allow exploration
                    score = unvisited_count + rng.random() * 0.5
                    scored.append((score, v))
                
                scored.sort(key=lambda x: x[0], reverse=True)
                
                # Pick the best, or random from top 2 to maintain some randomness
                if rng.random() < 0.8:
                    next_node = scored[0][1]
                else:
                    # Pick from top 2
                    if len(scored) > 1:
                        next_node = scored[1][1]
                    else:
                        next_node = scored[0][1]
                
                path.append(next_node)
                visited.add(next_node)
                current = next_node

    return best_cycle

def _decompose_once(edges, n, rng, deadline):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    remaining = len(edges)
    
    while remaining > 0:
        if time.time() > deadline:
            break
        
        # Find a long cycle using greedy randomized strategy
        best_cycle = _find_long_cycle(adj, n, rng, max_restarts=8)
        
        if best_cycle:
            m = len(best_cycle)
            for i in range(m):
                a = best_cycle[i]
                b = best_cycle[(i + 1) % m]
                adj[a].discard(b)
                adj[b].discard(a)
            remaining -= m
            pieces.append(best_cycle)
        else:
            # No cycle found, remove a single edge
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
    
    base_seed = 42
    attempts = 0
    max_attempts = 200 # Cap to prevent infinite loop if time is not checked
    
    while time.time() < deadline - 0.5:
        if attempts >= max_attempts:
            break
        rng = random.Random(base_seed + attempts)
        pieces = _decompose_once(edges, n, rng, deadline)
        cnt = len(pieces)
        
        if cnt < best_count:
            best_count = cnt
            best = pieces
            
        attempts += 1
    
    if best is None:
        # Fallback: just edges
        best = [[u, v] for u, v in edges]
        
    return best

def construct():
    start_time = time.time()
    total_budget = 145.0
    per_graph_budget = total_budget / 3.0
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        
        # Allocate slightly more time to later graphs if earlier ones are fast?
        # Or just fixed split. Fixed split is safer.
        graph_deadline = start_time + per_graph_budget * (i + 1)
        
        # Safety check: don't exceed total budget
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0
            
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        
        # If we are running out of time, fill remaining with trivial solution
        if time.time() > start_time + total_budget - 3.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
    
    return results
