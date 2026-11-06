"""Path decomposition of three fixed connected graphs (Gallai's conjecture)."""
import time
import random
from collections import defaultdict, deque

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _get_odd_vertices(edges):
    deg = defaultdict(int)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return [v for v, d in deg.items() if d % 2 != 0]

def _beam_search_longest_path(n, remaining_edges, start_node, beam_width=100, max_depth=100):
    """
    Finds a long simple path starting from start_node using beam search.
    Returns the path as a list of vertices.
    """
    if not remaining_edges:
        return []
    
    # Build adjacency for remaining edges
    rem_adj = defaultdict(set)
    for u, v in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
        
    if start_node not in rem_adj:
        return []

    # Beam: list of (path_list, score)
    # Score can be length, or length + heuristic
    beam = [([start_node], 0)]
    
    for _ in range(max_depth):
        next_beam = []
        for path, score in beam:
            current = path[-1]
            visited = set(path)
            
            neighbors = [nb for nb in rem_adj[current] if nb not in visited]
            
            if not neighbors:
                # Dead end, keep this path as a candidate
                next_beam.append((path, score))
                continue
            
            # Generate extensions
            for nb in neighbors:
                new_path = path + [nb]
                # Heuristic: prefer nodes with higher remaining degree?
                # Or just length.
                new_score = len(new_path)
                next_beam.append((new_path, new_score))
                
        if not next_beam:
            break
            
        # Sort by score descending and keep top beam_width
        next_beam.sort(key=lambda x: x[1], reverse=True)
        beam = next_beam[:beam_width]
        
        # If all paths in beam are dead ends (no extensions possible in next step), 
        # we might want to stop if we can't improve.
        # But for now, just continue until beam is empty or max_depth.
        # Optimization: if all paths in beam have the same length and no extensions, break.
        # Simple check: if the best path in next_beam has no neighbors, we might be done.
        # Actually, if we can't extend any path, we should return the best one.
        
        # Check if any path can be extended
        can_extend = False
        for path, _ in beam:
            current = path[-1]
            visited = set(path)
            if any(nb not in visited for nb in rem_adj[current]):
                can_extend = True
                break
        if not can_extend:
            break
            
    if not beam:
        return []
        
    # Return the longest path
    best_path = max(beam, key=lambda x: len(x[0]))[0]
    return best_path

def _solve_graph(n, p, seed, time_budget=30.0):
    edges = _instance(n, p, seed)
    m = len(edges)
    if m == 0:
        return []
    
    odd_verts = _get_odd_vertices(edges)
    lower_bound = max(1, len(odd_verts) // 2)
    
    best_paths = None
    best_count = float('inf')
    
    start_time = time.time()
    rng = random.Random(seed + 100)
    
    # Strategy: 
    # 1. Try to build paths starting from odd vertices to satisfy the lower bound.
    # 2. Use beam search to find long paths.
    
    for attempt in range(50): # Limit attempts to ensure time safety
        if time.time() - start_time > time_budget:
            break
            
        remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
        paths = []
        
        # Sort odd vertices to try to connect them
        # We will greedily pick a start node, find a long path, remove edges.
        # To encourage covering odd vertices as endpoints, we can prioritize them.
        
        # Simple heuristic: 
        # While edges remain:
        #   Pick a start node. 
        #   If odd vertices remain, pick one. Else pick random high degree.
        #   Find longest path from start using beam search.
        #   Add to paths.
        
        rem_edges = set(remaining_edges)
        
        while rem_edges:
            # Determine start node
            # Calculate degrees in rem_edges
            rem_deg = defaultdict(int)
            for u, v in rem_edges:
                rem_deg[u] += 1
                rem_deg[v] += 1
                
            # Find odd degree vertices in rem_edges
            current_odd = [v for v, d in rem_deg.items() if d % 2 != 0]
            
            if current_odd:
                # Pick a random odd vertex
                start_node = rng.choice(current_odd)
            else:
                # Pick a random vertex with highest degree
                if rem_deg:
                    max_deg = max(rem_deg.values())
                    candidates = [v for v, d in rem_deg.items() if d == max_deg]
                    start_node = rng.choice(candidates)
                else:
                    break
                    
            # Find path
            path = _beam_search_longest_path(n, rem_edges, start_node, beam_width=50, max_depth=n)
            
            if not path or len(path) < 2:
                # Fallback: take a single edge
                # Find any edge incident to start_node or just any edge
                if rem_edges:
                    # Prefer edge incident to start_node
                    incident = [(u, v) for u, v in rem_edges if u == start_node or v == start_node]
                    if incident:
                        u, v = rng.choice(incident)
                        path = [u, v]
                    else:
                        u, v = rng.choice(list(rem_edges))
                        path = [u, v]
                else:
                    break
            
            # Add path
            paths.append(path)
            
            # Remove edges from path
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                e = (min(u, v), max(u, v))
                rem_edges.discard(e)
                
        count = len(paths)
        if count < best_count:
            best_count = count
            best_paths = [list(p) for p in paths]
            
        if best_count == lower_bound:
            break
            
    if best_paths is None:
        # Fallback: single edges
        best_paths = [[u, v] for (u, v) in edges]
        
    return best_paths

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            # Not enough time left, use fallback
            edges = _instance(*spec)
            results.append([[u, v] for (u, v) in edges])
        else:
            # Allocate time budget
            remaining_time = deadline - elapsed - 10
            time_budget = max(1.0, min(40.0, remaining_time / 3))
            paths = _solve_graph(*spec, time_budget=time_budget)
            results.append(paths)
    
    return results
