import time
import random
from collections import defaultdict

def instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _degree(v, edges):
    deg = 0
    for u, w in edges:
        if u == v or w == v:
            deg += 1
    return deg

def _greedy_decompose_random(edges, rng):
    """
    Randomized greedy path decomposition.
    """
    if not edges:
        return []
    
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    paths = []
    
    while remaining:
        # Pick a starting edge
        # Heuristic: Pick an edge incident to a vertex with high degree, but randomize among them
        # Or just pick a random edge?
        # Let's pick a random edge to ensure diversity.
        e = rng.choice(list(remaining))
        u, v = e
        
        remaining.remove(e)
        adj[u].discard(v)
        adj[v].discard(u)
        
        path = [u, v]
        path_set = {u, v}
        
        # Extend from v
        curr = v
        while True:
            neighbors = list(adj[curr])
            # Filter valid neighbors
            valid = [w for w in neighbors if w not in path_set]
            if not valid:
                break
            
            next_v = rng.choice(valid)
            
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.append(next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                # This shouldn't happen if adj is maintained correctly, but just in case
                break

        # Extend from u
        curr = u
        while True:
            neighbors = list(adj[curr])
            valid = [w for w in neighbors if w not in path_set]
            if not valid:
                break
            
            next_v = rng.choice(valid)
            
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.insert(0, next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                break

        paths.append(path)
        
    return paths

def _local_search_merge(paths, edges):
    """
    Iteratively try to merge paths using free edges.
    """
    # Build set of used edges
    used_edges = set()
    for p in paths:
        for k in range(len(p) - 1):
            e = tuple(sorted((p[k], p[k+1])))
            used_edges.add(e)
            
    free_edges = set(edges) - used_edges
    
    changed = True
    while changed:
        changed = False
        n_paths = len(paths)
        for i in range(n_paths):
            if changed:
                # Re-evaluate n_paths if list size changed? 
                # We break out of inner loop on change, so we restart outer loop.
                break
            for j in range(i + 1, n_paths):
                p1 = paths[i]
                p2 = paths[j]
                
                set1 = set(p1)
                set2 = set(p2)
                if set1 & set2:
                    continue
                    
                # Try 4 connections
                merged = False
                new_path = None
                
                # 1. p1[-1] -- p2[0]
                if tuple(sorted((p1[-1], p2[0]))) in free_edges:
                    new_path = p1 + p2
                    merged = True
                # 2. p1[-1] -- p2[-1]
                elif tuple(sorted((p1[-1], p2[-1]))) in free_edges:
                    new_path = p1 + p2[::-1]
                    merged = True
                # 3. p1[0] -- p2[0]
                elif tuple(sorted((p1[0], p2[0]))) in free_edges:
                    new_path = p1[::-1] + p2
                    merged = True
                # 4. p1[0] -- p2[-1]
                elif tuple(sorted((p1[0], p2[-1]))) in free_edges:
                    new_path = p1[::-1] + p2[::-1]
                    merged = True
                
                if merged:
                    paths[i] = new_path
                    paths.pop(j)
                    changed = True
                    # Update free edges? 
                    # The edge used for merging is now "used".
                    # We don't strictly need to update free_edges immediately if we restart the loop,
                    # but it's safer to recalculate or update.
                    # Since we break and restart the while loop, we will recalculate used_edges?
                    # No, we don't recalculate in the while loop body.
                    # Let's recalculate used_edges and free_edges after a merge.
                    used_edges = set()
                    for p in paths:
                        for k in range(len(p) - 1):
                            e = tuple(sorted((p[k], p[k+1])))
                            used_edges.add(e)
                    free_edges = set(edges) - used_edges
                    break # Break inner loop, restart outer while
    
    return paths

def decompose(edges):
    if not edges:
        return []
    
    # Calculate lower bound
    vertices = set()
    for u, v in edges:
        vertices.add(u)
        vertices.add(v)
    odd_count = 0
    for v in vertices:
        if _degree(v, edges) % 2 != 0:
            odd_count += 1
    lb = max(1, odd_count // 2)
    
    best_paths = None
    best_len = float('inf')
    
    start_time = time.time()
    # Allocate time based on graph size? 
    # N=20, 25, 30. 
    # Let's use 40s per graph.
    deadline = start_time + 40.0
    
    # Run multiple restarts
    max_attempts = 500
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        rng = random.Random(attempt)
        paths = _greedy_decompose_random(edges, rng)
        paths = _local_search_merge(paths, edges)
        
        if len(paths) < best_len:
            best_len = len(paths)
            best_paths = paths
            
        if best_len <= lb:
            break
            
    return best_paths

def construct():
    deadline = time.time() + 145
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        
        # Adjust deadline for remaining graphs
        # If we have 3 graphs, and we are at index i, we have (2-i) graphs left.
        # Current graph should get (145 - elapsed) / (3 - i) seconds?
        # Simpler: fixed 45s per graph, but check global deadline.
        graph_deadline = min(time.time() + 45, deadline)
        
        # Temporarily override decompose's internal deadline?
        # decompose has its own 40s deadline. 
        # We should pass deadline or make it flexible.
        # For now, 40s * 3 = 120s < 145s. Safe.
        
        paths = decompose(edges)
        results.append(paths)
        
        if time.time() > deadline:
             # Should not happen, but break if it does
             pass
             
    return results
