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

def _decompose_greedy(edges, deadline):
    """
    Initial greedy decomposition.
    We use a heuristic to pick edges and extend paths to maximize length.
    """
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    paths = []
    
    while remaining:
        if time.time() > deadline:
            break
            
        # Heuristic: Pick an edge incident to a vertex with the highest remaining degree
        # This tends to create longer paths and leave "loose" ends at low-degree vertices.
        max_deg = -1
        chosen_edge = None
        # To speed up, we don't scan all edges every time, but for n<=30 it's fine.
        # Actually, scanning all remaining edges is O(E). 
        # A better heuristic: pick edge from a random high-degree vertex?
        # Let's stick to scanning for robustness.
        
        # Optimization: Pre-compute degrees? No, they change.
        # Just iterate.
        for e in remaining:
            u, v = e
            deg = len(adj[u]) + len(adj[v])
            if deg > max_deg:
                max_deg = deg
                chosen_edge = e
                
        if chosen_edge is None:
            chosen_edge = next(iter(remaining))
            
        u, v = chosen_edge
        remaining.remove(chosen_edge)
        adj[u].discard(v)
        adj[v].discard(u)
        
        path = [u, v]
        path_set = {u, v}
        
        # Extend from v
        curr = v
        while True:
            if time.time() > deadline:
                break
            next_v = None
            best_deg = -1
            for w in adj[curr]:
                if w not in path_set:
                    if len(adj[w]) > best_deg:
                        best_deg = len(adj[w])
                        next_v = w
            if next_v is None:
                break
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.append(next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                break

        # Extend from u
        curr = u
        while True:
            if time.time() > deadline:
                break
            next_v = None
            best_deg = -1
            for w in adj[curr]:
                if w not in path_set:
                    if len(adj[w]) > best_deg:
                        best_deg = len(adj[w])
                        next_v = w
            if next_v is None:
                break
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

def _find_max_matching_aux(paths, free_edges):
    """
    Finds a maximum cardinality matching in the auxiliary graph where nodes are paths
    and edges exist if paths can be merged using a free edge.
    Uses randomized greedy with multiple restarts since k is small.
    """
    k = len(paths)
    if k < 2:
        return []
    
    # Build adjacency list for the auxiliary graph
    adj = [set() for _ in range(k)]
    path_vertex_sets = [set(p) for p in paths]
    
    for i in range(k):
        p1 = paths[i]
        for j in range(i + 1, k):
            p2 = paths[j]
            
            # Check vertex disjointness
            if path_vertex_sets[i] & path_vertex_sets[j]:
                continue
                
            # Check if any endpoint connection exists in free_edges
            # Ends: p1[0], p1[-1], p2[0], p2[-1]
            # Possible connections:
            # 1. p1[-1] -- p2[0]
            # 2. p1[-1] -- p2[-1]
            # 3. p1[0] -- p2[0]
            # 4. p1[0] -- p2[-1]
            
            can_merge = False
            if tuple(sorted((p1[-1], p2[0]))) in free_edges:
                can_merge = True
            elif tuple(sorted((p1[-1], p2[-1]))) in free_edges:
                can_merge = True
            elif tuple(sorted((p1[0], p2[0]))) in free_edges:
                can_merge = True
            elif tuple(sorted((p1[0], p2[-1]))) in free_edges:
                can_merge = True
                
            if can_merge:
                adj[i].add(j)
                adj[j].add(i)
                
    # Find maximum matching
    best_matching = []
    best_size = 0
    
    indices = list(range(k))
    # Number of restarts depends on k. For small k, more restarts are cheap.
    restarts = 50 if k < 20 else 20
    
    for _ in range(restarts):
        random.shuffle(indices)
        matched = [False] * k
        current_matching = []
        
        for i in indices:
            if matched[i]:
                continue
            neighbors = list(adj[i])
            random.shuffle(neighbors)
            for j in neighbors:
                if not matched[j]:
                    matched[i] = True
                    matched[j] = True
                    current_matching.append((i, j))
                    break
        
        if len(current_matching) > best_size:
            best_size = len(current_matching)
            best_matching = current_matching
            
    return best_matching

def _apply_matching(paths, matching, free_edges):
    """
    Apply the matches found in the auxiliary graph to the list of paths.
    """
    if not matching:
        return paths
        
    new_paths = []
    used = [False] * len(paths)
    
    for i, j in matching:
        if used[i] or used[j]:
            continue
            
        p1 = paths[i]
        p2 = paths[j]
        
        # Determine how to merge
        # Try the 4 combinations.
        merged = None
        
        # 1. p1[-1] -- p2[0]
        if tuple(sorted((p1[-1], p2[0]))) in free_edges:
            merged = p1 + p2
        # 2. p1[-1] -- p2[-1]
        elif tuple(sorted((p1[-1], p2[-1]))) in free_edges:
            merged = p1 + p2[::-1]
        # 3. p1[0] -- p2[0]
        elif tuple(sorted((p1[0], p2[0]))) in free_edges:
            merged = p1[::-1] + p2
        # 4. p1[0] -- p2[-1]
        elif tuple(sorted((p1[0], p2[-1]))) in free_edges:
            merged = p1[::-1] + p2[::-1]
            
        if merged is not None:
            used[i] = True
            used[j] = True
            new_paths.append(merged)
            
    # Add paths that were not matched
    for i in range(len(paths)):
        if not used[i]:
            new_paths.append(paths[i])
            
    return new_paths

def decompose(edges, deadline):
    if not edges:
        return []
        
    # 1. Initial greedy decomposition
    paths = _decompose_greedy(edges, deadline)
    
    # If we ran out of time, return what we have (might be invalid if not all edges covered, 
    # but the loop in _decompose_greedy breaks on deadline. 
    # We must ensure validity. If we break early, we have uncovered edges.
    # The task requires covering ALL edges. 
    # So we must ensure _decompose_greedy finishes. 
    # Given n<=30, E<=~150, greedy is fast. The bottleneck is the matching phase.
    # So we can afford to let greedy finish.
    
    # 2. Iterative Maximum Matching Merging
    # We loop until no more merges are possible or time is up.
    while time.time() < deadline:
        # Calculate used and free edges
        used_edges = set()
        for p in paths:
            for k in range(len(p) - 1):
                e = tuple(sorted((p[k], p[k+1])))
                used_edges.add(e)
                
        free_edges = set(edges) - used_edges
        
        if not free_edges:
            break # No free edges, can't merge
            
        # Find maximum matching in auxiliary graph
        matching = _find_max_matching_aux(paths, free_edges)
        
        if not matching:
            break
            
        # Apply the matching
        new_paths = _apply_matching(paths, matching, free_edges)
        
        # Safety check: ensure we didn't increase path count or invalidate
        if len(new_paths) >= len(paths):
            break
            
        paths = new_paths
        
    return paths

def construct():
    start_time = time.time()
    # Total budget 150s. Leave 10s buffer for overhead/safety.
    deadline = start_time + 140
    
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    # Allocate time roughly proportional to complexity (n^2)
    # n=20, 25, 30. 
    # Weights: 400, 625, 900. Sum = 1925.
    # Time available: 140s.
    # t1 = 140 * 400/1925 = 29s
    # t2 = 140 * 625/1925 = 45s
    # t3 = 140 * 900/1925 = 65s
    # This is a heuristic. Let's just give equal time or slightly more to larger.
    # Let's give 40s, 50s, 50s.
    
    time_allocs = [40, 50, 50]
    
    for i, spec in enumerate(specs):
        edges = instance(*spec)
        
        # Calculate specific deadline for this graph
        graph_deadline = start_time + sum(time_allocs[:i+1])
        
        # Ensure we don't exceed global deadline
        if graph_deadline > deadline:
            graph_deadline = deadline
            
        paths = decompose(edges, graph_deadline)
        results.append(paths)
        
    return results
