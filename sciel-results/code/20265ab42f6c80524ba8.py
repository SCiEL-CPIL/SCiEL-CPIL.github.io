import time
from collections import defaultdict
import random

def instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _greedy_decompose(edges):
    """
    Initial greedy decomposition to get a valid starting point.
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
        # Pick an edge. To improve initial quality, maybe pick high degree?
        # For now, just pick first to ensure speed.
        e = next(iter(remaining))
        u, v = e
        
        remaining.remove(e)
        adj[u].discard(v)
        adj[v].discard(u)
        
        path = [u, v]
        path_set = {u, v}
        
        # Extend from v
        curr = v
        while True:
            next_v = None
            # Try to extend to a vertex with high degree to keep options open?
            # Or just any. Let's stick to simple for now.
            for w in adj[curr]:
                if w not in path_set:
                    next_v = w
                    break
            
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
            next_v = None
            for w in adj[curr]:
                if w not in path_set:
                    next_v = w
                    break
            
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

def _try_merge(paths, edges_set):
    """
    Try to merge two paths in the list into a single path.
    Returns True if a merge was successful.
    """
    n_paths = len(paths)
    if n_paths < 2:
        return False
    
    # We need to check all pairs of paths.
    # For each pair (i, j), check if we can connect an end of i to an end of j.
    # Connection types:
    # 1. end(i) -- start(j)  -> new path = path_i + path_j
    # 2. end(i) -- end(j)    -> new path = path_i + reversed(path_j)
    # 3. start(i) -- start(j) -> new path = reversed(path_i) + path_j
    # 4. start(i) -- end(j)    -> new path = reversed(path_i) + reversed(path_j)
    
    # To be efficient, we can precompute sets of vertices for each path to check simplicity quickly?
    # Actually, simplicity of the merged path requires that the two paths are vertex-disjoint 
    # EXCEPT for the connecting vertex? No, the paths are already simple and edge-disjoint.
    # The new path is simple if and only if the vertex sets of the two paths are disjoint.
    # Wait, if they share a vertex, the union might not be a simple path.
    # Example: P1: 1-2-3, P2: 3-4-5. Merge at 3: 1-2-3-4-5. Simple.
    # Example: P1: 1-2-3, P2: 2-4-5. Merge at 2? 
    # If we connect end(1)=3 to start(2)=2? Edge (3,2) must exist.
    # New path: 1-2-3-2-4-5. Not simple (2 repeated).
    # So we MUST ensure vertex sets are disjoint before attempting a merge that doesn't share endpoints?
    # Actually, if the connection is between distinct vertices u (in P1) and v (in P2), and u != v,
    # then the new path is simple iff V(P1) ∩ V(P2) = ∅.
    # If u == v, then the paths share a vertex. But they are edge-disjoint.
    # If they share a vertex, can we merge?
    # If P1 ends at u and P2 starts at u.
    # P1: ...-x-u. P2: u-y-...
    # Merged: ...-x-u-y-... This is a simple path as long as no other vertices are shared.
    # So generally, we should check if the union of vertices is a set of size len(P1)+len(P2) (disjoint)
    # OR if they share exactly one vertex which is the connection point, and no other vertices.
    
    # Simpler heuristic: Only try to merge if vertex sets are disjoint.
    # This is a safe subset of valid merges. It might miss some merges where paths touch at a node,
    # but disjointness is the most common case for "separate" paths.
    
    # Optimization: Sort paths by length? Or just iterate.
    # Given small N (20-30), number of paths is small (< 15).
    # Pairs are ~ 15*14/2 = 105.
    # Checking disjointness is O(V).
    
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            p1 = paths[i]
            p2 = paths[j]
            
            # Check vertex disjointness
            set1 = set(p1)
            set2 = set(p2)
            if set1 & set2:
                # If they share vertices, it's more complex. 
                # For simplicity and speed, skip non-disjoint merges in this first pass.
                # We can add a second pass for shared-vertex merges if needed.
                continue
                
            # Check possible connections
            # Ends of p1: p1[0], p1[-1]
            # Ends of p2: p2[0], p2[-1]
            
            # 1. p1[-1] -- p2[0]
            if tuple(sorted((p1[-1], p2[0]))) in edges_set:
                # Merge: p1 + p2
                new_path = p1 + p2
                # Remove p2, replace p1
                paths[i] = new_path
                paths.pop(j)
                return True
                
            # 2. p1[-1] -- p2[-1]
            if tuple(sorted((p1[-1], p2[-1]))) in edges_set:
                # Merge: p1 + reversed(p2)
                new_path = p1 + p2[::-1]
                paths[i] = new_path
                paths.pop(j)
                return True
                
            # 3. p1[0] -- p2[0]
            if tuple(sorted((p1[0], p2[0]))) in edges_set:
                # Merge: reversed(p1) + p2
                new_path = p1[::-1] + p2
                paths[i] = new_path
                paths.pop(j)
                return True
                
            # 4. p1[0] -- p2[-1]
            if tuple(sorted((p1[0], p2[-1]))) in edges_set:
                # Merge: reversed(p1) + reversed(p2)
                new_path = p1[::-1] + p2[::-1]
                paths[i] = new_path
                paths.pop(j)
                return True
                
    return False

def decompose(edges):
    # 1. Initial greedy decomposition
    paths = _greedy_decompose(edges)
    
    # 2. Local search: Try to merge paths
    # We need the set of all edges to check connectivity for merging.
    # Note: The edges used in the paths are a subset of 'edges'.
    # The merging condition "edge exists in graph" is sufficient because 
    # if the edge is in the graph, is it necessarily available?
    # No! The edge might already be used in one of the paths.
    # However, if the edge connects an endpoint of P1 to an endpoint of P2,
    # and P1 and P2 are vertex-disjoint, the edge (u,v) is NOT in P1 (since v not in P1) 
    # and NOT in P2 (since u not in P2).
    # So the edge is NOT used by P1 or P2.
    # Could it be used by P3? Yes.
    # If it is used by P3, we cannot use it to merge P1 and P2.
    # So we must check if the edge is "free" (not used by ANY path).
    
    # Let's build a set of used edges.
    used_edges = set()
    for p in paths:
        for k in range(len(p) - 1):
            e = tuple(sorted((p[k], p[k+1])))
            used_edges.add(e)
            
    free_edges = set(edges) - used_edges
    
    # Now try to merge using only free edges.
    # We loop until no more merges are possible.
    while True:
        # We need to pass free_edges to _try_merge, or check inside.
        # Let's modify _try_merge to accept free_edges.
        merged = _try_merge_with_free(paths, free_edges)
        if not merged:
            break
        
        # Update used_edges and free_edges
        # This is inefficient if we recalculate from scratch.
        # But N is small. Let's just recalculate for simplicity and correctness.
        used_edges = set()
        for p in paths:
            for k in range(len(p) - 1):
                e = tuple(sorted((p[k], p[k+1])))
                used_edges.add(e)
        free_edges = set(edges) - used_edges

    return paths

def _try_merge_with_free(paths, free_edges):
    """
    Try to merge two paths using only edges from free_edges.
    """
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            p1 = paths[i]
            p2 = paths[j]
            
            set1 = set(p1)
            set2 = set(p2)
            if set1 & set2:
                continue
                
            # 1. p1[-1] -- p2[0]
            if tuple(sorted((p1[-1], p2[0]))) in free_edges:
                new_path = p1 + p2
                paths[i] = new_path
                paths.pop(j)
                return True
                
            # 2. p1[-1] -- p2[-1]
            if tuple(sorted((p1[-1], p2[-1]))) in free_edges:
                new_path = p1 + p2[::-1]
                paths[i] = new_path
                paths.pop(j)
                return True
                
            # 3. p1[0] -- p2[0]
            if tuple(sorted((p1[0], p2[0]))) in free_edges:
                new_path = p1[::-1] + p2
                paths[i] = new_path
                paths.pop(j)
                return True
                
            # 4. p1[0] -- p2[-1]
            if tuple(sorted((p1[0], p2[-1]))) in free_edges:
                new_path = p1[::-1] + p2[::-1]
                paths[i] = new_path
                paths.pop(j)
                return True
                
    return False

def construct():
    deadline = time.time() + 140
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
        if time.time() > deadline:
             # Safety break, though unlikely to be needed
             pass
             
    return results
