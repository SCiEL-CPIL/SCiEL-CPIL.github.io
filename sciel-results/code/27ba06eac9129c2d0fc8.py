import time
import sys
from collections import deque

# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set(tuple(sorted(e)) for e in P['edges'])
    
    # Deadline guard: evaluator kills at 90s, use 85s limit
    deadline = time.time() + 85.0
    
    # Precompute bipartite candidate edges (even-odd pairs not in base)
    # This guarantees triangle-free since the graph remains bipartite
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u + v) % 2 == 1:  # one even, one odd
                e = (u, v) if u < v else (v, u)
                if e not in base_edges:
                    candidates.append(e)
    
    # Build adjacency helper
    def build_adj(added_set):
        adj = [[] for _ in range(n)]
        for u, v in P['edges']:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in added_set:
            adj[u].append(v)
            adj[v].append(u)
        return adj
    
    def get_diameter(adj):
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        q.append(nb)
                        if dist[nb] > max_d:
                            max_d = dist[nb]
                            if max_d > 4:
                                return max_d
        return max_d
    
    def is_valid(added_set):
        # Check no duplicates and no base edges
        for e in added_set:
            if e in base_edges:
                return False
        adj = build_adj(added_set)
        return get_diameter(adj) <= 4
    
    # Greedy construction: start empty, add edges that most reduce diameter
    # then prune
    def greedy_construct():
        # Start with a hub-based solution as initial guess
        # Try multiple hubs and pick best
        best_added = None
        best_diam = 100
        
        for hub in range(n):
            # Connect hub to all opposite-parity vertices
            added = []
            for v in range(n):
                if v != hub and (hub + v) % 2 == 1:
                    e = tuple(sorted((hub, v)))
                    if e not in base_edges:
                        added.append(e)
            
            adj = build_adj(added)
            d = get_diameter(adj)
            if d < best_diam:
                best_diam = d
                best_added = added[:]
        
        if best_added is None:
            return []
        
        # Prune: repeatedly try to remove edges
        improved = True
        while improved:
            improved = False
            for i in range(len(best_added)):
                if time.time() > deadline - 5.0:
                    return best_added
                trial = best_added[:i] + best_added[i+1:]
                if is_valid(trial):
                    best_added = trial
                    improved = True
                    break
        
        return best_added
    
    # Local search: start from greedy solution, try swapping edges
    def local_search(added_set):
        current = added_set[:]
        current_diam = get_diameter(build_adj(current))
        
        # Try to remove edges
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            removed = False
            for i in range(len(current)):
                trial = current[:i] + current[i+1:]
                if is_valid(trial):
                    current = trial
                    removed = True
                    break
            if not removed:
                break
        
        # Try swapping: remove one edge, add another
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            swapped = False
            for i in range(len(current)):
                for cand in candidates:
                    if cand in current:
                        continue
                    trial = current[:i] + current[i+1:] + [cand]
                    # Check no duplicates
                    if len(trial) != len(set(trial)):
                        continue
                    if is_valid(trial):
                        current = trial
                        swapped = True
                        break
                if swapped:
                    break
            if not swapped:
                break
        
        return current
    
    # Branch and bound for small edge counts
    def branch_and_bound(added_set, target_size):
        """Try to find a valid config of exactly target_size edges using B&B"""
        if len(added_set) > target_size:
            return None
        if len(added_set) == target_size:
            if is_valid(added_set):
                return added_set[:]
            return None
        
        # Get current diameter to guide search
        adj = build_adj(added_set)
        diam = get_diameter(adj)
        if diam > 4:
            # Need more edges, find which pair to add
            # Find vertices at distance > 2 from each other
            # Prioritize adding edges that connect distant vertices
            far_pairs = []
            for u in range(n):
                for v in range(u+1, n):
                    e = (u, v) if u < v else (v, u)
                    if e in base_edges or e in set(added_set):
                        continue
                    if (u + v) % 2 == 1:
                        # Check if this edge helps
                        far_pairs.append(e)
            
            # Sort by how "central" the edge is (reduces diameter most)
            # Heuristic: prefer edges between vertices with high eccentricity
            def edge_priority(e):
                u, v = e
                # Distance from u to v in current graph
                dist = [-1] * n
                dist[u] = 0
                q = deque([u])
                while q:
                    node = q.popleft()
                    for nb in adj[node]:
                        if dist[nb] == -1:
                            dist[nb] = dist[node] + 1
                            q.append(nb)
                # Prefer edges where both endpoints are far from center
                return -(dist[v] if dist[v] != -1 else 0)
            
            far_pairs.sort(key=edge_priority)
            
            for e in far_pairs[:20]:  # Limit branching
                if time.time() > deadline - 5.0:
                    return None
                result = branch_and_bound(added_set + [e], target_size)
                if result is not None:
                    return result
            return None
        else:
            # Already valid, try to see if we can do with fewer
            # But we're building up, so this shouldn't happen in this context
            return added_set[:]
    
    # Main strategy
    # 1. Start with greedy hub-based construction
    # 2. Prune aggressively
    # 3. Try local search
    # 4. Try branch and bound for smaller sizes
    
    best_result = greedy_construct()
    best_size = len(best_result)
    
    # Try local search from the greedy result
    if best_result:
        ls_result = local_search(best_result)
        if len(ls_result) < best_size:
            best_result = ls_result
            best_size = len(ls_result)
    
    # Try branch and bound for smaller sizes
    for target in range(max(1, best_size - 3), best_size):
        if time.time() > deadline - 10.0:
            break
        # Start B&B from empty or from a subset
        # Try from empty with B&B
        result = branch_and_bound([], target)
        if result is not None:
            best_result = result
            best_size = target
            break
    
    # Also try different hub configurations with pruning
    for hub in range(n):
        if time.time() > deadline - 5.0:
            break
        added = []
        for v in range(n):
            if v != hub and (hub + v) % 2 == 1:
                e = tuple(sorted((hub, v)))
                if e not in base_edges:
                    added.append(e)
        
        # Aggressive pruning
        improved = True
        while improved:
            improved = False
            for i in range(len(added)):
                if time.time() > deadline - 5.0:
                    break
                trial = added[:i] + added[i+1:]
                if is_valid(trial):
                    added = trial
                    improved = True
                    break
        
        if len(added) < best_size:
            best_result = added
            best_size = len(added)
    
    # Final local search on best result
    if best_result:
        final = local_search(best_result)
        if len(final) < best_size:
            best_result = final
    
    # Convert to list of [u, v]
    if best_result:
        return [[int(u), int(v)] for u, v in best_result]
    
    # Fallback: known 10-edge solution
    return [[11, 0], [11, 2], [11, 4], [11, 6], [11, 8], [11, 14], [11, 16], [11, 18], [11, 20], [11, 22]]

# EVOLVE-BLOCK-END
