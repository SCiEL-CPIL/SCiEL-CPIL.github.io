import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin
    
    # We want to maximize n.
    # Known best is 39. Let's try from 39 down to 31.
    for n in range(39, 30, -1):
        if time.time() > deadline:
            break
            
        # Try a few random seeds for this n
        for seed in range(50):
            if time.time() > deadline:
                break
                
            random.seed(seed)
            # Initialize graph
            # We want a graph that is likely C4-free and has small alpha.
            # Random graph with p ~ 0.2 might be good for n=39?
            # Avg degree ~ 7.7.
            p = 0.2 
            edges = set()
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < p:
                        edges.add((u, v))
            
            # Local search
            success = False
            max_iters = 2000
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # 1. Check C4s
                c4_found = False
                # Efficient C4 check: for each pair (u,v), count common neighbors
                # But we need to find *a* C4 to fix.
                # Let's just find any violation.
                # Build adjacency list for faster neighbor intersection
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                    
                # Find a C4
                # Iterate pairs
                c4_edge_to_remove = None
                for u in range(n):
                    for v in range(u+1, n):
                        common = adj[u].intersection(adj[v])
                        if len(common) > 1:
                            # Found C4: u - w1 - v - w2 - u
                            # Remove one edge, e.g., (u, w1)
                            w1 = list(common)[0]
                            c4_edge_to_remove = (u, w1)
                            c4_found = True
                            break
                    if c4_found:
                        break
                
                if c4_found:
                    # Remove the edge
                    edges.discard(c4_edge_to_remove)
                    continue
                
                # 2. Check Independent Set of size 11
                # If no C4, check alpha
                ind_set = find_independent_set(adj, n, 11)
                if ind_set is None:
                    # Valid graph!
                    success = True
                    break
                else:
                    # Found an independent set of size 11
                    # Add an edge to break it
                    # Pick two random vertices from the set
                    v1, v2 = random.sample(ind_set, 2)
                    if (v1, v2) not in edges and (v2, v1) not in edges:
                        edges.add((v1, v2))
                    else:
                        # They are already connected? Impossible for independent set.
                        # So this shouldn't happen.
                        pass

            if success:
                # Convert edges to list of lists
                edge_list = [[u, v] for u, v in edges]
                return {'n': n, 'edges': edge_list}
                
    # Fallback to seed if no better found
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def find_independent_set(adj, n, target):
    """
    Finds an independent set of size 'target' in the graph defined by adj.
    Returns the set if found, None otherwise.
    Uses simple backtracking with pruning.
    """
    # Order vertices by degree (ascending) to prune better?
    # Actually, for independent set, picking low degree vertices first might be better?
    # Or high degree?
    # Let's just use standard DFS.
    
    candidates = list(range(n))
    
    def dfs(candidates, current_set):
        if len(current_set) == target:
            return current_set[:]
        
        # Pruning: if not enough candidates left
        if len(candidates) < target - len(current_set):
            return None
            
        # Heuristic: Pick the vertex with the smallest degree in the subgraph induced by candidates?
        # Or just pick the first one.
        # To speed up, we can sort candidates by degree (in original graph) ascending.
        # Low degree vertices are more likely to be in an independent set.
        
        # Optimization: If any vertex in candidates has no neighbors in candidates,
        # it can be added to the set safely? No, we just need *one* set.
        # If we add it, we remove its neighbors.
        
        # Let's pick a vertex v.
        # Strategy: Pick v with min degree in candidates.
        v_idx = 0
        min_deg = float('inf')
        for i, v in enumerate(candidates):
            # Count neighbors in candidates
            deg = 0
            for w in candidates:
                if w != v and w in adj[v]:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
                v_idx = i
            # Early break if deg 0
            if deg == 0:
                break
                
        v = candidates[v_idx]
        
        # Option 1: Include v
        # New candidates: candidates - {v} - N(v)
        new_candidates = []
        for w in candidates:
            if w == v:
                continue
            if w in adj[v]:
                continue
            new_candidates.append(w)
            
        res = dfs(new_candidates, current_set + [v])
        if res:
            return res
            
        # Option 2: Exclude v
        new_candidates = [w for w in candidates if w != v]
        res = dfs(new_candidates, current_set)
        if res:
            return res
            
        return None
        
    return dfs(candidates, [])
