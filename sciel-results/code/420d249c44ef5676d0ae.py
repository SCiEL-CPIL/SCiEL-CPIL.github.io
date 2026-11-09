import random
import itertools

def is_valid(edges, n):
    # Build adjacency list and degree
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    degrees = [len(adj[i]) for i in range(n)]
    
    # Check all pairs with same degree
    for s in range(n):
        for t in range(s + 1, n):
            if degrees[s] != degrees[t]:
                continue
            
            # Check if there is a simple path of length 3 between s and t
            # s - u - v - t
            found = False
            for u in adj[s]:
                if u == t:
                    continue
                for v in adj[u]:
                    if v == s or v == t or v == u:
                        continue
                    if t in adj[v]:
                        found = True
                        break
                if found:
                    break
            if found:
                return False
    return True

def construct():
    n = 15
    best_edges = []
    best_count = 0
    
    # Helper to get edge set for fast lookup
    def get_edge_set(edges):
        return set((min(u,v), max(u,v)) for u,v in edges)
    
    # Strategy: Randomized Local Search
    # 1. Start with K_15, remove edges to fix conflicts
    # 2. Add edges to maximize count
    
    for restart in range(200):
        # Initialize with K_15
        current_edges = [[u, v] for u in range(n) for v in range(u+1, n)]
        current_set = get_edge_set(current_edges)
        
        # Phase 1: Remove edges until valid
        valid = False
        max_removals = 100 # Safety limit
        removals = 0
        while not valid and removals < max_removals:
            if is_valid(current_edges, n):
                valid = True
                break
            
            # Find a conflict
            # We need to find s,t,u,v such that deg(s)==deg(t) and path s-u-v-t exists
            # is_valid returns False but doesn't give the conflict.
            # We will re-implement a finder or just pick random edges to remove?
            # Picking random edges is inefficient.
            
            # Let's find the conflict explicitly
            adj = [set() for _ in range(n)]
            for u, v in current_edges:
                adj[u].add(v)
                adj[v].add(u)
            degrees = [len(adj[i]) for i in range(n)]
            
            conflict_found = False
            for s in range(n):
                for t in range(s + 1, n):
                    if degrees[s] != degrees[t]:
                        continue
                    # Check path
                    for u in adj[s]:
                        if u == t: continue
                        for v in adj[u]:
                            if v == s or v == t or v == u: continue
                            if t in adj[v]:
                                # Conflict found: path s-u-v-t
                                # Edges: (s,u), (u,v), (v,t)
                                # Remove one randomly
                                candidates = [
                                    (min(s,u), max(s,u)),
                                    (min(u,v), max(u,v)),
                                    (min(v,t), max(v,t))
                                ]
                                # Ensure they are in current_set
                                valid_candidates = [c for c in candidates if c in current_set]
                                if valid_candidates:
                                    edge_to_remove = random.choice(valid_candidates)
                                    # Remove it
                                    u_rem, v_rem = edge_to_remove
                                    current_set.remove(edge_to_remove)
                                    current_edges = [e for e in current_edges if (min(e[0],e[1]), max(e[0],e[1])) != edge_to_remove]
                                    conflict_found = True
                                    break
                        if conflict_found: break
                    if conflict_found: break
                if conflict_found: break
            
            if not conflict_found:
                # Should not happen if is_valid said False, but safety
                break
            removals += 1
        
        if not valid:
            continue
            
        # Phase 2: Add edges to maximize
        improved = True
        while improved:
            improved = False
            # Try all missing edges
            for u in range(n):
                for v in range(u+1, n):
                    if (u,v) not in current_set:
                        # Try adding
                        new_set = current_set | {(u,v)}
                        # Check validity
                        # To save time, we can just call is_valid on the list
                        # But we need to construct the list.
                        # Since n is small, we can just append and check.
                        test_edges = current_edges + [[u,v]]
                        if is_valid(test_edges, n):
                            current_set.add((u,v))
                            current_edges.append([u,v])
                            improved = True
                            if len(current_edges) > best_count:
                                best_count = len(current_edges)
                                best_edges = list(current_edges)
        
        # Optional: Local search swap (remove 1, add 1) to escape local optima?
        # Given time, simple add is probably enough if we have many restarts.
        
    return best_edges
