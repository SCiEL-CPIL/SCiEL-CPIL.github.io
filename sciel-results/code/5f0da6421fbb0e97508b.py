import time
import random

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 85
    n = 15
    
    def check_constraint(adj, edges_set):
        """Check if adding any edge would violate. Returns True if current graph is valid."""
        for u, v in edges_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u:
                        continue
                    if s != t and len(adj[s]) == len(adj[t]):
                        return False
        return True
    
    def try_add_edge(adj, edges_set, u, v):
        """Try to add edge (u,v). Returns True if valid."""
        edges_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        
        # Check all 3-paths that involve this new edge
        valid = True
        # Paths s-u-v-t where s~u, u~v, v~t, s!=t
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u:
                    continue
                if s != t and len(adj[s]) == len(adj[t]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            # Paths s-v-u-t
            for s in adj[v]:
                if s == u:
                    continue
                for t in adj[u]:
                    if t == v:
                        continue
                    if s != t and len(adj[s]) == len(adj[t]):
                        valid = False
                        break
                if not valid:
                    break
        
        if not valid:
            edges_set.discard((u, v))
            adj[u].discard(v)
            adj[v].discard(u)
        
        return valid
    
    best_edges = set()
    best_count = 0
    
    # Multiple random restarts with different edge orderings
    for attempt in range(200):
        if time.time() > deadline - 3:
            break
        
        rng = random.Random(attempt * 104729 + 7)
        
        adj = [set() for _ in range(n)]
        edges_set = set()
        
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if time.time() > deadline - 3:
                break
            try_add_edge(adj, edges_set, u, v)
        
        if len(edges_set) > best_count:
            best_count = len(edges_set)
            best_edges = set(edges_set)
        
        # Early stop if we hit theoretical max
        if best_count >= 56:
            break
    
    # Local search: try to swap edges to improve
    if time.time() < deadline - 5:
        adj = [set() for _ in range(n)]
        for u, v in best_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_pairs)
            
            for u, v in all_pairs:
                if time.time() > deadline - 3:
                    break
                if (u, v) in best_edges:
                    continue
                
                # Try adding (u,v)
                if try_add_edge(adj, best_edges, u, v):
                    improved = True
    
    result = []
    for u, v in best_edges:
        result.append([u, v])
    
    return result

# EVOLVE-BLOCK-END
