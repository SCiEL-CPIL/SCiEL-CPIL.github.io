import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n = 16
    
    # Build Clebsch graph: vertices 0..15, edge (u,v) if popcount(u^v) in {1,4}
    def build_clebsch():
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                xor = u ^ v
                dist = bin(xor).count('1')
                if dist == 1 or dist == 4:
                    edges.add(frozenset((u, v)))
        return edges
    
    clebsch = build_clebsch()
    
    # Convert to adjacency lists
    def edges_to_adj(edge_set):
        adj = [set() for _ in range(n)]
        for e in edge_set:
            u, v = tuple(e)
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def is_triangle_free_adj(adj):
        for u in range(n):
            for v in adj[u]:
                for w in adj[v]:
                    if w != u and w in adj[u]:
                        return False
        return True
    
    def is_3_colorable_adj(adj):
        """Check 3-colorability with backtracking, ordered by degree."""
        color = [-1] * n
        # Order vertices by degree descending for better pruning
        degrees = [len(adj[v]) for v in range(n)]
        order = sorted(range(n), key=lambda v: degrees[v], reverse=True)
        
        def backtrack(idx):
            if idx == n:
                return True
            v = order[idx]
            for c in range(3):
                valid = True
                for u in adj[v]:
                    if color[u] == c:
                        valid = False
                        break
                if valid:
                    color[v] = c
                    if backtrack(idx + 1):
                        return True
                    color[v] = -1
            return False
        
        return backtrack(0)
    
    # Verify Clebsch is valid
    adj = edges_to_adj(clebsch)
    assert is_triangle_free_adj(adj), "Clebsch should be triangle-free"
    assert not is_3_colorable_adj(adj), "Clebsch should not be 3-colorable"
    
    best_edges = set(clebsch)
    best_adj = [set(s) for s in adj]
    best_count = len(best_edges)
    
    # Precompute all possible edges not in Clebsch
    all_missing_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if frozenset((u, v)) not in clebsch:
                all_missing_edges.append((u, v))
    
    # Local search: try adding edges, then swap edges to improve
    def try_add_single_edge(adj, edge_set, u, v):
        """Try adding edge (u,v). Returns (new_adj, new_edge_set) if valid, else None."""
        # Check triangle-free: no w such that (u,w) and (v,w) both exist
        for w in adj[u]:
            if w in adj[v]:
                return None
        # Check 4-chromatic: adding edge should not make it 3-colorable
        new_adj = [set(s) for s in adj]
        new_adj[u].add(v)
        new_adj[v].add(u)
        if is_3_colorable_adj(new_adj):
            return None
        new_edge_set = set(edge_set)
        new_edge_set.add(frozenset((u, v)))
        return new_adj, new_edge_set
    
    def try_swap(adj, edge_set, remove_edge, add_edge):
        """Try swapping: remove remove_edge, add add_edge."""
        u, v = remove_edge
        x, y = add_edge
        # First remove
        new_adj = [set(s) for s in adj]
        new_adj[u].discard(v)
        new_adj[v].discard(u)
        # Check triangle-free after removal (should still be triangle-free)
        # Then add new edge
        for w in new_adj[x]:
            if w in new_adj[y]:
                return None
        new_adj[x].add(y)
        new_adj[y].add(x)
        # Check 4-chromatic
        if is_3_colorable_adj(new_adj):
            return None
        new_edge_set = set(edge_set)
        new_edge_set.discard(frozenset(remove_edge))
        new_edge_set.add(frozenset(add_edge))
        return new_adj, new_edge_set
    
    # Greedy edge addition
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        for u, v in all_missing_edges:
            if time.time() > deadline:
                break
            if frozenset((u, v)) in best_edges:
                continue
            result = try_add_single_edge(best_adj, best_edges, u, v)
            if result is not None:
                best_adj, best_edges = result
                best_count += 1
                improved = True
                # Reorder missing edges
                all_missing_edges = [(a, b) for a in range(n) for b in range(a + 1, n) 
                                     if frozenset((a, b)) not in best_edges]
                break
        if not improved:
            break
    
    # Local search with swaps
    for _ in range(100):
        if time.time() > deadline:
            break
        improved = False
        
        # Try single edge swaps: remove one edge, add one missing edge
        current_edges_list = list(best_edges)
        missing_edges = [(a, b) for a in range(n) for b in range(a + 1, n) 
                        if frozenset((a, b)) not in best_edges]
        
        # Sort by some heuristic
        for remove_edge in current_edges_list:
            if time.time() > deadline:
                break
            u, v = tuple(remove_edge)
            # For each missing edge, try swap
            for add_edge in missing_edges:
                x, y = add_edge
                # Quick check: skip if add_edge would create triangle even without swap
                # (this is a necessary condition, not sufficient)
                # Just try the swap
                result = try_swap(best_adj, best_edges, remove_edge, add_edge)
                if result is not None:
                    best_adj, best_edges = result
                    improved = True
                    # Update missing edges
                    missing_edges = [(a, b) for a in range(n) for b in range(a + 1, n) 
                                    if frozenset((a, b)) not in best_edges]
                    break
            if improved:
                break
        
        if not improved:
            # Try random perturbations
            if len(missing_edges) > 0 and len(current_edges_list) > 0:
                # Remove 1-2 random edges, add 1-2 random missing edges
                k_remove = random.randint(1, 2)
                k_add = random.randint(1, 2)
                
                remove_candidates = random.sample(current_edges_list, min(k_remove, len(current_edges_list)))
                add_candidates = random.sample(missing_edges, min(k_add, len(missing_edges)))
                
                new_adj = [set(s) for s in best_adj]
                new_edges = set(best_edges)
                
                valid = True
                for re in remove_candidates:
                    u, v = tuple(re)
                    new_adj[u].discard(v)
                    new_adj[v].discard(u)
                    new_edges.discard(frozenset(re))
                
                for ae in add_candidates:
                    x, y = ae
                    # Check triangle-free
                    for w in new_adj[x]:
                        if w in new_adj[y]:
                            valid = False
                            break
                    if not valid:
                        break
                    new_adj[x].add(y)
                    new_adj[y].add(x)
                    new_edges.add(frozenset(ae))
                
                if valid and not is_3_colorable_adj(new_adj):
                    best_adj = new_adj
                    best_edges = new_edges
                    improved = True
        
        if not improved:
            # Random restart with perturbation
            if random.random() < 0.3:
                # Perturb: remove 3-5 random edges
                current_edges_list = list(best_edges)
                k = random.randint(3, 5)
                remove_set = set(random.sample(current_edges_list, min(k, len(current_edges_list))))
                
                new_adj = [set(s) for s in best_adj]
                new_edges = set(best_edges)
                for re in remove_set:
                    u, v = tuple(re)
                    new_adj[u].discard(v)
                    new_adj[v].discard(u)
                    new_edges.discard(frozenset(re))
                
                # Try to greedily add edges back
                missing_edges = [(a, b) for a in range(n) for b in range(a + 1, n) 
                                if frozenset((a, b)) not in new_edges]
                
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    found = False
                    for u, v in missing_edges:
                        result = try_add_single_edge(new_adj, new_edges, u, v)
                        if result is not None:
                            new_adj, new_edges = result
                            missing_edges = [(a, b) for a in range(n) for b in range(a + 1, n) 
                                            if frozenset((a, b)) not in new_edges]
                            found = True
                            break
                    if not found:
                        break
                
                # Only accept if better
                if len(new_edges) > best_count:
                    best_adj = new_adj
                    best_edges = new_edges
                    best_count = len(best_edges)
    
    # Convert best_adj to edge list
    result = []
    for u in range(n):
        for v in best_adj[u]:
            if u < v:
                result.append([u, v])
    
    return result
