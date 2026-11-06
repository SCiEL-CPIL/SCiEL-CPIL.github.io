import random
import math

def construct():
    n = 40
    target_edges = 401
    
    # 1. Initialize with Circulant Graph C_40(1..10)
    edges = set()
    for i in range(n):
        for k in range(1, 11):
            j = (i + k) % n
            u, v = (i, j) if i < j else (j, i)
            edges.add((u, v))
            
    # 2. Add 1 edge to reach 401.
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    missing_edges = [e for e in all_possible if e not in edges]
    
    def get_max_book(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        max_book = 0
        for u, v in edge_set:
            common = len(adj[u] & adj[v])
            if common > max_book:
                max_book = common
        return max_book

    def get_sum_sq_books(edge_set, adj=None):
        if adj is None:
            adj = [set() for _ in range(n)]
            for u, v in edge_set:
                adj[u].add(v)
                adj[v].add(u)
        sum_sq = 0
        for u, v in edge_set:
            common = len(adj[u] & adj[v])
            sum_sq += common * common
        return sum_sq

    best_initial_edge = None
    best_initial_max = float('inf')
    best_initial_sum_sq = float('inf')
    
    for e in missing_edges:
        candidate_edges = edges | {e}
        mb = get_max_book(candidate_edges)
        if mb < best_initial_max or (mb == best_initial_max and get_sum_sq_books(candidate_edges) < best_initial_sum_sq):
            best_initial_max = mb
            best_initial_edge = e
            best_initial_sum_sq = get_sum_sq_books(candidate_edges)
            
    current_edges = edges | {best_initial_edge}
    current_max = best_initial_max
    current_sum_sq = best_initial_sum_sq
    
    best_edges = set(current_edges)
    best_max = current_max
    best_sum_sq = current_sum_sq
    
    rng = random.Random(123)
    
    # Precompute adjacency for current graph
    adj = [set() for _ in range(n)]
    for u, v in current_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    edge_list = list(current_edges)
    missing_list = [e for e in all_possible if e not in current_edges]
    
    # Hill Climbing
    # We limit iterations to ensure speed.
    max_iter = 10000
    
    for step in range(max_iter):
        improved = False
        
        # Pick a random edge to remove
        rem_idx = rng.randrange(len(edge_list))
        rem_edge = edge_list[rem_idx]
        
        # Try a sample of missing edges
        # To increase diversity, we can try edges that are "close" to the removed edge
        # or just random. Random is fine for now.
        
        # Sample 5 missing edges
        for _ in range(5):
            if not missing_list:
                break
            add_idx = rng.randrange(len(missing_list))
            add_edge = missing_list[add_idx]
            
            # Perform swap
            new_edges = current_edges.copy()
            new_edges.discard(rem_edge)
            new_edges.add(add_edge)
            
            # Update adjacency for new_edges to calculate books
            # This is the expensive part.
            # We can update adj incrementally?
            # Remove rem_edge: adj[ru].remove(rv), adj[rv].remove(ru)
            # Add add_edge: adj[au].add(av), adj[av].add(au)
            # Then calculate max book.
            
            ru, rv = rem_edge
            au, av = add_edge
            
            # Temporarily modify adj
            adj[ru].remove(rv)
            adj[rv].remove(ru)
            adj[au].add(av)
            adj[av].add(au)
            
            # Calculate max book
            new_max = 0
            # Only edges incident to ru, rv, au, av can change book size
            # Edges not incident to these 4 vertices have unchanged common neighbors
            # So we only need to check edges incident to ru, rv, au, av
            # And the new edge (au, av)
            
            # Collect unique vertices involved
            involved = {ru, rv, au, av}
            
            # Check all edges in new_edges that are incident to involved vertices
            # This is still potentially many edges (degree ~20 * 4 = 80)
            # But much less than 401.
            
            for u, v in new_edges:
                if u in involved or v in involved:
                    common = len(adj[u] & adj[v])
                    if common > new_max:
                        new_max = common
            
            # Revert adj
            adj[ru].add(rv)
            adj[rv].add(ru)
            adj[au].remove(av)
            adj[av].remove(au)
            
            if new_max < current_max:
                # Accept
                current_edges = new_edges
                current_max = new_max
                current_sum_sq = get_sum_sq_books(current_edges, adj) # adj is correct for current_edges now?
                # Wait, adj was reverted. We need to update adj to match current_edges.
                # Actually, we reverted to the state BEFORE the swap.
                # So we need to apply the swap to adj now.
                
                # Apply swap to adj
                adj[ru].remove(rv)
                adj[rv].remove(ru)
                adj[au].add(av)
                adj[av].add(au)
                
                if current_max < best_max or (current_max == best_max and current_sum_sq < best_sum_sq):
                    best_edges = set(current_edges)
                    best_max = current_max
                    best_sum_sq = current_sum_sq
                    improved = True
                    
                edge_list = list(current_edges)
                missing_list = [e for e in all_possible if e not in current_edges]
                break
                
            elif new_max == current_max:
                # Calculate sum sq for this candidate
                # We need adj to be in the candidate state
                # We reverted adj, so we need to re-apply or just calculate from scratch?
                # Calculating from scratch is safer.
                new_sum_sq = get_sum_sq_books(new_edges)
                
                if new_sum_sq < current_sum_sq:
                    # Accept
                    current_edges = new_edges
                    current_sum_sq = new_sum_sq
                    
                    # Update adj
                    adj[ru].remove(rv)
                    adj[rv].remove(ru)
                    adj[au].add(av)
                    adj[av].add(au)
                    
                    if current_max < best_max or (current_max == best_max and current_sum_sq < best_sum_sq):
                        best_edges = set(current_edges)
                        best_max = current_max
                        best_sum_sq = current_sum_sq
                        improved = True
                        
                    edge_list = list(current_edges)
                    missing_list = [e for e in all_possible if e not in current_edges]
                    break
                    
    return [[u, v] for u, v in best_edges]
