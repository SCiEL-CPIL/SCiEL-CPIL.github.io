"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n = 40
    target_edges = 401
    
    # Set random seed for reproducibility within the function
    rng = random.Random(42)
    
    # Generate a random graph with exactly target_edges edges
    all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edges = set(rng.sample(all_possible, target_edges))
    
    # Build adjacency structure
    def build_adj(edges_set):
        adj = [set() for _ in range(n)]
        for u, v in edges_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj
    
    def compute_books(adj, edges_set):
        """Compute book size for each edge: number of common neighbors."""
        books = {}
        for u, v in edges_set:
            # Ensure u < v for consistent key
            if u > v:
                u, v = v, u
            common = len(adj[u] & adj[v])
            books[(u, v)] = common
        return books
    
    def max_book(books):
        if not books:
            return 0
        return max(books.values())
    
    adj = build_adj(edges)
    books = compute_books(adj, edges)
    current_max = max_book(books)
    
    # Local search: try to reduce max book by swapping edges
    # For the edge with max book, try removing it and adding a non-edge
    # that doesn't increase the max book
    
    iterations = 50000
    for it in range(iterations):
        if current_max <= 6:  # Theoretical lower bound is 40/6 ≈ 6.67, so 7 is achievable
            break
            
        # Find edge(s) with maximum book
        max_edges = [(u, v) for (u, v), b in books.items() if b == current_max]
        if not max_edges:
            break
        
        # Pick one max-book edge to try to remove
        u, v = rng.choice(max_edges)
        
        # Try adding a non-edge that minimizes the resulting max book
        # Sample some candidate non-edges
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in edges and (j, i) not in edges]
        
        if not non_edges:
            break
        
        # Sample a subset of non-edges for efficiency
        sample_size = min(200, len(non_edges))
        candidates = rng.sample(non_edges, sample_size)
        
        best_swap = None
        best_max_after = current_max + 1  # Only accept if max doesn't increase
        
        for a, b in candidates:
            # Skip if this would create a duplicate or self-loop
            if a == b:
                continue
            if (a, b) in edges or (b, a) in edges:
                continue
            
            # Simulate the swap: remove (u,v), add (a,b)
            # Compute new books affected
            # Only edges sharing vertices with (u,v) or (a,b) change
            
            # Quick check: what's the book of (a,b) after adding it?
            # (a,b) book = |N(a) ∩ N(b)| in current graph (before removing u,v)
            # But if u or v is in N(a)∩N(b), removing (u,v) doesn't change N(a)∩N(b) unless u-v edge matters
            # Actually, removing edge (u,v) changes adj[u] and adj[v]
            
            # Let's do a more careful simulation for promising candidates
            # First, quick estimate: book of new edge (a,b)
            common_ab = len(adj[a] & adj[b])
            
            # If common_ab >= current_max, this swap might not help much
            # But removing (u,v) might reduce some other books
            
            # Full simulation is expensive, so only do it for candidates with common_ab < current_max
            if common_ab >= current_max:
                continue
            
            # Simulate: remove (u,v), add (a,b)
            # Temporarily update adj
            adj[u].discard(v)
            adj[v].discard(u)
            adj[a].add(b)
            adj[b].add(a)
            
            # Recompute books for edges that might be affected
            # Affected edges: those sharing a vertex with (u,v) or (a,b)
            affected_vertices = {u, v, a, b}
            
            new_max = 0
            # Check all edges, but optimize by only checking affected ones + old max edges
            # For simplicity in this search, check a subset
            # Actually, let's just compute max book properly but efficiently
            
            # Edges that could have increased: those incident to a or b (new edge)
            # Edges that could have decreased: those incident to u or v
            # Other edges unchanged
            
            # Check new edge (a,b)
            common_ab_new = len(adj[a] & adj[b])
            new_max = max(new_max, common_ab_new)
            
            # Check edges incident to a (excluding (a,b))
            for x in adj[a]:
                if x == b:
                    continue
                if x > a:
                    key = (a, x)
                else:
                    key = (x, a)
                if key in books:  # was an edge
                    common = len(adj[a] & adj[x])
                    new_max = max(new_max, common)
            
            # Check edges incident to b (excluding (a,b))
            for x in adj[b]:
                if x == a:
                    continue
                if x > b:
                    key = (b, x)
                else:
                    key = (x, b)
                if key in books:
                    common = len(adj[b] & adj[x])
                    new_max = max(new_max, common)
            
            # Check edges incident to u (excluding (u,v))
            for x in adj[u]:
                if x == v:
                    continue
                if x > u:
                    key = (u, x)
                else:
                    key = (x, u)
                if key in books:
                    common = len(adj[u] & adj[x])
                    new_max = max(new_max, common)
            
            # Check edges incident to v (excluding (u,v))
            for x in adj[v]:
                if x == u:
                    continue
                if x > v:
                    key = (v, x)
                else:
                    key = (x, v)
                if key in books:
                    common = len(adj[v] & adj[x])
                    new_max = max(new_max, common)
            
            # Also need to check that no OTHER edge has book > new_max
            # This is the expensive part. For now, trust that if local changes are good, global is too.
            # But this is a heuristic. Let's do a full check for safety when new_max < current_max
            if new_max < current_max:
                # Full verification
                for eu, ev in edges:
                    if (eu, ev) == (u, v):
                        continue
                    if eu > ev:
                        eu, ev = ev, eu
                    common = len(adj[eu] & adj[ev])
                    if common > new_max:
                        new_max = common
                        break
            
            # Restore adj
            adj[a].discard(b)
            adj[b].discard(a)
            adj[u].add(v)
            adj[v].add(u)
            
            if new_max < best_max_after:
                best_max_after = new_max
                best_swap = (a, b)
        
        if best_swap is not None and best_max_after < current_max:
            a, b = best_swap
            # Perform the swap
            edges.discard((u, v))
            edges.discard((v, u))
            edges.add((a, b))
            edges.add((b, a))
            
            # Rebuild adj
            adj = build_adj(edges)
            books = compute_books(adj, edges)
            current_max = max_book(books)
        else:
            # No improving swap found, try a different max edge
            continue
    
    # Return edges as list of [u, v]
    result = []
    for u, v in edges:
        result.append([u, v])
    
    return result
# EVOLVE-BLOCK-END
