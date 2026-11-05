"""Combined: structural K20,20 init + efficient bitmask local search to minimize largest book."""
import time
import random

def construct():
    n = 40
    m = 401
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin

    # --- Structural initialization: K20,20 minus 9 edges + star of 10 in A ---
    # A = 0..19, B = 20..39
    # Remove (0, 20)..(0, 28) from bipartite, add (0,1)..(0,10) internal
    edges = []
    edge_set = set()
    
    for u in range(20):
        for v in range(20, 40):
            if u == 0 and 20 <= v <= 28:
                continue
            edge_set.add((u, v))
            edges.append((u, v))
    
    for i in range(1, 11):
        edge_set.add((0, i))
        edges.append((0, i))
    
    assert len(edge_set) == m, f"Expected {m} edges, got {len(edge_set)}"

    # --- Bitmask adjacency for fast common-neighbor computation ---
    adj = [0] * n
    for u, v in edge_set:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def compute_max_book(adj, edge_set):
        mb = 0
        for u, v in edge_set:
            common = (adj[u] & adj[v]).bit_count()
            if common > mb:
                mb = common
        return mb

    def compute_all_books(adj, edge_set):
        books = {}
        for u, v in edge_set:
            books[(u, v)] = (adj[u] & adj[v]).bit_count()
        return books

    # Initial score
    best_max = compute_max_book(adj, edge_set)
    best_edge_set = set(edge_set)
    best_adj = list(adj)

    # --- Local search: edge swaps to reduce max book ---
    # Precompute all non-edges for candidate additions
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # We'll use a list of non-edges as candidates
    non_edges = [p for p in all_pairs if p not in edge_set]
    random.shuffle(non_edges)
    
    # Track which edges are in the graph for O(1) lookup
    in_graph = [False] * (n * n)
    for u, v in edge_set:
        in_graph[u * n + v] = True
        in_graph[v * n + u] = True

    # Helper to update in_graph
    def add_edge(u, v):
        edge_set.add((u, v))
        in_graph[u * n + v] = True
        in_graph[v * n + u] = True
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v):
        edge_set.discard((u, v))
        in_graph[u * n + v] = False
        in_graph[v * n + u] = False
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Local search loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Get current books
        books = compute_all_books(adj, edge_set)
        
        # Sort edges by book size descending
        sorted_edges = sorted(books.items(), key=lambda x: -x[1])
        
        # Try to improve high-book edges
        for (u, v), bk in sorted_edges:
            if time.time() >= deadline:
                break
            if bk < best_max:
                break  # No more edges with book >= best_max
            
            # Try swapping: remove (u,v), add a non-edge (c,d)
            # Focus on non-edges with low potential book size
            # Try a limited number of candidates
            for _ in range(100):
                if time.time() >= deadline:
                    break
                
                # Pick a random non-edge
                c, d = random.choice(non_edges)
                
                # Skip if it's the edge we're removing
                if (c, d) == (u, v) or (d, c) == (u, v):
                    continue
                
                # Compute book size of (c,d) after removing (u,v)
                # adj'[c] = adj[c] minus {u} if u==c or if u in adj[c] and we're removing (u,v)... 
                # Actually, removing (u,v) only changes adj[u] and adj[v].
                # So if c not in {u,v} and d not in {u,v}, common(c,d) is unchanged.
                # If c or d is u or v, we need to adjust.
                
                ac = adj[c]
                ad = adj[d]
                
                # Adjust for removal of (u,v)
                if c == u:
                    ac = adj[u] & ~(1 << v)
                elif c == v:
                    ac = adj[v] & ~(1 << u)
                else:
                    ac = adj[c]
                    # If u or v is a neighbor of c, and we're removing (u,v), 
                    # c's adjacency doesn't change unless c is u or v.
                    # So no adjustment needed for c not in {u,v}.
                
                if d == u:
                    ad = adj[u] & ~(1 << v)
                elif d == v:
                    ad = adj[v] & ~(1 << u)
                else:
                    ad = adj[d]
                
                common_cd = (ac & ad).bit_count()
                
                # Only consider if this would reduce the book size of the new edge
                # compared to the old edge
                if common_cd >= bk:
                    continue
                
                # Perform the swap tentatively
                remove_edge(u, v)
                add_edge(c, d)
                
                # Recompute max book
                new_max = compute_max_book(adj, edge_set)
                
                if new_max <= best_max:
                    best_max = new_max
                    best_edge_set = set(edge_set)
                    best_adj = list(adj)
                    improved = True
                    # Update non_edges list: (c,d) is now an edge, (u,v) is now non-edge
                    # Remove (c,d) from non_edges if present
                    try:
                        non_edges.remove((c, d))
                    except ValueError:
                        pass
                    # Add (u,v) to non_edges
                    non_edges.append((u, v))
                    # Shuffle to randomize
                    random.shuffle(non_edges)
                    break  # Try next high-book edge
                else:
                    # Revert
                    remove_edge(c, d)
                    add_edge(u, v)
                    # Update non_edges back
                    try:
                        non_edges.remove((u, v))
                    except ValueError:
                        pass
                    non_edges.append((c, d))
                    random.shuffle(non_edges)
            
            if time.time() >= deadline:
                break
    
    # Return the best edge set found
    return [[u, v] for u, v in best_edge_set]
