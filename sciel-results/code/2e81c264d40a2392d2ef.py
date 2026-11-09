import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 40
    target_edges = 401
    
    # 1. Deterministic Initialization:
    # Start with a balanced 5-partite graph structure (blow-up of C5).
    # 40 vertices, 8 per part.
    # Parts: 0..7, 8..15, 16..23, 24..31, 32..39
    # Edges between consecutive parts (cyclically).
    # Total edges: 5 * (8 * 8) = 320.
    # We need 401 edges.
    # We will add 81 more edges.
    # To maintain balance, we add edges between non-adjacent parts (distance 2 in C5).
    # There are 5 such pairs. 5 * 8 * 8 = 320 available.
    # We need 81. We can distribute them evenly.
    
    adj = np.zeros((n, n), dtype=bool)
    
    # Define parts
    part_size = 8
    parts = [list(range(i * part_size, (i + 1) * part_size)) for i in range(5)]
    
    # Add edges for C5 structure (distance 1)
    for i in range(5):
        p1 = parts[i]
        p2 = parts[(i + 1) % 5]
        for u in p1:
            for v in p2:
                adj[u, v] = True
                adj[v, u] = True
                
    # Current edges: 320. Need 81 more.
    # Add edges for distance 2 (non-adjacent parts)
    # Pairs: (0,2), (1,3), (2,4), (3,0), (4,1)
    dist2_pairs = [(0, 2), (1, 3), (2, 4), (3, 0), (4, 1)]
    
    edges_to_add = target_edges - 320 # 81
    
    # Distribute edges_to_add across the 5 pairs as evenly as possible
    # 81 / 5 = 16.2. So some get 17, some get 16.
    # 1 pair gets 17, 4 pairs get 16. 17 + 4*16 = 17+64=81.
    
    counts = [16] * 5
    counts[0] = 17
    
    for idx, (i, j) in enumerate(dist2_pairs):
        p1 = parts[i]
        p2 = parts[j]
        c = counts[idx]
        
        # Add 'c' edges between p1 and p2
        # To keep it regular-ish, we can just take the first 'c' combinations
        # or distribute round-robin.
        # Let's just take a deterministic subset.
        added = 0
        for u in p1:
            if added >= c:
                break
            for v in p2:
                if added >= c:
                    break
                adj[u, v] = True
                adj[v, u] = True
                added += 1
                
    # Now we have exactly 401 edges.
    # This graph is 5-partite, so it has no triangles within parts, 
    # but it has triangles formed by vertices in 3 mutually adjacent parts?
    # In a 5-partite graph with edges only between distance 1 and 2:
    # Triangles can only form if there is a triangle in the "quotient" graph.
    # The quotient graph has vertices 0..4. Edges (i, i+1) and (i, i+2).
    # This quotient graph contains triangles? 
    # 0-1, 1-2, 0-2. Yes, (0,1,2) is a triangle in the quotient.
    # So the graph has triangles.
    
    # 2. Spectral Heuristic / Local Search
    # We want to minimize the max book size.
    # Book size of edge (u,v) is number of common neighbors.
    # Let T be the triangle count matrix? No, just track common neighbors.
    
    # We will perform swaps to reduce the maximum book size.
    # A swap: remove edge (a,b), add edge (c,d).
    # We prioritize removing edges with high book size and adding edges with low potential book size.
    
    def get_books(adj_mat):
        books = np.zeros((n, n), dtype=int)
        for u in range(n):
            for v in range(u + 1, n):
                if adj_mat[u, v]:
                    books[u, v] = np.sum(adj_mat[u] & adj_mat[v])
        return books
    
    def max_book(adj_mat):
        # Fast max book calculation
        max_b = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj_mat[u, v]:
                    b = np.sum(adj_mat[u] & adj_mat[v])
                    if b > max_b:
                        max_b = b
        return int(max_b)
    
    current_score = max_book(adj)
    
    # Optimization Loop
    # We use a deterministic hill climbing with swaps.
    # To avoid getting stuck, we allow "neutral" moves (score stays same) 
    # if they reduce the variance of book sizes (spectral balance).
    
    best_score = current_score
    best_adj = adj.copy()
    
    # Precompute list of edges for fast iteration? 
    # n=40 is small.
    
    no_improvement_count = 0
    max_no_improvement = 500
    
    while time.time() < deadline:
        if no_improvement_count > max_no_improvement:
            break
            
        # Find the edge with the maximum book size
        max_edge = None
        max_b = -1
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    b = np.sum(adj[u] & adj[v])
                    if b > max_b:
                        max_b = b
                        max_edge = (u, v)
                        
        if max_edge is None:
            break
            
        # Try to improve by swapping max_edge with another edge
        # We look for an edge (c, d) not in the graph such that
        # removing (u,v) and adding (c,d) reduces the max book size.
        
        # To be efficient, we only consider a subset of candidate additions.
        # Candidates: edges between vertices that currently have low common neighbor counts.
        
        found_improvement = False
        found_neutral = False
        
        # Iterate over all possible non-edges (c, d)
        # There are ~380 non-edges.
        # For each, simulate the swap.
        
        u, v = max_edge
        
        # Remove u,v from adj temporarily for simulation? 
        # Better to calculate the effect directly.
        
        # Effect of removing (u,v):
        # 1. Book of (u,v) disappears.
        # 2. Books of edges (u,w) and (v,w) decrease by 1 if w is a common neighbor of u and v?
        #    No, book of (u,w) is common neighbors of u and w.
        #    If v is a common neighbor of u and w, then removing (u,v) reduces book(u,w) by 1.
        #    Similarly for (v,w).
        #    Also, any edge (x,y) where u and v are both common neighbors? No, that's triangles.
        
        # This incremental update is complex. 
        # Given n=40, calculating max_book from scratch is O(n^3) ~ 64k ops.
        # 64k ops is very fast in Python/Numpy.
        # We can just try swaps and recalculate max_book.
        
        # We will try a limited number of swaps per iteration to save time.
        # But we want to be deterministic.
        
        # Let's try all possible non-edges as additions, paired with the removal of max_edge.
        # If we find one that improves, take it.
        # If not, try neutral moves that reduce variance.
        
        best_swap_score = current_score
        best_swap_adj = None
        best_swap_variance = float('inf')
        
        # Calculate current variance of book sizes (for tie-breaking)
        books_curr = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    books_curr.append(np.sum(adj[i] & adj[j]))
        var_curr = np.var(books_curr) if books_curr else 0
        
        # Try swaps
        for c in range(n):
            for d in range(c + 1, n):
                if c == u or d == u or c == v or d == v:
                    continue
                if adj[c, d]:
                    continue
                
                # Simulate swap: remove (u,v), add (c,d)
                adj[u, v] = False
                adj[v, u] = False
                adj[c, d] = True
                adj[d, c] = True
                
                new_score = max_book(adj)
                
                if new_score < best_swap_score:
                    best_swap_score = new_score
                    best_swap_adj = adj.copy()
                    best_swap_variance = np.var([np.sum(adj[i] & adj[j]) for i in range(n) for j in range(i+1, n) if adj[i,j]])
                    found_improvement = True
                elif new_score == best_swap_score:
                    # Check variance
                    new_books = [np.sum(adj[i] & adj[j]) for i in range(n) for j in range(i+1, n) if adj[i,j]]
                    new_var = np.var(new_books)
                    if new_var < best_swap_variance:
                        best_swap_variance = new_var
                        best_swap_adj = adj.copy()
                        if not found_improvement:
                            found_neutral = True
                            
                # Revert
                adj[u, v] = True
                adj[v, u] = True
                adj[c, d] = False
                adj[d, c] = False
                
        if best_swap_adj is not None:
            adj = best_swap_adj
            current_score = best_swap_score
            
            if current_score < best_score:
                best_score = current_score
                best_adj = adj.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        else:
            # No swap found that improves or even neutralizes with better variance?
            # This means we are in a local minimum with respect to single-edge swaps involving the max edge.
            # To escape, we might need to accept a worse move temporarily? 
            # Or just break if we've been stuck.
            no_improvement_count += 100 # Accelerate break
            
    # Return best_adj
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                result.append([u, v])
                
    return result
