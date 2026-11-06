import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    n = 40
    m = 401
    
    # Initial construction: K20,20 + 1 edge
    # Vertices 0-19 in A, 20-39 in B
    edges = set()
    for a in range(20):
        for b in range(20, 40):
            edges.add((a, b))
    edges.add((0, 1))
    
    # Build adjacency matrix
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
        
    # Helper to get canonical edge tuple
    def canon(u, v):
        return (u, v) if u < v else (v, u)
        
    # Precompute non-edges for faster sampling? 
    # With 401 edges out of 780, non-edges are 379.
    # We can just sample and check.
    
    # Current state metrics
    # We need to track book sizes for all edges.
    # Book size of (u,v) = number of common neighbors.
    
    def compute_books():
        books = {}
        for u, v in edges:
            # Efficient common neighbors using bitsets or boolean array sum
            # adj[u] & adj[v] gives boolean array of common neighbors
            # We exclude u and v themselves? 
            # In a simple graph, u is not neighbor of u. 
            # Is v neighbor of u? Yes. Is u neighbor of v? Yes.
            # common neighbors of u and v are vertices w such that (u,w) in E and (v,w) in E.
            # adj[u] & adj[v] includes v (if u-v edge) and u (if v-u edge).
            # Wait. adj[u][v] is True. adj[v][v] is False. So v is not in adj[v].
            # So adj[u] & adj[v] will have 0 at index v.
            # Similarly 0 at index u.
            # So np.sum(adj[u] & adj[v]) is exactly the number of triangles containing edge (u,v).
            books[(u, v)] = np.sum(adj[u] & adj[v])
        return books

    current_books = compute_books()
    max_book = max(current_books.values()) if current_books else 0
    sum_sq = sum(v*v for v in current_books.values())
    
    best_max_book = max_book
    best_edges = set(edges)
    best_adj = adj.copy()
    best_books = dict(current_books)
    
    # Greedy Local Search
    # At each step, evaluate all possible single edge swaps?
    # There are ~400 edges to remove and ~380 non-edges to add.
    # 400 * 380 = 152,000 swaps.
    # Each swap evaluation requires updating book sizes for affected edges.
    # Affected edges are those incident to the 4 vertices involved (u, v, a, b).
    # Degree is ~20. So ~80 edges affected.
    # 152,000 * 80 = 12 million operations per full sweep.
    # In Python, this might be too slow for a full sweep every second.
    # 12M ops is ~1-2 seconds in Python.
    # So we can do ~50-100 full sweeps in 115s.
    # Each sweep is a greedy best-improvement step.
    
    # To speed up, we can randomize the order of evaluation and stop early if we find a huge improvement?
    # Or just do full sweeps.
    
    # Let's optimize the evaluation of a single swap.
    # Swap: remove (u,v), add (a,b).
    # Vertices involved: u, v, a, b.
    # Affected edges: all edges in the new graph incident to u, v, a, or b.
    # Note: (u,v) is removed, (a,b) is added.
    
    # We need to calculate the new book sizes for these affected edges.
    # And compare (max_book, sum_sq) before and after.
    
    # To make it faster, we can precompute the list of edges incident to each vertex.
    # But the graph changes.
    
    # Alternative: Randomized Greedy / Tabu
    # Pick a random set of candidate swaps (e.g., 1000) and pick the best one.
    # This is faster and often works well.
    
    # Let's try:
    # 1. Generate K random swaps.
    # 2. Evaluate them.
    # 3. Apply the best one.
    # 4. Repeat.
    
    K = 500  # Candidates per step
    
    while time.time() < deadline:
        best_delta_max = 0
        best_delta_sq = 0
        best_swap = None
        
        # Sample K swaps
        for _ in range(K):
            # Pick random edge to remove
            e_remove = random.choice(list(edges))
            u, v = e_remove
            
            # Pick random non-edge to add
            found = False
            for _try in range(10):
                a, b = random.sample(range(n), 2)
                if not adj[a, b]:
                    found = True
                    break
            if not found:
                continue
            
            # Evaluate swap (u,v) -> (a,b)
            # Temporarily apply? No, calculate delta.
            
            # Affected vertices: u, v, a, b
            # Edges affected:
            # Old affected: incident to u, v (excluding u-v)
            # New affected: incident to u, v (excluding u-v), incident to a, b (excluding a-b, but a-b is new)
            # Actually, any edge incident to u, v, a, or b might change its book size.
            
            affected = set()
            # Edges incident to u
            for z in range(n):
                if adj[u, z]:
                    affected.add(canon(u, z))
            # Edges incident to v
            for z in range(n):
                if adj[v, z]:
                    affected.add(canon(v, z))
            # Edges incident to a
            for z in range(n):
                if adj[a, z]:
                    affected.add(canon(a, z))
            # Edges incident to b
            for z in range(n):
                if adj[b, z]:
                    affected.add(canon(b, z))
            
            # Store old values
            old_vals = {}
            for e in affected:
                if e in current_books:
                    old_vals[e] = current_books[e]
            
            # Simulate change
            # Remove (u,v)
            adj[u, v] = False
            adj[v, u] = False
            # Add (a,b)
            adj[a, b] = True
            adj[b, a] = True
            
            # Calculate new values for affected edges
            new_vals = {}
            # Note: (u,v) is no longer an edge. (a,b) is now an edge.
            # We need to check if an edge is currently in `edges` to compute its book?
            # Book size is only defined for edges in `edges`.
            # So we only care about edges that are in the NEW edge set.
            
            # Let's update `edges` temporarily? 
            # Or just check:
            # If e == (u,v), it's removed.
            # If e == (a,b), it's added.
            # Others remain.
            
            # We need to compute book size for any e in affected that is in the new edge set.
            # New edge set = (edges - {(u,v)}) | {(a,b)}
            
            new_sum_sq = 0
            new_max = 0
            
            # We need to calculate the global max and sum_sq after change.
            # This is expensive if we do it from scratch.
            # But we only changed books for `affected` edges.
            # Unaffected edges keep their book sizes.
            
            # Let's maintain a list of all book sizes?
            # Or just compute the new max/sum_sq by:
            # 1. Start with old max/sum_sq.
            # 2. Subtract contributions of affected edges that are removed or changed.
            # 3. Add contributions of affected edges that are added or changed.
            
            # But `max` is hard to update incrementally if the max element changes.
            # However, if we just want to compare candidates, we can compute the new max for the affected set + old max of unaffected set.
            # Unaffected max is just the max of books for edges NOT in affected.
            # This requires knowing the second max, etc.
            
            # Simpler: Since K is small, and affected set is small (~80),
            # we can just recompute the max and sum_sq for the WHOLE graph?
            # No, that's 401 edges * 40 ops = 16k ops per candidate.
            # 500 candidates * 16k = 8M ops.
            # 8M ops in Python is ~0.5s.
            # So we can do ~200 steps in 115s.
            
            # Let's do this. It's simpler and less error-prone.
            
            # Wait, I already modified `adj`. I need to revert it after evaluation.
            
            # Compute new books for all edges?
            # That's expensive.
            
            # Let's stick to updating only affected edges and maintaining a heap?
            # Or just:
            # new_books_dict = current_books.copy()
            # for e in affected:
            #    if e in new_edges:
            #       new_books_dict[e] = np.sum(adj[e[0]] & adj[e[1]])
            #    else:
            #       if e in new_books_dict: del new_books_dict[e]
            
            # Then max(new_books_dict.values())
            
            # Let's try this.
            
            new_edges_set = set(edges)
            new_edges_set.discard((u, v))
            new_edges_set.add((a, b))
            
            # Update books for affected
            new_books = dict(current_books)
            
            for e in affected:
                if e in new_edges_set:
                    x, y = e
                    # Calculate new book size
                    # Note: adj is currently in "swapped" state
                    new_size = np.sum(adj[x] & adj[y])
                    new_books[e] = new_size
                else:
                    # Edge not in new graph
                    if e in new_books:
                        del new_books[e]
            
            # What if an edge NOT in affected had its book size changed?
            # No, book size of (x,y) depends on N(x) and N(y).
            # N(z) only changed for z in {u,v,a,b}.
            # So only edges incident to u,v,a,b are affected.
            # So `affected` covers all changed edges.
            
            # Calculate metrics
            if not new_books:
                continue
                
            new_max = max(new_books.values())
            new_sq = sum(v*v for v in new_books.values())
            
            # Revert adj
            adj[u, v] = True
            adj[v, u] = True
            adj[a, b] = False
            adj[b, a] = False
            
            # Compare
            # We want to minimize max_book, then sum_sq
            # delta_max = new_max - max_book
            # If new_max < max_book, it's an improvement.
            # If new_max == max_book, check sum_sq.
            
            if new_max < max_book:
                # Improvement
                if best_swap is None or new_max < best_delta_max or (new_max == best_delta_max and new_sq < best_delta_sq):
                    best_swap = (e_remove, (a, b))
                    best_delta_max = new_max
                    best_delta_sq = new_sq
            elif new_max == max_book:
                if new_sq < sum_sq:
                    if best_swap is None or (best_delta_max == max_book and new_sq < best_delta_sq):
                        best_swap = (e_remove, (a, b))
                        best_delta_max = new_max
                        best_delta_sq = new_sq
            else:
                # Worse max book, ignore unless we allow moves? 
                # Greedy descent usually ignores worse max.
                pass
        
        if best_swap is None:
            # No improvement found in this batch
            # Try a larger batch or break?
            # Let's just break to save time, or randomize restart?
            # Given time constraint, break.
            break
            
        # Apply best swap
        e_remove, e_add = best_swap
        u, v = e_remove
        a, b = e_add
        
        # Update state
        adj[u, v] = False
        adj[v, u] = False
        adj[a, b] = True
        adj[b, a] = True
        
        edges.discard(e_remove)
        edges.add(e_add)
        
        # Update books
        # Recompute affected books
        affected = set()
        for z in range(n):
            if adj[u, z]: affected.add(canon(u, z))
            if adj[v, z]: affected.add(canon(v, z))
            if adj[a, z]: affected.add(canon(a, z))
            if adj[b, z]: affected.add(canon(b, z))
            
        # We need to update current_books
        # Remove books for e_remove
        if e_remove in current_books:
            del current_books[e_remove]
            
        # Add/Update books for affected edges in new edge set
        for e in affected:
            if e in edges:
                x, y = e
                current_books[e] = np.sum(adj[x] & adj[y])
                
        # Note: e_add is in affected (incident to a or b) and in edges, so it's added.
        # Other affected edges are updated.
        
        # Update global metrics
        max_book = best_delta_max
        sum_sq = best_delta_sq
        
        # Check if this is a new best
        if max_book < best_max_book:
            best_max_book = max_book
            best_edges = set(edges)
            best_adj = adj.copy()
            best_books = dict(current_books)
            
    # Return best found
    # Reconstruct edges from best_edges
    result = [[u, v] for u, v in best_edges]
    
    # Sanity check
    if len(result) != 401:
        return [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return result
