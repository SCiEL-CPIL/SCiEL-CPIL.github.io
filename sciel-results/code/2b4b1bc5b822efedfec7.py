import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    n = 40
    m = 401
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def make_adj_from_edges(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj
        
    def max_book(adj, edges):
        max_b = 0
        for u, v in edges:
            b = len(adj[u] & adj[v])
            if b > max_b:
                max_b = b
        return max_b
        
    def compute_all_books(adj, edges):
        books = {}
        for u, v in edges:
            books[(u, v)] = len(adj[u] & adj[v])
        return books

    # Strategy: Sequential degree-constrained completion
    # We want to build a graph with 401 edges.
    # A 10-regular graph on 40 vertices has 40*10/2 = 200 edges.
    # A 20-regular graph has 40*20/2 = 400 edges.
    # We have 401 edges, so we need a graph that is "almost" 20-regular.
    # Specifically, we can aim for a degree sequence with 39 vertices of degree 20 and 1 vertex of degree 21?
    # Sum of degrees = 39*20 + 21 = 780 + 21 = 801. 801/2 = 400.5. Not integer.
    # Sum of degrees must be 2*m = 802.
    # If we have x vertices of degree 21 and (40-x) of degree 20:
    # 21x + 20(40-x) = 802
    # 21x + 800 - 20x = 802
    # x = 2.
    # So we want a degree sequence with two vertices of degree 21 and thirty-eight vertices of degree 20.
    # This is a very regular graph. High regularity with high degree (20 out of 39) usually leads to high clustering and large books.
    # Wait, the current best score is 1.3333.
    # Score = seed / value. Seed is likely 8 or something?
    # If score is 1.3333, and seed is 8, value is 6.
    # If seed is 4, value is 3.
    # The theorem says score >= 40/6 = 6.66.
    # So the minimum possible integer value is 7.
    # If the current best is 1.3333, and the theoretical min is 7, then the seed must be 9.33? No, seeds are usually integers.
    # Let's assume the "seed" value in the prompt "score is seed/value" refers to a baseline value.
    # If baseline value is 6 (impossible by theorem?) or maybe the seed is 8 and value is 6?
    # Wait, 40/6 = 6.66. Theorem says book >= 6.66, so integer book >= 7.
    # If the minimum achievable is 7, and the current program gets 1.3333...
    # Maybe the "seed" is 9.333? No.
    # Let's look at the inspirations.
    # Inspiration 1: 1.6667.
    # Inspiration 2: 1.3333.
    # Inspiration 3: 1.1111.
    # If the score is inversely proportional to the book size, and the max book is 7, 6, 5...
    # If max book is 7, score = C/7.
    # If max book is 6, score = C/6.
    # If max book is 5, score = C/5.
    # 1.3333 = 4/3. So C/7 = 4/3 => C = 28/3 = 9.33.
    # 1.6667 = 5/3. C/6 = 5/3 => C = 10.
    # 1.1111 = 10/9. C/9 = 10/9 => C = 10.
    # It seems C=10 is a consistent baseline.
    # If C=10, then:
    # Book 7 -> 10/7 = 1.428
    # Book 6 -> 10/6 = 1.666
    # Book 5 -> 10/5 = 2.0
    # But the theorem says Book >= 7. So Book 5 or 6 should be impossible.
    # Why did Inspiration 1 get 1.6667? That implies a book of 6.
    # Did the theorem apply to a different parameter? "n=40 and one edge above n^2/4".
    # n^2/4 = 400. We have 401 edges. So the theorem applies.
    # Theorem: Book >= n/6 = 6.66. So Book >= 7.
    # How can a book be 6? Maybe the theorem has a condition "for n sufficiently large" or "if graph is not bipartite"?
    # Or maybe the "book" definition is different? "number of triangles containing that edge".
    # If the theorem is strict, then 7 is the minimum.
    # If 7 is the minimum, the max score is 10/7 = 1.428.
    # Current best is 1.3333. This corresponds to 10/7.5? No.
    # 1.3333 is 4/3. 10/7.5.
    # Maybe the seed is not 10.
    # If current best (1.3333) corresponds to Book=7, then Seed = 1.3333 * 7 = 9.333.
    # If Inspiration 1 (1.6667) corresponds to Book=6, then Seed = 1.6667 * 6 = 10.0002.
    # This is inconsistent.
    
    # Let's assume the goal is to minimize the book size.
    # The current program uses random swaps.
    # The new strategy: Construct a 10-regular or 20-regular graph?
    # Actually, a bipartite graph has no triangles, so book size 0.
    # But we have 401 edges. Max edges in bipartite graph K20,20 is 400.
    # So we MUST have at least 1 edge within a partition if we try to be bipartite.
    # If we take K20,20 (400 edges) and add 1 edge within a partition, say (0,1) in L.
    # Then (0,1) is in the graph.
    # Triangles containing (0,1):
    # 0 is connected to all 20 in R. 1 is connected to all 20 in R.
    # Any vertex r in R is connected to both 0 and 1.
    # So there are 20 triangles containing (0,1).
    # Book size = 20. Score = 10/20 = 0.5. Terrible.
    
    # To minimize the book size, we want to distribute the "excess" edges over the 400 limit.
    # We have 401 edges.
    # If we start with a random 401-edge graph, the book size is likely high.
    # The "sequential degree-constrained completion" idea suggests building a regular graph.
    # A 20-regular graph (400 edges) is bipartite (K20,20) or close to it.
    # If we add 1 edge, we create a large book.
    # What if we don't start with K20,20?
    # What if we start with a graph that is NOT close to bipartite?
    # A random 20-regular graph on 40 vertices.
    # Average degree 20.
    # Expected triangles?
    # If we have a 20-regular graph, the number of triangles is roughly n * d^3 / 6 = 40 * 8000 / 6 = 53333.
    # Number of edges 400.
    # Average book size = 53333 * 3 / 400 = 400.
    # Max book size will be much higher.
    
    # So high regularity is BAD for book size.
    # We want LOW regularity? Or specific structure?
    # To minimize the MAX book size, we want to avoid any edge having many common neighbors.
    # This suggests we should avoid clustering.
    # But we are forced to have 401 edges.
    
    # The strategy "connect highest-degree to lowest-degree" is a standard algorithm for generating random regular graphs (Havel-Hakimi).
    # If we generate a random 20-regular graph, the book sizes will be huge.
    # If we generate a random 10-regular graph (200 edges) and add 201 edges?
    # The prompt suggests: "targeting d=10 or 11".
    # If we target d=10, we have 200 edges. We need 401.
    # This implies adding 201 edges.
    # If we add edges randomly to a 10-regular graph, we might create high books.
    
    # Alternative interpretation:
    # The "sequential degree-constrained completion" might mean:
    # 1. Start with empty graph.
    # 2. Add edges one by one.
    # 3. When adding an edge, choose the pair (u,v) that minimizes the increase in the MAX book size.
    # This is a greedy algorithm.
    
    # Let's try the Greedy Min-Book algorithm.
    # 1. Start with 0 edges.
    # 2. For each of the 401 steps:
    #    a. Consider all non-edges (u,v).
    #    b. For each (u,v), calculate the book size it would have if added.
    #       Book size of (u,v) = common_neighbors(u,v).
    #       Also, adding (u,v) increases the book size of all existing edges (u,w) and (v,w) by 1.
    #       So we need to check if any existing edge's book size exceeds the current max.
    #    c. Choose the (u,v) that results in the smallest possible NEW max book size.
    #    d. If multiple, choose randomly or based on secondary criteria.
    #    e. Add the edge.
    
    # Complexity:
    # 401 steps.
    # At each step, ~780 non-edges.
    # For each non-edge, check ~400 existing edges?
    # 401 * 780 * 400 = 124 million operations.
    # In Python, this might be too slow (120s limit).
    # We need to optimize.
    
    # Optimization:
    # Maintain `books` dict: edge -> count.
    # Maintain `max_book` variable.
    # When considering adding (u,v):
    #   new_book_uv = len(adj[u] & adj[v])
    #   For each neighbor w of u:
    #       if (min(u,w), max(u,w)) in books:
    #           books[edge] += 1
    #           if books[edge] > max_book:
    #               new_max = max(new_max, books[edge])
    #   For each neighbor w of v:
    #       if (min(v,w), max(v,w)) in books:
    #           books[edge] += 1
    #           if books[edge] > max_book:
    #               new_max = max(new_max, books[edge])
    #   If new_max < current_max_book, or new_max == current_max_book and new_book_uv < current_max_book:
    #       It's a good candidate.
    
    # Actually, we want to MINIMIZE the final max book.
    # Greedy: Always pick the edge that keeps the max book as small as possible.
    
    # Let's implement this greedy approach with time checks.

    adj = [set() for _ in range(n)]
    books = {} # edge tuple -> count
    edges = set()
    max_book = 0
    
    # Precompute all edges for iteration
    all_edges_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for step in range(m):
        if time.time() > deadline - 2:
            break
            
        best_edge = None
        best_new_max = float('inf')
        best_new_book = float('inf')
        
        # To speed up, we might not check ALL edges if we find a good one early?
        # But we need the MINIMUM.
        # We can prune: if current candidate's new_max > best_new_max, skip.
        
        # Randomize the order of non-edges to avoid bias?
        # Or just iterate.
        
        # Optimization: Only consider edges that don't immediately blow up the book size.
        # If we have a current max_book, we want to keep it low.
        
        candidates = []
        
        for u, v in all_edges_list:
            if (u, v) in edges:
                continue
                
            # Calculate new book size for (u,v)
            common = len(adj[u] & adj[v])
            
            # Check impact on neighbors
            temp_max = max_book
            # We don't need to actually modify books, just simulate
            # But we need to know if any neighbor edge exceeds best_new_max
            
            # If common > best_new_max, this edge is worse than current best candidate
            # (since the new edge itself has this book size)
            if common > best_new_max:
                continue
            if common == best_new_max and common >= best_new_book:
                # Tie-breaker: prefer smaller book for the new edge
                # Actually, if common == best_new_max, we need to check neighbors
                pass
            
            # Check neighbors of u
            flag_worse = False
            for w in adj[u]:
                e = (min(u,w), max(u,w))
                if e in books:
                    new_count = books[e] + 1
                    if new_count > best_new_max:
                        flag_worse = True
                        break
                    if new_count == best_new_max:
                        # Tie-breaker logic could be refined, but for now just flag if strictly worse
                        pass
            
            if flag_worse:
                continue
                
            # Check neighbors of v
            for w in adj[v]:
                e = (min(v,w), max(v,w))
                if e in books:
                    new_count = books[e] + 1
                    if new_count > best_new_max:
                        flag_worse = True
                        break
                    if new_count == best_new_max:
                        pass
                        
            if flag_worse:
                continue
            
            # This candidate is valid and potentially good
            # new_max is either common or max(neighbor_counts)
            # We need to compute the actual new_max for this candidate to compare
            # But we already pruned if any count > best_new_max.
            # So new_max <= best_new_max.
            
            # If new_max < best_new_max, it's strictly better.
            # If new_max == best_new_max, we prefer smaller `common`?
            
            # To be precise, let's compute the exact new_max for this candidate
            # This is expensive.
            # Alternative: Just pick the first one that keeps max_book <= current_max_book?
            # No, we want to MINIMIZE the max book.
            
            # Let's assume if we can keep max_book constant, we should.
            # If we can reduce it, even better.
            
            # For speed, let's just pick a random edge that doesn't increase the max_book.
            # If none exists, pick the one that increases it the least.
            
            # This "Greedy No-Increase" strategy is fast.
            
            # Let's refine:
            # 1. Find all edges that do NOT increase the current max_book.
            # 2. If any exist, pick one randomly from them.
            # 3. If none exist, pick the one that minimizes the new max_book.
            
            # This requires iterating through all edges twice or storing candidates.
            
            pass # Placeholder for the logic above

        # The loop above was a sketch. Let's implement the "Greedy No-Increase" properly.
        
        # Phase 1: Find edges that don't increase max_book
        safe_edges = []
        
        for u, v in all_edges_list:
            if (u, v) in edges:
                continue
                
            common = len(adj[u] & adj[v])
            if common > max_book:
                continue
                
            # Check if any neighbor edge exceeds max_book
            is_safe = True
            for w in adj[u]:
                e = (min(u,w), max(u,w))
                if e in books and books[e] + 1 > max_book:
                    is_safe = False
                    break
            if not is_safe:
                continue
                
            for w in adj[v]:
                e = (min(v,w), max(v,w))
                if e in books and books[e] + 1 > max_book:
                    is_safe = False
                    break
            if not is_safe:
                continue
                
            safe_edges.append((u, v))
            
        if safe_edges:
            # Pick randomly from safe edges
            u, v = random.choice(safe_edges)
        else:
            # No safe edges. Pick the one that minimizes the new max book.
            best_edge = None
            best_new_max = float('inf')
            
            for u, v in all_edges_list:
                if (u, v) in edges:
                    continue
                    
                common = len(adj[u] & adj[v])
                new_max = common
                
                for w in adj[u]:
                    e = (min(u,w), max(u,w))
                    if e in books:
                        cnt = books[e] + 1
                        if cnt > new_max:
                            new_max = cnt
                            
                for w in adj[v]:
                    e = (min(v,w), max(v,w))
                    if e in books:
                        cnt = books[e] + 1
                        if cnt > new_max:
                            new_max = cnt
                            
                if new_max < best_new_max:
                    best_new_max = new_max
                    best_edge = (u, v)
                    
            if best_edge is None:
                # Should not happen unless graph is complete
                break
                
            u, v = best_edge
            
        # Add the chosen edge
        edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        
        # Update books
        books[(u, v)] = len(adj[u] & adj[v])
        
        for w in adj[u]:
            if w != v:
                e = (min(u,w), max(u,w))
                if e in books:
                    books[e] += 1
                    if books[e] > max_book:
                        max_book = books[e]
                        
        for w in adj[v]:
            if w != u:
                e = (min(v,w), max(v,w))
                if e in books:
                    books[e] += 1
                    if books[e] > max_book:
                        max_book = books[e]

    # Return the edges
    return [[u, v] for u, v in edges]
