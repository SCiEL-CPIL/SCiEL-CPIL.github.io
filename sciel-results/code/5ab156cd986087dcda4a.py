import random
import math
import sys

def construct():
    n = 40
    target_edges = 401
    
    # 1. Greedy Construction
    # We want to build a graph with 401 edges minimizing the max book size.
    # Strategy: Add edges one by one. At each step, evaluate all non-edges.
    # Score for adding edge (u,v):
    #   - The book size of (u,v) itself: b_uv = |N(u) & N(v)|
    #   - The increase in book sizes of existing edges (u,w) and (v,w) for w in N(u)&N(v).
    #   - We want to minimize the resulting MAX book size.
    
    adj = [set() for _ in range(n)]
    edges = set()
    
    # Precompute all possible edges
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # To speed up, we can just iterate through all possible edges and pick the best one each time.
    # "Best" means: the edge that, when added, results in the smallest possible maximum book size in the graph.
    
    # Optimization: 
    # Current max book size is M.
    # If we add (u,v), the new book size of (u,v) is b_uv.
    # The book size of any existing edge (u,w) where w is common neighbor increases by 1.
    # So the new max book size will be at least max(M, b_uv, max_{w in common} (book(u,w) + 1)).
    # We want to minimize this lower bound.
    
    # Let's maintain a list of current edges and their book sizes? 
    # Or just recompute max book size? 
    # Recomputing max book size for all edges is O(E*N). E=400, N=40 -> 16k ops.
    # Doing this for 800 candidates is 12M ops per step. 400 steps -> 4.8B ops. Too slow in Python.
    
    # Better Greedy:
    # Minimize the sum of squares of book sizes? Or just minimize the immediate book size b_uv, 
    # with a penalty if it exceeds current max?
    
    # Let's try a simpler greedy: 
    # Pick the edge (u,v) that minimizes |N(u) & N(v)|. 
    # If ties, pick one that minimizes the increase in sum of squares.
    
    # Actually, a known heuristic for minimizing max triangles per edge is to balance the degrees and avoid clustering.
    # Let's use a "Min-Conflict" style construction.
    
    # Revised Greedy:
    # Maintain current graph.
    # For each candidate edge (u,v) not in graph:
    #   b_uv = len(adj[u] & adj[v])
    #   # Check if adding this edge creates a book size > current_max_book?
    #   # If yes, penalty.
    
    # Let's just use a randomized greedy with look-ahead 1.
    # At each step, pick k random non-edges, evaluate their impact, pick the best.
    
    current_max_book = 0
    
    def get_book_sizes():
        books = {}
        for u, v in edges:
            books[(u, v)] = len(adj[u] & adj[v])
            # Also need to handle (v, u) if we store both? 
            # We store edges as (min, max). 
            # But adj is symmetric. 
            # The book size of edge (u,v) is the number of common neighbors.
            return books

    # This is still slow if we recompute for all candidates.
    
    # Alternative: 
    # Start with a random graph of 401 edges.
    # Then use a specialized Local Search that ONLY looks at edges with high book sizes.
    # This is different from the current SA which uses sum-of-squares cost.
    # The new cost function is strictly: Max Book Size.
    # Local Search: 
    # 1. Find edge e with max book B.
    # 2. Try to remove e.
    # 3. Try to add a non-edge f that minimizes the new max book size.
    # 4. Accept if new max < old max.
    
    # This "Min-Max" search is often more effective for this specific metric than sum-of-squares.
    
    rng = random.Random(123)
    
    # Initialize with a "balanced" random graph
    # To get a good start, we can try to make the graph as regular as possible.
    # 401 edges, 40 vertices. Average degree = 20.05.
    # Let's try to construct a graph close to 20-regular.
    
    # Simple random init:
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(all_possible)
    edges = set(all_possible[:target_edges])
    
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def compute_max_book():
        max_b = 0
        for u, v in edges:
            b = len(adj[u] & adj[v])
            if b > max_b:
                max_b = b
        return max_b

    current_max = compute_max_book()
    best_max = current_max
    best_edges = set(edges)
    
    # Local Search: Min-Max Swap
    # Iterate until no improvement or time limit.
    # Time limit: 100 seconds.
    
    import time
    start_time = time.time()
    timeout = 100.0
    
    # Precompute list of all non-edges for faster access? 
    # Or just generate random non-edges.
    
    # To be more effective, we should try to replace high-book edges with low-book non-edges.
    
    for i in range(500000): # High iteration count, but break on time
        if time.time() - start_time > timeout:
            break
            
        # Find an edge with max book size
        # We can cache book sizes? 
        # Let's just compute max book and pick one such edge.
        max_b = -1
        worst_edge = None
        for u, v in edges:
            b = len(adj[u] & adj[v])
            if b > max_b:
                max_b = b
                worst_edge = (u, v)
                
        if max_b == 0:
            break
            
        # Try to remove worst_edge
        u, v = worst_edge
        
        # Try a few random non-edges to add
        # We want a non-edge (x, y) such that:
        # 1. It doesn't create a book size > current max (ideally < current max)
        # 2. Removing (u,v) reduces the book size of other edges sufficiently.
        
        # Let's try 20 random non-edges
        improved = False
        for _ in range(20):
            x = rng.randrange(n)
            y = rng.randrange(n)
            if x == y:
                continue
            if x > y:
                x, y = y, x
            if (x, y) in edges:
                continue
                
            # Simulate adding (x,y) and removing (u,v)
            # Check if (x,y) creates a huge book
            b_xy = len(adj[x] & adj[y])
            
            # The new max book size will be at least b_xy.
            # Also, removing (u,v) decreases book sizes of edges (u,w) and (v,w) for w in common(u,v).
            # This might lower the max if the max was one of those.
            
            # Heuristic: Only accept if b_xy < current_max
            # Or if b_xy == current_max, we need to verify that removing (u,v) lowers the global max.
            
            if b_xy < current_max:
                # Apply change
                edges.discard((u, v))
                edges.add((x, y))
                adj[u].discard(v)
                adj[v].discard(u)
                adj[x].add(y)
                adj[y].add(x)
                
                new_max = compute_max_book()
                
                if new_max < current_max:
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_edges = set(edges)
                    improved = True
                    break
                else:
                    # Revert
                    edges.discard((x, y))
                    edges.add((u, v))
                    adj[x].discard(y)
                    adj[y].discard(x)
                    adj[u].add(v)
                    adj[v].add(u)
                    
            # If b_xy >= current_max, skip this candidate (likely won't improve max)
            
        if not improved and current_max == best_max:
            # If we are stuck at best_max, maybe try a different strategy?
            # For now, just continue searching for random improvements.
            pass

    return [[u, v] for u, v in best_edges]
