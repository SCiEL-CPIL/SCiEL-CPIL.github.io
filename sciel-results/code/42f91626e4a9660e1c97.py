import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin for 120s limit
    
    n = 40
    m = 401
    
    # Initial construction: 
    # A good starting point for minimizing max triangles is a graph that is 
    # "as far from cliques as possible". 
    # K20,20 has no triangles. Adding edges creates triangles.
    # If we start with K20,20 + 1 edge, the 1 edge in A has book size 20 (connected to all B).
    # Max book is 20. This is high.
    # A better start: A random graph G(40, 401/780 ~ 0.51). 
    # Expected triangles ~ C(40,3) * p^3 ~ 9880 * 0.13 ~ 1280.
    # Total edges 401. Average book size ~ 1280*3/401 ~ 9.5.
    # Max will be higher, maybe 15-20.
    
    # Let's try a quasi-random start: 
    # Use a modular construction or just a random graph with a fixed seed for reproducibility 
    # but optimized for lower initial max.
    
    # Actually, let's use a specific construction: 
    # The theorem says score >= 40/6 ~ 6.67. So max >= 7.
    # We want to get close to 7.
    
    # Strategy: Start with a random graph, then aggressively reduce max book.
    
    # Initialize with a random graph G(n, p) where m=401.
    # p = 401 / (n*(n-1)/2) = 401 / 780 ~ 0.514
    rng = np.random.default_rng(42)
    
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    # Randomly select 401 edges
    initial_edges = rng.choice(len(all_edges), size=m, replace=False)
    edges = set(all_edges[i] for i in initial_edges)
    
    adj = np.zeros((n, n), dtype=np.int8)
    for u, v in edges:
        adj[u, v] = 1
        adj[v, u] = 1
        
    # Compute Book Sizes
    # C[u, v] = number of common neighbors.
    # Book size of edge (u,v) is C[u,v].
    
    def get_books(adj):
        C = adj @ adj
        books = {}
        for u in range(n):
            for v in range(u+1, n):
                if adj[u, v]:
                    books[(u, v)] = C[u, v]
        return books, C

    books, C = get_books(adj)
    current_max_book = max(books.values())
    current_sum_sq = sum(b*b for b in books.values())
    
    # Track best
    best_max_book = current_max_book
    best_sum_sq = current_sum_sq
    best_adj = adj.copy()
    best_edges = set(edges)
    
    # Local Search: Greedy Best Improvement
    # We look for a swap (remove e1, add e2) that reduces max_book, 
    # or if max_book is unchanged, reduces sum_sq.
    
    # To make this efficient:
    # 1. Identify "hot" edges: edges with book size == current_max_book.
    # 2. Identify "cold" non-edges: non-edges that, if added, would create few triangles.
    #    A non-edge (a,b) creates C[a,b] triangles.
    # 3. Try swapping a hot edge (u,v) with a cold non-edge (a,b).
    
    # Precompute list of non-edges
    all_possible = set(all_edges)
    non_edges = all_possible - edges
    
    steps = 0
    no_improvement_count = 0
    
    while time.time() < deadline:
        steps += 1
        
        # Identify hot edges
        hot_edges = [e for e, b in books.items() if b == current_max_book]
        
        if not hot_edges:
            break # Should not happen if current_max_book is correct
            
        # If we are stuck at a high max, we might need to accept a temporary increase 
        # in sum_sq to reduce max. 
        # But pure greedy on max might get stuck if reducing max requires increasing 
        # sum_sq significantly. 
        # Let's stick to: Reduce Max. If tie, reduce SumSq.
        
        # To avoid O(H * N_non) which is too slow, we sample.
        # But n=40 is small. 
        # Max hot edges: maybe 10-50.
        # Max non-edges: ~379.
        # 50 * 379 = ~20k evaluations per step. 
        # Each evaluation involves updating C. 
        # C update is O(n^2) if naive. 1600 ops. 
        # 20k * 1600 = 32M ops. Too slow in Python loop.
        
        # Optimization: 
        # We don't need to recompute full C for every candidate.
        # We can estimate the change.
        
        # Let's try a smarter heuristic:
        # For each hot edge (u,v), we want to remove it.
        # Removing (u,v) decreases book sizes of all edges (u,w) and (v,w) for w in N(u) U N(v) \ {u,v}.
        # Specifically, for an edge (u,w), its book size decreases by 1 if w is connected to v (i.e., w is in N(u) intersect N(v) is not right).
        # Book(u,w) = |N(u) intersect N(w)|.
        # If we remove (u,v), then v is no longer in N(u). 
        # So for any w, if v in N(w), then Book(u,w) decreases by 1.
        # Similarly, if u in N(w), Book(v,w) decreases by 1.
        
        # Adding (a,b):
        # Increases Book(a,b) to C[a,b] (new edge).
        # For any w, if a in N(w), Book(b,w) increases by 1.
        # For any w, if b in N(w), Book(a,w) increases by 1.
        
        # This local delta calculation is O(deg(u)+deg(v)+deg(a)+deg(b)).
        # This is much faster.
        
        # Let's implement a function to calculate the new max and sum_sq given a swap,
        # using only local updates to the books dictionary.
        
        def evaluate_swap(remove_e, add_e):
            u, v = remove_e
            a, b = add_e
            
            # Check if add_e is already an edge (shouldn't be)
            if add_e in edges:
                return None
            
            # 1. Calculate impact on existing books
            # Books affected by removal of (u,v):
            # - (u,v) itself is removed.
            # - Edges (u,w) where w in N(v) (and w != u)
            # - Edges (v,w) where w in N(u) (and w != v)
            
            # Books affected by addition of (a,b):
            # - (a,b) itself is added with size C[a,b]
            # - Edges (a,w) where w in N(b) (and w != a)
            # - Edges (b,w) where w in N(a) (and w != b)
            
            # We need to check if the new max is lower.
            
            # Let's just simulate the change on the `books` dict copy? 
            # That's O(E) ~ 400. 
            # If we do this for ~1000 candidates per step, that's 400k ops. 
            # 1000 steps * 400k = 400M ops. Might be too slow in Python.
            
            # Let's restrict candidates.
            # Only consider adding edges (a,b) where C[a,b] is small.
            # C[a,b] is the number of triangles formed.
            # If C[a,b] > current_max_book, it will definitely increase the max (unless we remove the max edge and it was the only one).
            # So we should prefer adding edges with low C[a,b].
            
            # Sort non-edges by C[a,b] (ascending).
            # We need C for non-edges.
            # C is already computed as adj @ adj.
            
            # Let's pick the top K "best" non-edges to add (lowest C).
            # And top K "worst" edges to remove (highest Book).
            
            # Get C for all non-edges
            # C is n x n.
            # non_edges list.
            
            # To speed up, let's just sample a few candidates.
            # 1. Pick a random hot edge to remove.
            # 2. Pick 5 random non-edges with the lowest C values.
            
            # We need to maintain a list of non-edges sorted by C?
            # C changes after every swap.
            # Recalculating C for all non-edges is O(N_non * n) ~ 380 * 40 = 15k. Fast.
            
            # Let's do this:
            # 1. Compute C = adj @ adj.
            # 2. Create list of non-edges.
            # 3. Sort non-edges by C[u,v].
            # 4. Take top 10 non-edges.
            # 5. Take top 10 hot edges.
            # 6. Try all 10x10 = 100 swaps.
            # 7. For each swap, calculate new max and sum_sq.
            
            # How to calculate new max/sum_sq efficiently for 100 swaps?
            # We can precompute the "delta" for each edge.
            
            # Actually, let's just recompute books for the candidate graph.
            # It's only 401 edges. 
            # 100 candidates * 401 edges * (small constant) = 40k ops.
            # This is very fast.
            
            # Let's implement a fast book calculator.
            
            def calc_stats(adj_mat, edge_set):
                C_loc = adj_mat @ adj_mat
                max_b = 0
                sum_sq = 0
                for u, v in edge_set:
                    b = C_loc[u, v]
                    if b > max_b:
                        max_b = b
                    sum_sq += b*b
                return max_b, sum_sq

            # Get current C
            C_curr = adj @ adj
            
            # Prepare candidate non-edges
            # We need to filter non_edges that are not in edges.
            # And get their C values.
            # non_edges is a set.
            # Let's convert to list.
            
            # To save time, we don't sort all 380 non-edges every step.
            # We just sample 20 random non-edges and pick the ones with lowest C.
            
            candidates = []
            # Sample 50 random non-edges
            for _ in range(50):
                if not non_edges: break
                # Pick random from non_edges
                # random.choice on set is slow. 
                # Let's maintain a list of non-edges.
                pass
            
            # Better: Maintain a list `non_edge_list` and `edge_list`.
            # Update them on swap.
            
            # Let's refactor to use lists.
            
            # Re-initialize lists if this is the first iteration or if we are switching strategy
            if 'non_edge_list' not in locals():
                non_edge_list = list(non_edges)
                edge_list = list(edges)
            
            # Shuffle non_edge_list to avoid bias
            random.shuffle(non_edge_list)
            
            # Pick top 10 from non_edge_list based on C_curr
            # We need C_curr for these.
            
            # Get hot edges
            hot_edges_list = [e for e in edge_list if books.get(e, 0) == current_max_book]
            random.shuffle(hot_edges_list)
            hot_candidates = hot_edges_list[:10]
            
            # Get cold non-edges
            # We'll just take the first 20 from the shuffled list and filter by C
            cold_candidates = []
            for e in non_edge_list[:50]:
                a, b = e
                if C_curr[a, b] < current_max_book: # Only consider if it doesn't immediately exceed current max
                    cold_candidates.append(e)
                if len(cold_candidates) >= 10:
                    break
            
            if not cold_candidates:
                # If no cold candidates, just pick random non-edges
                cold_candidates = non_edge_list[:10]
                
            # Try swaps
            best_swap = None
            best_new_max = current_max_book
            best_new_sum = current_sum_sq
            
            for e_rm in hot_candidates:
                for e_add in cold_candidates:
                    # Check if e_add is in edges (should not be)
                    if e_add in edges: continue
                    
                    # Simulate swap
                    # Create new edge set
                    new_edges = set(edges)
                    new_edges.discard(e_rm)
                    new_edges.add(e_add)
                    
                    # Create new adj matrix? 
                    # Or just update the existing one?
                    # Updating adj is faster.
                    
                    u, v = e_rm
                    a, b = e_add
                    
                    # Temporarily modify adj
                    adj[u, v] = 0
                    adj[v, u] = 0
                    adj[a, b] = 1
                    adj[b, a] = 1
                    
                    # Calculate stats
                    new_max, new_sum = calc_stats(adj, new_edges)
                    
                    # Revert adj
                    adj[u, v] = 1
                    adj[v, u] = 1
                    adj[a, b] = 0
                    adj[b, a] = 0
                    
                    # Compare
                    if new_max < best_new_max or (new_max == best_new_max and new_sum < best_new_sum):
                        best_new_max = new_max
                        best_new_sum = new_sum
                        best_swap = (e_rm, e_add)
                        
            if best_swap:
                e_rm, e_add = best_swap
                u, v = e_rm
                a, b = e_add
                
                # Apply swap
                adj[u, v] = 0
                adj[v, u] = 0
                adj[a, b] = 1
                adj[b, a] = 1
                
                edges.discard(e_rm)
                edges.add(e_add)
                
                # Update lists
                # Remove e_rm from edge_list, add to non_edge_list
                # Add e_add to edge_list, remove from non_edge_list
                # O(N) operations, fine for N=400.
                
                try:
                    edge_list.remove(e_rm)
                except ValueError:
                    pass
                non_edge_list.append(e_rm)
                
                try:
                    non_edge_list.remove(e_add)
                except ValueError:
                    pass
                edge_list.append(e_add)
                
                # Update books and C
                books, C = get_books(adj)
                current_max_book = best_new_max
                current_sum_sq = best_new_sum
                
                # Update best
                if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                    best_max_book = current_max_book
                    best_sum_sq = current_sum_sq
                    best_adj = adj.copy()
                    best_edges = set(edges)
                    
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                # If stuck for too many steps, random restart or perturbation
                if no_improvement_count > 50:
                    # Random perturbation: swap 2 random edges
                    # This helps escape local minima
                    if edge_list and non_edge_list:
                        e1 = random.choice(edge_list)
                        e2 = random.choice(non_edge_list)
                        
                        u, v = e1
                        a, b = e2
                        
                        adj[u, v] = 0
                        adj[v, u] = 0
                        adj[a, b] = 1
                        adj[b, a] = 1
                        
                        edges.discard(e1)
                        edges.add(e2)
                        
                        edge_list.remove(e1)
                        non_edge_list.append(e1)
                        non_edge_list.remove(e2)
                        edge_list.append(e2)
                        
                        books, C = get_books(adj)
                        current_max_book = max(books.values())
                        current_sum_sq = sum(b*b for b in books.values())
                        
                        if current_max_book < best_max_book or (current_max_book == best_max_book and current_sum_sq < best_sum_sq):
                            best_max_book = current_max_book
                            best_sum_sq = current_sum_sq
                            best_adj = adj.copy()
                            best_edges = set(edges)
                            
                        no_improvement_count = 0

    # Return best found
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_adj[i, j]:
                result.append([i, j])
                
    # Sanity check
    if len(result) != 401:
        # Fallback
        result = [[a, b] for a in range(20) for b in range(20, 40)] + [[0, 1]]
        
    return result
