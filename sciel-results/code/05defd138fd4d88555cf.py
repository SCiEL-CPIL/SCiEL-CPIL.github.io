import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = 40
    target_edges = 401
    
    def evaluate_adj(adj):
        """Compute the maximum book size (max triangles per edge)."""
        max_b = 0
        # Iterate over upper triangle
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    # Common neighbors of u and v
                    # adj[u] & adj[v] gives a vector where both are True
                    # Exclude u and v themselves (they can't be neighbors of themselves in a simple graph)
                    common = adj[u] & adj[v]
                    common[u] = False
                    common[v] = False
                    b = int(np.sum(common))
                    if b > max_b:
                        max_b = b
        return max_b

    def local_search(adj, time_limit):
        """
        Perform local search on the given adjacency matrix.
        Uses a 'random swap' strategy but optimized with incremental book size updates.
        Returns the best adjacency matrix and score found.
        """
        current_score = evaluate_adj(adj)
        best_score = current_score
        best_adj = adj.copy()
        
        # Precompute degrees for potential use, though not strictly needed for basic swap
        # For speed, we will just re-evaluate the whole graph or use a fast approximation?
        # Given n=40, full evaluation is ~800 ops. 110s / 800 ops is plenty.
        # The bottleneck in the previous program was likely the random sampling retries or Python overhead.
        
        present_edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    present_edges.add((u, v))
        
        # To speed up, we can pre-compute the book size for every edge.
        # book[u, v] = number of triangles containing edge (u, v)
        # This allows O(1) lookup of the current max book size if we maintain a histogram.
        
        # Let's implement a more efficient local search using a histogram of book sizes.
        # hist[k] = number of edges with book size k
        
        def compute_book_sizes(adj_mat):
            books = np.zeros((n, n), dtype=int)
            for u in range(n):
                for v in range(u + 1, n):
                    if adj_mat[u, v]:
                        common = adj_mat[u] & adj_mat[v]
                        common[u] = False
                        common[v] = False
                        books[u, v] = int(np.sum(common))
                        books[v, u] = books[u, v]
            return books
        
        books = compute_book_sizes(adj)
        
        # Build histogram of book sizes for existing edges
        # Max possible book size is n-2 = 38
        hist = np.zeros(n, dtype=int)
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    b = books[u, v]
                    if b < n:
                        hist[b] += 1
        
        def get_max_book(hist_vec):
            # Find the highest index with non-zero count
            non_zero = np.nonzero(hist_vec)[0]
            if len(non_zero) == 0:
                return 0
            return non_zero[-1]
            
        current_max_book = get_max_book(hist)
        
        # Local search loop
        # We will try to improve by swapping edges.
        # A swap (remove e1, add e2) changes book sizes for:
        # 1. e1 itself (removed)
        # 2. e2 itself (added)
        # 3. All edges sharing an endpoint with e1 or e2
        
        # This incremental update is complex. Given the small size (n=40), 
        # a full re-evaluation might be competitive if the Python overhead is minimized.
        # However, the previous program was slow. Let's try to optimize the full re-evaluation 
        # or use a simpler heuristic.
        
        # Alternative: Use a "Greedy + Random Restart" approach with a faster evaluation.
        # Let's stick to a robust local search but with better move generation.
        
        # Move: Pick a random edge to remove, pick a random non-edge to add.
        # To avoid retries, maintain a list of present and absent edges.
        
        present_list = list(present_edges)
        all_possible = [(u, v) for u in range(n) for v in range(u + 1, n)]
        absent_set = set(all_possible) - present_set
        absent_list = list(absent_set)
        
        T = 1.0
        T_min = 0.01
        alpha = 0.999
        
        while time.time() < time_limit:
            if not present_list or not absent_list:
                break
                
            # Random swap
            idx_out = random.randint(0, len(present_list) - 1)
            idx_in = random.randint(0, len(absent_list) - 1)
            
            e_out = present_list[idx_out]
            e_in = absent_list[idx_in]
            
            # Apply move
            u1, v1 = e_out
            u2, v2 = e_in
            
            adj[u1, v1] = False
            adj[v1, u1] = False
            adj[u2, v2] = True
            adj[v2, u2] = True
            
            # Update lists
            present_list[idx_out] = e_in
            absent_list[idx_in] = e_out
            present_edges.discard(e_out)
            present_edges.add(e_in)
            absent_set.discard(e_in)
            absent_set.add(e_out)
            
            # Evaluate new score
            # For n=40, full evaluation is fast enough if vectorized.
            new_score = evaluate_adj(adj)
            
            delta = new_score - current_score
            
            # Simulated Annealing acceptance
            if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < np.exp(-delta / T)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = adj.copy()
            else:
                # Revert
                adj[u1, v1] = True
                adj[v1, u1] = True
                adj[u2, v2] = False
                adj[v2, u2] = False
                
                present_list[idx_out] = e_out
                absent_list[idx_in] = e_in
                present_edges.discard(e_in)
                present_edges.add(e_out)
                absent_set.discard(e_out)
                absent_set.add(e_in)
            
            T *= alpha
            if T < T_min:
                T = T_min
                
        return best_adj, best_score

    # Multi-start strategy
    # We will run several different initializations and local searches.
    # Time budget: 110s total.
    # Let's allocate time to multiple runs.
    
    best_overall_adj = None
    best_overall_score = float('inf')
    
    # Strategy 1: Bipartite + Perturbation
    # Start with K_20,20 + 1 edge
    adj1 = np.zeros((n, n), dtype=bool)
    for u in range(20):
        for v in range(20, 40):
            adj1[u, v] = True
            adj1[v, u] = True
    adj1[0, 1] = True
    adj1[1, 0] = True
    
    # Strategy 2: Random Graph
    adj2 = np.zeros((n, n), dtype=bool)
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_pairs)
    for u, v in all_pairs[:target_edges]:
        adj2[u, v] = True
        adj2[v, u] = True
        
    # Strategy 3: Block Diagonal (e.g., 4 blocks of 10)
    adj3 = np.zeros((n, n), dtype=bool)
    # Make 4 complete graphs K_10? That would be 4 * 45 = 180 edges. Need 401.
    # Maybe 2 K_20? That's K_20,20 which is Strategy 1 without the extra edge.
    # Let's try a random regular-ish graph.
    # Or just use Strategy 1 and 2, and maybe a third one based on Turan graph T_2(40) which is K_20,20.
    
    initializations = [adj1, adj2]
    
    # Add a third: Random but biased towards high connectivity?
    # Let's just run multiple instances of the local search on these starts.
    # To maximize coverage, we split the time.
    
    num_runs = 3
    time_per_run = (deadline - time.time()) / num_runs
    
    for i in range(num_runs):
        if time.time() > deadline:
            break
            
        # Pick an initialization
        # Cycle through them or pick random
        if i == 0:
            start_adj = adj1.copy()
        elif i == 1:
            start_adj = adj2.copy()
        else:
            # Third start: Random
            start_adj = np.zeros((n, n), dtype=bool)
            random.shuffle(all_pairs)
            for u, v in all_pairs[:target_edges]:
                start_adj[u, v] = True
                start_adj[v, u] = True
                
        run_deadline = time.time() + time_per_run
        # Leave a small buffer for the next run setup
        run_deadline = min(run_deadline, deadline - 1.0)
        
        if time.time() >= run_deadline:
            continue
            
        adj_res, score_res = local_search(start_adj, run_deadline)
        
        if score_res < best_overall_score:
            best_overall_score = score_res
            best_overall_adj = adj_res.copy()
            
    # If no valid result found (shouldn't happen), fallback to Strategy 1
    if best_overall_adj is None:
        best_overall_adj = adj1
        best_overall_score = evaluate_adj(adj1)
        
    # Convert to list of edges
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_overall_adj[u, v]:
                result.append([u, v])
                
    # Ensure exactly 401 edges (safety check, though logic preserves count)
    if len(result) != target_edges:
        # This should not happen, but if it does, return a valid fallback
        # Just return the first 401 edges from a known valid graph
        fallback = []
        for u in range(20):
            for v in range(20, 40):
                fallback.append([u, v])
        fallback.append([0, 1])
        return fallback
        
    return result
