import time
import random
import numpy as np
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for evaluation overhead

    def check_validity(n, adj_np):
        """
        Checks if the graph has no regular induced subgraph of size >= 6.
        adj_np: numpy array of shape (n, n) with 0/1.
        """
        # Precompute masks for sizes 6 to n? 
        # It's better to iterate sizes.
        
        # Helper to check if a specific set of vertices (indices array) induces a regular subgraph
        def is_reg(indices):
            k = len(indices)
            if k < 2:
                return True # Trivially regular? Or undefined? Task says >= 6.
            sub_adj = adj_np[np.ix_(indices, indices)]
            degrees = np.sum(sub_adj, axis=1)
            return np.all(degrees == degrees[0])

        # Check sizes from 6 to n
        for k in range(6, n + 1):
            for comb in itertools.combinations(range(n), k):
                if is_reg(comb):
                    return False
        return True

    def count_bad_6(n, adj_np):
        """
        Counts the number of 6-vertex subsets that induce a regular subgraph.
        Uses NumPy for speed.
        """
        count = 0
        # Precompute all 6-combinations
        # To avoid regenerating, we could cache, but n is fixed per search.
        # C(20, 6) = 38760.
        for comb in itertools.combinations(range(n), 6):
            indices = np.array(comb)
            sub_adj = adj_np[np.ix_(indices, indices)]
            degrees = np.sum(sub_adj, axis=1)
            if np.all(degrees == degrees[0]):
                count += 1
        return count

    def local_search(n, max_iters=500):
        """
        Attempts to find a valid graph on n vertices using local search.
        Returns adj_np if valid, else None.
        """
        # Initialize random graph
        adj_list = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj_list[i, j] = 1
                    adj_list[j, i] = 1
        
        # Check initial
        # Optimization: Only do full check if count_bad_6 is 0
        if count_bad_6(n, adj_list) == 0:
            if check_validity(n, adj_list):
                return adj_list

        best_adj = adj_list.copy()
        best_score = count_bad_6(n, adj_list)
        
        current_score = best_score
        
        # Precompute edge list for random flipping
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random edge to flip
            u, v = random.choice(edges)
            
            # Flip edge
            adj_list[u, v] ^= 1
            adj_list[v, u] ^= 1
            
            new_score = count_bad_6(n, adj_list)
            
            # Accept if better or equal
            if new_score <= current_score:
                current_score = new_score
                if current_score == 0:
                    # Check full validity
                    if check_validity(n, adj_list):
                        return adj_list
            else:
                # Revert
                adj_list[u, v] ^= 1
                adj_list[v, u] ^= 1
                
            # Track best
            if current_score < best_score:
                best_score = current_score
                best_adj = adj_list.copy()
        
        return None

    # Try n=21
    t_start = time.time()
    target_n = 21
    max_restarts = 3
    for i in range(max_restarts):
        if time.time() > t_start + 15: # 15s budget for n=21
            break
        adj = local_search(target_n, max_iters=200)
        if adj is not None:
            return {'n': target_n, 'edges': [[u, v] for u in range(target_n) for v in range(u + 1, target_n) if adj[u, v] == 1]}

    # Try n=20
    t_start = time.time()
    target_n = 20
    max_restarts = 5
    for i in range(max_restarts):
        if time.time() > t_start + 20: # 20s budget for n=20
            break
        adj = local_search(target_n, max_iters=500)
        if adj is not None:
            return {'n': target_n, 'edges': [[u, v] for u in range(target_n) for v in range(u + 1, target_n) if adj[u, v] == 1]}

    # Try n=15 as a safer fallback
    t_start = time.time()
    target_n = 15
    max_restarts = 10
    for i in range(max_restarts):
        if time.time() > t_start + 10:
            break
        adj = local_search(target_n, max_iters=200)
        if adj is not None:
            return {'n': target_n, 'edges': [[u, v] for u in range(target_n) for v in range(u + 1, target_n) if adj[u, v] == 1]}

    # Fallback: Return a known valid small graph.
    # K6 minus one edge is valid (n=6).
    # Let's make a non-regular graph on 10 vertices to be safe.
    # Just return a random non-regular graph on 10 vertices?
    # Ensure it's valid by checking.
    n = 10
    adj_list = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < 0.5:
                adj_list[i, j] = 1
                adj_list[j, i] = 1
    # Make it non-regular by construction?
    # Degrees: make vertex 0 degree 0, others random.
    adj_list[0, :] = 0
    adj_list[:, 0] = 0
    # Now check validity for n=10
    # If invalid, try again a few times
    for _ in range(10):
        if count_bad_6(n, adj_list) == 0:
            if check_validity(n, adj_list):
                return {'n': n, 'edges': [[u, v] for u in range(n) for v in range(u + 1, n) if adj_list[u, v] == 1]}
        # Perturb
        u, v = random.randint(0, n-1), random.randint(0, n-1)
        if u != v and u != 0 and v != 0:
            adj_list[u, v] ^= 1
            adj_list[v, u] ^= 1

    # Ultimate fallback: n=6 K6 minus edge
    edges = []
    for u in range(6):
        for v in range(u + 1, 6):
            if not (u == 0 and v == 1):
                edges.append([u, v])
    return {'n': 6, 'edges': edges}
