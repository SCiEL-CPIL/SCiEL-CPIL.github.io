import itertools
import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Target n=20 first, as it yields the highest score.
    # We use a local search minimizing the number of regular induced subgraphs of size 6.
    # If we find a graph with 0 such subgraphs, we verify for larger sizes.
    
    def get_violation_count(adj_matrix, n):
        """
        Returns the number of regular induced subgraphs of size 6.
        Uses NumPy for speed.
        """
        # Precompute combinations of size 6
        # For n=20, this is 38760 combos.
        # We cache this per n.
        if not hasattr(get_violation_count, "cache"):
            get_violation_count.cache = {}
        
        if n not in get_violation_count.cache:
            combos = list(itertools.combinations(range(n), 6))
            idx = np.array(combos)
            # Create X matrix: (num_combos, n)
            X = np.zeros((len(combos), n), dtype=np.int8)
            for i, combo in enumerate(combos):
                for v in combo:
                    X[i, v] = 1
            get_violation_count.cache[n] = (X, idx)
        
        X, idx = get_violation_count.cache[n]
        
        # Degrees of vertices in each subset
        # D[i, v] = sum_u X[i, u] * A[u, v]
        # A is symmetric, so A[u, v] = A[v, u].
        # D = X @ A
        D = X @ adj_matrix
        
        # Extract degrees for the specific vertices in each combo
        # D_subset[i, j] = D[i, idx[i, j]]
        D_subset = D[np.arange(len(combos))[:, None], idx]
        
        # Check if all degrees in the row are equal
        # Regular if max == min
        maxes = np.max(D_subset, axis=1)
        mins = np.min(D_subset, axis=1)
        
        violations = np.sum(maxes == mins)
        return violations

    def full_check_validity(n, adj_matrix):
        """
        Strict check for any regular induced subgraph of size >= 6.
        """
        # Convert to masks for the fast Python check
        masks = [0] * n
        for i in range(n):
            m = 0
            row = adj_matrix[i]
            for j in range(n):
                if row[j]:
                    m |= (1 << j)
            masks[i] = m
            
        # Check size 6 again (just in case, though fast check said 0)
        # Then check 7, 8, ...
        # For n=20, C(20,7) = 77520. C(20,8) = 125970.
        # This is getting expensive.
        # However, if size 6 is 0, larger ones are rare.
        # We can sample or use a heuristic.
        # But for correctness, we must check all.
        # Optimization: If a subset S of size k is regular, and S' subset S is size 6, 
        # S' is NOT necessarily regular. 
        # So we can't prune based on size 6.
        
        # However, we can use the bitmask method which is faster than Python loops.
        # Let's stick to the Python loop for the final verification only.
        
        for k in range(6, n + 1):
            for subset in itertools.combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check regularity
                degs = []
                for v in subset:
                    common = masks[v] & mask
                    d = common.bit_count()
                    degs.append(d)
                
                if all(d == degs[0] for d in degs):
                    return False
        return True

    def masks_to_edges(masks, n):
        edges = []
        for i in range(n):
            m = masks[i]
            for j in range(i + 1, n):
                if m & (1 << j):
                    edges.append([i, j])
        return edges

    def to_adj_masks(adj_matrix):
        n = adj_matrix.shape[0]
        masks = [0] * n
        for i in range(n):
            m = 0
            for j in range(n):
                if adj_matrix[i, j]:
                    m |= (1 << j)
            masks[i] = m
        return masks

    # Search strategy for n=20
    n_target = 20
    best_adj = None
    best_violations = float('inf')
    
    # Multiple restarts with local search
    for restart in range(20):
        if time.time() > deadline:
            break
            
        # Initialize with a random regular graph of degree ~ n/2
        # Or just random p=0.5
        adj_matrix = (np.random.rand(n_target, n_target) < 0.5).astype(np.int8)
        adj_matrix = np.triu(adj_matrix, 1)
        adj_matrix = adj_matrix + adj_matrix.T
        np.fill_diagonal(adj_matrix, 0)
        
        current_violations = get_violation_count(adj_matrix, n_target)
        
        if current_violations == 0:
            # Verify fully
            if full_check_validity(n_target, adj_matrix):
                masks = to_adj_masks(adj_matrix)
                edges = masks_to_edges(masks, n_target)
                return {'n': n_target, 'edges': edges}
        
        # Local Search: Flip edges to reduce violations
        # We pick a random edge and flip it. If violations decrease, keep it.
        # Otherwise, keep it with some probability (Simulated Annealing) or reject.
        # Given the time limit, simple hill climbing with random restarts is safer.
        
        for step in range(5000):
            if time.time() > deadline:
                break
                
            # Pick a random edge
            u = random.randint(0, n_target - 1)
            v = random.randint(0, n_target - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            # Flip edge
            adj_matrix[u, v] = 1 - adj_matrix[u, v]
            adj_matrix[v, u] = adj_matrix[u, v]
            
            new_violations = get_violation_count(adj_matrix, n_target)
            
            if new_violations < current_violations:
                current_violations = new_violations
                if new_violations == 0:
                    if full_check_validity(n_target, adj_matrix):
                        masks = to_adj_masks(adj_matrix)
                        edges = masks_to_edges(masks, n_target)
                        return {'n': n_target, 'edges': edges}
            else:
                # Revert
                adj_matrix[u, v] = 1 - adj_matrix[u, v]
                adj_matrix[v, u] = adj_matrix[u, v]
                
            # Track best
            if current_violations < best_violations:
                best_violations = current_violations
                best_adj = adj_matrix.copy()
                
    # If we didn't find a perfect graph, try n=19, 18, etc.
    # But n=20 is the target.
    # If best_violations is low but not 0, we can't use it.
    
    # Fallback to smaller n if 20 fails
    for target_n in [19, 18, 17, 16, 15, 14, 13, 12, 11, 10]:
        if time.time() > deadline:
            break
            
        for restart in range(10):
            if time.time() > deadline:
                break
                
            adj_matrix = (np.random.rand(target_n, target_n) < 0.5).astype(np.int8)
            adj_matrix = np.triu(adj_matrix, 1)
            adj_matrix = adj_matrix + adj_matrix.T
            np.fill_diagonal(adj_matrix, 0)
            
            # For smaller n, we can do a more exhaustive check or simpler local search
            # Just check validity directly
            if full_check_validity(target_n, adj_matrix):
                masks = to_adj_masks(adj_matrix)
                edges = masks_to_edges(masks, target_n)
                return {'n': target_n, 'edges': edges}
                
    # Absolute fallback
    return {'n': 5, 'edges': []}
