import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    # 1. Precompute smoothness for sums up to 20000
    max_sum = 20000
    is_smooth = np.zeros(max_sum + 1, dtype=bool)
    
    # Generate all 7-smooth numbers
    for a in range(15):
        p2 = 2 ** a
        if p2 > max_sum: break
        for b in range(10):
            p3 = p2 * (3 ** b)
            if p3 > max_sum: break
            for c in range(7):
                p5 = p3 * (5 ** c)
                if p5 > max_sum: break
                for d in range(5):
                    p7 = p5 * (7 ** d)
                    if p7 > max_sum: break
                    is_smooth[p7] = True

    def verify_clique(clique):
        if len(clique) != 12:
            return False
        if len(set(clique)) != 12:
            return False
        for x in clique:
            if x < 1 or x > 10000:
                return False
        for i in range(12):
            for j in range(i+1, 12):
                if not is_smooth[clique[i] + clique[j]]:
                    return False
        return True

    def find_clique(candidates, adj_matrix, target_size, time_limit):
        n = len(candidates)
        if n < target_size:
            return None
            
        # Calculate degrees
        degrees = adj_matrix.sum(axis=1)
        # Sort indices by degree descending
        order = np.argsort(-degrees)
        
        # Greedy Attempt
        current_clique = []
        for idx in order:
            if time.time() > time_limit:
                break
            # Check compatibility with all in current_clique
            # adj_matrix[idx, current_clique]
            if len(current_clique) == 0:
                current_clique.append(idx)
            else:
                # Check if all are True
                if np.all(adj_matrix[idx, current_clique]):
                    current_clique.append(idx)
                    if len(current_clique) == target_size:
                        return [candidates[i] for i in current_clique]
        
        # Backtracking Attempt (limited)
        # Use the top 50 candidates by degree
        top_k = min(50, n)
        top_indices = [order[i] for i in range(top_k)]
        
        def backtrack(path, available):
            if time.time() > time_limit:
                return None
            if len(path) == target_size:
                return path[:]
            
            for i, idx in enumerate(available):
                if not np.all(adj_matrix[idx, path]):
                    continue
                path.append(idx)
                res = backtrack(path, available[i+1:])
                if res:
                    return res
                path.pop()
            return None
        
        res = backtrack([], top_indices)
        if res:
            return [candidates[i] for i in res]
            
        return None

    # Pools to try
    pools = []
    # 1. Small numbers
    pools.append(np.arange(1, 2001))
    # 2. Random numbers
    np.random.seed(42)
    pools.append(np.random.choice(np.arange(1, 10001), 2000, replace=False))
    # 3. Multiples of 2
    pools.append(np.arange(2, 2001, 2))
    
    for pool in pools:
        if time.time() > deadline:
            break
        
        n = len(pool)
        # Build adjacency matrix
        # Use broadcasting
        sums = np.add.outer(pool, pool)
        adj_matrix = is_smooth[sums]
        
        # We don't need diagonal
        # np.fill_diagonal(adj_matrix, False) # Not strictly necessary for clique search if we handle distinctness
        
        result = find_clique(pool, adj_matrix, 12, deadline)
        
        if result:
            if verify_clique(result):
                return result
    
    # Fallback
    return list(range(1, 13))
