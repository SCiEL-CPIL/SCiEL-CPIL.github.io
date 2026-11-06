import random
import time

def construct():
    target_n = 25
    time_limit = 110
    start_time = time.time()
    
    # Bitset representation
    # adj[i] is an integer where bit j is 1 if (i,j) is an edge
    
    def is_valid(n, adj):
        # Check K4
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    # Common neighbors of i and j
                    common = adj[i] & adj[j]
                    # Remove i and j from common? 
                    # Actually, if there is a k in common, and k has a neighbor l in common, we have K4.
                    # K4: i, j, k, l. 
                    # i-j, i-k, i-l, j-k, j-l, k-l.
                    # If k is in common, then i-k and j-k exist.
                    # We need k-l where l is also in common and l != i, j.
                    # So we need common to have at least 2 vertices that are connected?
                    # No, we need a clique of size 2 in the subgraph induced by common?
                    # If k, l in common, and k-l is an edge, then {i, j, k, l} is a K4.
                    # So check if the subgraph induced by common has an edge.
                    
                    # To check if subgraph induced by common has an edge:
                    # Iterate through bits in common?
                    # Or: for each k in common, if (common & adj[k]) is not empty, return True.
                    # We must ensure k != i, j. But i, j are not in adj[i] or adj[j] typically (no self loops).
                    # So common does not contain i or j.
                    
                    # Optimization: 
                    # If popcount(common) < 2, skip.
                    if common & (common - 1): # has at least 2 bits
                        # Check for edge in common
                        # We can iterate
                        c = common
                        while c:
                            k = (c & -c).bit_length() - 1
                            c &= c - 1
                            if (common & adj[k]):
                                return False # Found K4
        # Check Independent Set of 6
        # Equivalent to Clique of 6 in complement
        # Complement adj: comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
        # We need to check if comp_adj has a clique of size 6.
        # This is harder. 
        # For n=25, we can use a simple backtracking or just check if any independent set of size 6 exists.
        # Greedy check? No, must be exact.
        
        # Simple backtracking for independent set of size 6
        # Use the original adj.
        # Independent set: no edges between them.
        
        def has_ind_set(adj, n, size):
            # Backtracking
            # candidates: list of vertices
            # To speed up, use bitsets?
            # Let's use a simple recursive function with pruning.
            
            # If n is small (25), this is fine.
            # But we call this in a loop.
            # We need it to be fast.
            
            # Alternative: 
            # If the graph is dense, independent sets are small.
            # If the graph is sparse, independent sets are large.
            
            # Let's use a bitset-based independent set check.
            # candidates_mask
            # Pick a vertex v.
            # Branch 1: Include v. New candidates = candidates & ~adj[v] & ~(1<<v)
            # Branch 2: Exclude v. New candidates = candidates & ~(1<<v)
            
            def backtrack(candidates, k):
                if k == 0:
                    return True
                if not candidates:
                    return False
                if candidates.bit_count() < k:
                    return False
                
                # Heuristic: pick vertex with max degree in induced subgraph?
                # Or just pick the first one.
                v = (candidates & -candidates).bit_length() - 1
                
                # Include v
                new_cand = candidates & ~adj[v] & ~(1 << v)
                if backtrack(new_cand, k - 1):
                    return True
                
                # Exclude v
                if backtrack(candidates & ~(1 << v), k):
                    return True
                
                return False

            return backtrack((1 << n) - 1, size)

        if has_ind_set(adj, n, 6):
            return False
            
        return True

    def get_violation(n, adj):
        # Returns (u, v) of an edge to flip, or None
        # Check K4
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    if common & (common - 1):
                        c = common
                        while c:
                            k = (c & -c).bit_length() - 1
                            c &= c - 1
                            if (common & adj[k]):
                                # Found K4 {i, j, k, l} where l is in common & adj[k]
                                return (i, j) # Flip i-j to break K4
        # Check Ind 6
        # Find an independent set of size 6
        # We can reuse the backtracking logic to return the set
        def find_ind_set(adj, n, size):
            def backtrack(candidates, k, current_set):
                if k == 0:
                    return current_set
                if not candidates:
                    return None
                if candidates.bit_count() < k:
                    return None
                
                v = (candidates & -candidates).bit_length() - 1
                
                # Include v
                new_cand = candidates & ~adj[v] & ~(1 << v)
                res = backtrack(new_cand, k - 1, current_set + [v])
                if res:
                    return res
                
                # Exclude v
                res = backtrack(candidates & ~(1 << v), k, current_set)
                if res:
                    return res
                
                return None

            return backtrack((1 << n) - 1, size, [])
        
        ind_set = find_ind_set(adj, n, 6)
        if ind_set:
            # Add an edge within the independent set
            return (ind_set[0], ind_set[1])
            
        return None

    # Local Search
    best_n = 17
    best_edges = None
    # Seed edges for n=17 (Paley)
    squares = {x * x % 17 for x in range(1, 17)}
    seed_adj = [0] * 17
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                seed_adj[i] |= (1 << j)
                seed_adj[j] |= (1 << i)
    best_edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (seed_adj[i] >> j) & 1]
    
    random.seed(42)
    
    # Try to find a valid graph on target_n
    for restart in range(50):
        if time.time() - start_time > time_limit:
            break
            
        n = target_n
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        steps = 0
        while steps < 5000 and time.time() - start_time < time_limit:
            if is_valid(n, adj):
                # Convert adj to edges
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
            move = get_violation(n, adj)
            if move:
                u, v = move
                # Flip edge
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                # No violation found? Shouldn't happen if is_valid is correct
                # Random flip
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    if u > v: u, v = v, u
                    if (adj[u] >> v) & 1:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        
            steps += 1
            
    return {'n': best_n, 'edges': best_edges}
