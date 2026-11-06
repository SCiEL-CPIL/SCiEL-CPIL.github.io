import time
import random
import sys

def _max_independent_set_size(adj, n):
    """
    Exact branch and bound to find the size of the maximum independent set.
    adj: list of bitmasks or sets of neighbors.
    Returns the size of the max independent set.
    """
    # Use bitmasks for speed
    # adj_mask[i] is a bitmask of neighbors of i
    
    # Branch and bound
    # We can use a simple recursive DFS with pruning.
    # To make it fast, we order vertices by degree.
    
    # For n up to 40, a simple B&B might be slow if not optimized.
    # Let's use a greedy lower bound and a simple upper bound.
    
    # Actually, for n=39, a well-written B&B in Python might be too slow if called many times.
    # We need a faster check.
    
    # Alternative: Use a heuristic to find an independent set of size 10.
    # If we find one, alpha >= 10.
    # If we cannot find one, it doesn't mean alpha < 10.
    # So we need exactness or a very strong heuristic.
    
    # Given the time limit, we can use a randomized search to find an independent set of size 10.
    # If we fail to find one after many tries, we assume it's valid (risky).
    # Better: Use a branch and bound that is optimized for "does alpha >= 10 exist?".
    
    # Let's implement a fast B&B for alpha >= k.
    
    def has_independent_set_of_size_k(adj_mask, n, k):
        if k == 0:
            return True
        if n == 0:
            return False
        
        # Pick a vertex
        # Heuristic: Pick vertex with minimum degree in the remaining graph?
        # Or just pick 0 for simplicity, but ordering helps.
        
        # For speed, we'll just pick vertex 0 of the current set.
        # This is slow. Let's use a simple recursion.
        
        # To speed up, we can use a greedy coloring bound.
        # But for k=10, a simple DFS might suffice if the graph is dense enough.
        
        # Let's use a standard B&B.
        
        # State: (remaining_vertices_mask, k)
        # Pruning:
        # 1. If k > popcount(remaining_vertices_mask): return False
        # 2. If k == 0: return True
        
        # We need to pass the mask.
        
        # Let's implement this carefully.
        
        pass

    # Since implementing a fast B&B in Python is tricky for 120s limit with multiple calls,
    # let's rely on the fact that if a graph is "dense" (maximal triangle-free), 
    # the independent sets are smaller.
    
    # We will use a randomized greedy search to find an independent set of size 10.
    # If we find one, it's invalid.
    # If we don't find one after, say, 1000 tries, we accept it.
    # This is a heuristic, but likely to work for n < 40.
    
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Seed result
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    best_result = {'n': n_seed, 'edges': edges_seed}
    
    # Try to find better graphs
    for n in range(27, 40):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        for attempt in range(50): # Adjust based on time
            if time.time() > deadline:
                break
                
            # Generate a random maximal triangle-free graph
            # 1. Start with empty graph
            # 2. Add edges randomly, ensuring no triangle
            # This is slow.
            
            # Better: Start with a random bipartite graph
            # Partition vertices into A and B randomly
            # Add random edges between A and B
            # Then complete it to maximal triangle-free
            
            # 1. Random bipartite
            adj = [set() for _ in range(n)]
            # Random partition
            parts = [0] * n
            for i in range(n):
                parts[i] = random.randint(0, 1)
            
            # Add random edges between parts
            # Density 0.5
            for u in range(n):
                for v in range(u + 1, n):
                    if parts[u] != parts[v] and random.random() < 0.5:
                        adj[u].add(v)
                        adj[v].add(u)
            
            # 2. Complete to maximal triangle-free
            # Iterate through all pairs
            # To speed up, we can just add any edge that doesn't create a triangle
            # We do this in a random order to avoid bias
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if u not in adj[v]:
                    # Check if adding (u,v) creates a triangle
                    # Triangle if there exists w such that w in adj[u] and w in adj[v]
                    # i.e., adj[u] & adj[v] is not empty
                    if not (adj[u] & adj[v]):
                        adj[u].add(v)
                        adj[v].add(u)
            
            # 3. Check if alpha < 10
            # Use a heuristic: Try to find an independent set of size 10
            # If we find one, it's invalid.
            # If we don't, we accept it.
            
            valid = True
            # Greedy search for independent set of size 10
            # We can use a simple backtracking with a limit
            # Or just a greedy random sampling
            
            # Let's do a simple B&B for alpha >= 10
            # If it returns False, we accept.
            
            # For speed, we'll use a randomized greedy approach
            # Try to build an independent set of size 10
            # If we succeed, valid = False
            
            # Simple B&B for alpha >= 10
            # We represent the graph with bitmasks
            n_mask = (1 << n) - 1
            adj_mask = [0] * n
            for u in range(n):
                mask = 0
                for v in adj[u]:
                    mask |= (1 << v)
                adj_mask[u] = mask
            
            # B&B function
            # Returns True if an independent set of size >= 10 exists
            # We use a depth-limited search or a full search with pruning
            
            # Given the time, we'll use a simple recursive search
            # with a node limit
            
            node_limit = 1000
            nodes_visited = 0
            
            def search_independent_set(remaining_mask, k, adj_mask, n, node_limit, nodes_visited):
                nonlocal nodes_visited
                if k == 0:
                    return True
                if remaining_mask == 0:
                    return False
                if bin(remaining_mask).count('1') < k:
                    return False
                if nodes_visited > node_limit:
                    return None # Unknown
                    
                nodes_visited += 1
                
                # Pick a vertex
                # Heuristic: Pick vertex with min degree in remaining graph
                # For simplicity, pick the lowest set bit
                # This is slow but okay for small k
                # Let's pick a random vertex from remaining
                # Or the first one
                v = (remaining_mask & -remaining_mask).bit_length() - 1
                
                # Branch 1: Include v
                # New remaining: remaining_mask & ~adj_mask[v] & ~(1 << v)
                new_remaining = remaining_mask & ~adj_mask[v] & ~(1 << v)
                res = search_independent_set(new_remaining, k - 1, adj_mask, n, node_limit, nodes_visited)
                if res is True:
                    return True
                if res is None:
                    return None
                
                # Branch 2: Exclude v
                new_remaining = remaining_mask & ~(1 << v)
                res = search_independent_set(new_remaining, k, adj_mask, n, node_limit, nodes_visited)
                if res is True:
                    return True
                if res is None:
                    return None
                
                return False
            
            res = search_independent_set(n_mask, 10, adj_mask, n, node_limit, nodes_visited)
            
            if res is False:
                # No independent set of size 10 found within limit
                # Assume valid
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
            elif res is None:
                # Unknown, skip this graph
                continue
            else:
                # Found independent set of size 10
                # Invalid
                continue
                
    return best_result

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Seed result
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    best_result = {'n': n_seed, 'edges': edges_seed}
    
    # Try to find better graphs
    for n in range(27, 40):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        for attempt in range(20): # Adjust based on time
            if time.time() > deadline:
                break
                
            # Generate a random maximal triangle-free graph
            adj = [set() for _ in range(n)]
            parts = [0] * n
            for i in range(n):
                parts[i] = random.randint(0, 1)
            
            for u in range(n):
                for v in range(u + 1, n):
                    if parts[u] != parts[v] and random.random() < 0.5:
                        adj[u].add(v)
                        adj[v].add(u)
            
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if u not in adj[v]:
                    if not (adj[u] & adj[v]):
                        adj[u].add(v)
                        adj[v].add(u)
            
            # Check if alpha < 10
            n_mask = (1 << n) - 1
            adj_mask = [0] * n
            for u in range(n):
                mask = 0
                for v in adj[u]:
                    mask |= (1 << v)
                adj_mask[u] = mask
            
            node_limit = 500
            nodes_visited = 0
            
            def search_independent_set(remaining_mask, k):
                nonlocal nodes_visited
                if k == 0:
                    return True
                if remaining_mask == 0:
                    return False
                if bin(remaining_mask).count('1') < k:
                    return False
                if nodes_visited > node_limit:
                    return None
                    
                nodes_visited += 1
                
                v = (remaining_mask & -remaining_mask).bit_length() - 1
                
                new_remaining = remaining_mask & ~adj_mask[v] & ~(1 << v)
                res = search_independent_set(new_remaining, k - 1)
                if res is True:
                    return True
                if res is None:
                    return None
                
                new_remaining = remaining_mask & ~(1 << v)
                res = search_independent_set(new_remaining, k)
                if res is True:
                    return True
                if res is None:
                    return None
                
                return False
            
            res = search_independent_set(n_mask, 10)
            
            if res is False:
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append([u, v])
                return {'n': n, 'edges': edges}
                
    return best_result
# EVOLVE-BLOCK-END
