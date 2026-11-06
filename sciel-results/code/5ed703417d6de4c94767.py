import time
import random

def _has_k33(adj, n):
    """
    Check if graph with adjacency bitmasks adj on n vertices contains K3,3.
    Returns True if found, False otherwise.
    """
    # For each triple (a, b, c), check if they have >= 3 common neighbors
    # Common neighbors of a, b, c are in adj[a] & adj[b] & adj[c]
    # Since it's a simple graph, a,b,c are not in their own neighbor sets,
    # so we don't need to explicitly mask them out unless self-loops were possible (they are not).
    # Wait, adj[a] contains neighbors of a. a is not in adj[a].
    # So common = adj[a] & adj[b] & adj[c] correctly represents N(a) ∩ N(b) ∩ N(c).
    # We need to ensure the common neighbors are distinct from a, b, c.
    # Since a is not in adj[a], a is not in common. Same for b and c.
    # So we just check if popcount(common) >= 3.
    
    for a in range(n):
        for b in range(a + 1, n):
            common_ab = adj[a] & adj[b]
            if common_ab == 0:
                continue
            for c in range(b + 1, n):
                common_abc = common_ab & adj[c]
                # Count bits
                if common_abc & (common_abc - 1) & (common_abc - (common_abc & (common_abc - 1))) != 0:
                    # This checks if there are at least 3 bits set?
                    # Simpler: bin(common_abc).count('1') >= 3
                    # Even faster for small n:
                    x = common_abc
                    if x:
                        x &= x - 1
                        if x:
                            x &= x - 1
                            if x:
                                return True
    return False

def _greedy_fill(adj, n, deadline):
    """
    Greedily add edges to adj until no more can be added or deadline reached.
    Returns the updated adj list.
    """
    # Generate list of all missing edges
    missing = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                missing.append((u, v))
    
    random.shuffle(missing)
    
    for u, v in missing:
        if time.time() > deadline:
            break
        # Try adding edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if _has_k33(adj, n):
            # Revert
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        # If no K3,3, keep it
    
    return adj

def _local_search(adj, n, deadline):
    """
    Local search: Try to improve by removing edges and re-filling.
    """
    best_adj = adj[:]
    best_count = sum(bin(x).count('1') for x in adj) // 2
    
    while time.time() < deadline:
        # 1. Greedy fill
        adj = _greedy_fill(adj, n, deadline)
        
        # 2. Count edges
        count = sum(bin(x).count('1') for x in adj) // 2
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        # 3. Check if we can add more
        # If we can't add any more edges, we are in a local optimum.
        # Try to escape by removing a random edge and re-filling.
        
        # Find a missing edge that *cannot* be added? 
        # Actually, _greedy_fill already tried all missing edges.
        # So if we are stuck, we must remove an existing edge.
        
        # Pick a random existing edge to remove
        existing_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    existing_edges.append((u, v))
        
        if not existing_edges:
            break
            
        # Remove 1 to 3 random edges to create space
        num_remove = random.randint(1, 3)
        for _ in range(num_remove):
            if not existing_edges:
                break
            idx = random.randint(0, len(existing_edges) - 1)
            u, v = existing_edges[idx]
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            existing_edges.pop(idx)
            
        # Loop back to greedy fill

    return best_adj

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    best_edges = []
    best_count = 0
    
    # Seed: K2,18
    adj_seed = [0] * n
    for a in range(2):
        for b in range(2, n):
            adj_seed[a] |= (1 << b)
            adj_seed[b] |= (1 << a)
    count_seed = sum(bin(x).count('1') for x in adj_seed) // 2
    if count_seed > best_count:
        best_count = count_seed
        best_adj = adj_seed[:]
    
    # Multiple restarts
    # Strategy: Start with random graphs of varying densities, clean them, then local search.
    
    for restart in range(100): # Adjust based on speed
        if time.time() > deadline:
            break
        
        # Random initial graph
        # Density p between 0.1 and 0.4 seems good for K3,3-free
        p = random.uniform(0.1, 0.4)
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Clean: Remove edges until K3,3-free
        # We do this by finding a K3,3 and removing a random edge from it.
        # Repeat until no K3,3.
        max_clean_iters = 1000
        for _ in range(max_clean_iters):
            if time.time() > deadline:
                break
            if not _has_k33(adj, n):
                break
            
            # Find a violating triple
            found = False
            for a in range(n):
                for b in range(a + 1, n):
                    common_ab = adj[a] & adj[b]
                    if not common_ab:
                        continue
                    for c in range(b + 1, n):
                        common_abc = common_ab & adj[c]
                        # Check if >= 3 bits
                        x = common_abc
                        if x:
                            x &= x - 1
                            if x:
                                x &= x - 1
                                if x:
                                    # Found K3,3 witness: {a,b,c} and 3 common neighbors
                                    # Get list of common neighbors
                                    common_list = []
                                    mask = common_abc
                                    while mask:
                                        lsb = mask & -mask
                                        idx = lsb.bit_length() - 1
                                        common_list.append(idx)
                                        mask ^= lsb
                                    
                                    # Remove a random edge from the K3,3
                                    # The K3,3 edges are between {a,b,c} and common_list
                                    # Pick a random vertex from {a,b,c} and a random vertex from common_list
                                    u = random.choice([a, b, c])
                                    v = random.choice(common_list)
                                    adj[u] &= ~(1 << v)
                                    adj[v] &= ~(1 << u)
                                    found = True
                                    break
                if found:
                    break
            if not found:
                break
        
        # Local Search
        adj = _local_search(adj, n, deadline)
        
        # Update best
        count = sum(bin(x).count('1') for x in adj) // 2
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            
    # Final Greedy Fill on Best
    if time.time() < deadline:
        best_adj = _greedy_fill(best_adj, n, deadline)
        count = sum(bin(x).count('1') for x in best_adj) // 2
        if count > best_count:
            best_count = count
            # best_adj is already updated by reference in _greedy_fill? 
            # _greedy_fill returns adj, but I modified it in place in the function?
            # Let's check _greedy_fill. It modifies adj in place and returns it.
            # So best_adj is updated.

    # Convert best_adj to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
