import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin

    # 1. Initialize with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n = q
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_n = n
    best_adj = adj[:]

    # Helper: Check if subset S (list of indices) contains a K4
    def has_k4_in_subset(subset, adj):
        if len(subset) < 4:
            return False
        # Optimize: if subset is small, check all 4-combos
        for combo in itertools.combinations(subset, 4):
            a, b, c, d = combo
            # Check if all 6 edges exist
            if (adj[a] >> b) & 1:
                if (adj[a] >> c) & 1:
                    if (adj[a] >> d) & 1:
                        if (adj[b] >> c) & 1:
                            if (adj[b] >> d) & 1:
                                if (adj[c] >> d) & 1:
                                    return True
        return False

    # Helper: Check if subset T (list of indices) contains an independent set of size 4
    def has_ind4_in_subset(subset, adj):
        if len(subset) < 4:
            return False
        for combo in itertools.combinations(subset, 4):
            a, b, c, d = combo
            # Check if NO edges exist
            if not ((adj[a] >> b) & 1):
                if not ((adj[a] >> c) & 1):
                    if not ((adj[a] >> d) & 1):
                        if not ((adj[b] >> c) & 1):
                            if not ((adj[b] >> d) & 1):
                                if not ((adj[c] >> d) & 1):
                                    return True
        return False

    # Helper: Check if adding vertex with neighbor mask 'new_mask' is valid
    def is_valid_addition(new_mask, adj, n):
        # S: neighbors of new vertex
        # V\S: non-neighbors of new vertex
        S = [i for i in range(n) if (new_mask >> i) & 1]
        T = [i for i in range(n) if not ((new_mask >> i) & 1)]
        
        if has_k4_in_subset(S, adj):
            return False
        if has_ind4_in_subset(T, adj):
            return False
        return True

    # Helper: Add vertex
    def add_vertex(adj, new_mask):
        n = len(adj)
        new_adj = adj[:]
        new_adj.append(0)
        for i in range(n):
            if (new_mask >> i) & 1:
                new_adj[i] |= (1 << n)
                new_adj[n] |= (1 << i)
        return new_adj

    current_adj = best_adj
    current_n = best_n

    max_iterations = 1000
    iterations = 0
    
    while current_n < 46 and iterations < max_iterations:
        if time.time() > deadline:
            break
        
        # Try to find a valid connection pattern for a new vertex
        found = False
        attempts = 0
        max_attempts = 5000
        
        while attempts < max_attempts:
            if time.time() > deadline:
                break
            
            # Generate random mask
            # Bias towards 50% density
            mask = 0
            for i in range(current_n):
                if random.random() < 0.5:
                    mask |= (1 << i)
            
            if is_valid_addition(mask, current_adj, current_n):
                current_adj = add_vertex(current_adj, mask)
                current_n += 1
                found = True
                if current_n > best_n:
                    best_n = current_n
                    best_adj = current_adj[:]
                break
            
            attempts += 1
        
        if not found:
            # If we can't add a vertex, maybe we should try to restart from best_adj with different randomness?
            # Or just stop.
            # Let's try a few more random restarts of the search for the mask
            # Actually, if we are stuck, the graph might be "saturated".
            # We can try to remove a vertex and re-add? No, that reduces n.
            # We can try to flip an edge in the current graph to "unstick" it?
            # This is complex. Let's just break if we fail a few times.
            # To be safe, let's try to find ANY valid graph of size best_n + 1 by random search from scratch?
            # No, too slow.
            break

    # Reconstruct edges
    best_edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                best_edges.append((i, j))
    
    return best_n, best_edges
