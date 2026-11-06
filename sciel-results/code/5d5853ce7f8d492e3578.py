import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_edges(adj, n):
        """Extract edge list from bitmask adjacency."""
        edges = []
        for i in range(n):
            mask = adj[i]
            j = 0
            while mask:
                if mask & 1:
                    if i < j:
                        edges.append([i, j])
                mask >>= 1
                j += 1
        return edges

    def check_valid(adj, n, min_deg):
        """Check if graph is C4-free and has min degree >= min_deg."""
        for i in range(n):
            if adj[i].bit_count() < min_deg:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] & adj[j]).bit_count() > 1:
                    return False
        return True

    def random_repair_solve(n, min_deg, max_time=5.0):
        """
        Attempt to solve for n vertices with min degree min_deg using random init and repair.
        Returns adj list or None.
        """
        start = time.time()
        best_adj = None
        best_min_deg = 0
        
        # Try a few different random seeds/initializations within the time slice
        attempts = 0
        while time.time() - start < max_time:
            attempts += 1
            # Use a deterministic seed based on attempt number for reproducibility within the run
            # but vary it across calls if we call this multiple times
            # Here we just use random.random which is seeded externally or by system
            
            # Initialize with random graph
            # Density: aim for slightly above min_deg/n to allow for C4 removals
            p = min_deg / n * 1.2
            if p > 0.9: p = 0.9
            
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Repair loop
            improved = True
            stuck_count = 0
            max_stuck = 1000
            
            while time.time() - start < max_time:
                if stuck_count > max_stuck:
                    break
                
                # 1. Check for C4s
                c4_violations = []
                for i in range(n):
                    for j in range(i + 1, n):
                        common = adj[i] & adj[j]
                        if common.bit_count() > 1:
                            c4_violations.append((i, j, common))
                
                if c4_violations:
                    # Pick a random violation
                    i, j, common = random.choice(c4_violations)
                    # Get list of common neighbors
                    common_list = []
                    mask = common
                    k = 0
                    while mask:
                        if mask & 1:
                            common_list.append(k)
                        mask >>= 1
                        k += 1
                    
                    # Remove one edge to break the C4
                    # We need to reduce common neighbors to 1.
                    # Remove (i, w) or (j, w) for w in common_list[1:]
                    # Prefer removing from the vertex with higher degree to preserve min degree
                    deg_i = adj[i].bit_count()
                    deg_j = adj[j].bit_count()
                    
                    removed = False
                    for w in common_list[1:]:
                        # If removing (i,w) drops deg_i below min_deg, try (j,w)
                        # If both drop below, we are in trouble, but we have to remove one.
                        # Actually, if min_deg is high, we might not be able to remove.
                        # But C4 must be broken.
                        
                        # Heuristic: remove from higher degree vertex
                        if deg_i >= deg_j:
                            # Remove (i, w)
                            adj[i] &= ~(1 << w)
                            adj[w] &= ~(1 << i)
                            deg_i -= 1
                        else:
                            # Remove (j, w)
                            adj[j] &= ~(1 << w)
                            adj[w] &= ~(1 << j)
                            deg_j -= 1
                        removed = True
                        break
                    
                    if removed:
                        stuck_count = 0
                    else:
                        stuck_count += 1
                        
                else:
                    # No C4s
                    # Check min degree
                    min_deg_actual = min(adj[i].bit_count() for i in range(n))
                    
                    if min_deg_actual >= min_deg:
                        # Success
                        if check_valid(adj, n, min_deg):
                            return adj
                        else:
                            # Should not happen if logic is correct, but break just in case
                            break
                    else:
                        # Need to add edges
                        # Find vertex with min degree
                        u = min(range(n), key=lambda i: adj[i].bit_count())
                        
                        # Find non-neighbor v such that N(u) and N(v) are disjoint
                        candidates = []
                        for v in range(n):
                            if v == u:
                                continue
                            if not (adj[u] & (1 << v)):
                                if (adj[u] & adj[v]) == 0:
                                    candidates.append(v)
                        
                        if candidates:
                            # Pick a candidate. Prefer one with low degree to balance?
                            # Or random.
                            v = random.choice(candidates)
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            stuck_count = 0
                        else:
                            # Cannot add edge to u without creating C4
                            # Try to swap? Or just stuck.
                            # For now, mark as stuck
                            stuck_count += 1
                            
        return None

    # Try n from 47 down to 43
    for n in range(47, 42, -1):
        min_deg = n - 39
        if min_deg < 1:
            continue
        
        found = False
        
        # Run multiple independent attempts for this n
        # Each attempt gets a time slice
        # Total time for this n should be managed carefully
        # Let's give n=47 more time if we have it
        time_spent = time.time() - start_time
        time_remaining = deadline - time_spent
        if time_remaining < 10:
            break
            
        # Allocate time based on n
        # n=47: 40s, n=46: 30s, etc?
        # Or just loop attempts until time runs out.
        
        max_attempts_for_n = 50
        for attempt in range(max_attempts_for_n):
            if time.time() > deadline - 5.0:
                break
                
            # Use a fresh random state or seed
            # random.seed(attempt * 1000 + n) # Deterministic per attempt
            
            adj = random_repair_solve(n, min_deg, max_time=2.0)
            
            if adj is not None:
                if check_valid(adj, n, min_deg):
                    edges = get_edges(adj, n)
                    return {'n': n, 'edges': edges}
        
        # If we didn't find it for this n, try the next lower n

    # Fallback: Return a valid small graph
    # A cycle C_n is C4-free for n != 4. Min degree 2.
    # For n=40, min_deg = 1.
    n = 40
    min_deg = 1
    adj = [0] * n
    for i in range(n):
        adj[i] |= (1 << ((i + 1) % n))
        adj[(i + 1) % n] |= (1 << i)
    
    if check_valid(adj, n, min_deg):
        edges = get_edges(adj, n)
        return {'n': n, 'edges': edges}
    
    return {'n': 39, 'edges': []}
