import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Target n
    target_n = 20
    
    def is_regular_induced(subset, adj):
        """
        Check if the induced subgraph on 'subset' is regular.
        subset: list of vertex indices
        adj: list of bitmasks, adj[u] has bit v set if (u,v) is an edge
        """
        k = len(subset)
        if k < 6:
            return False
        
        # Map subset vertices to 0..k-1 for local degree calculation
        # Actually, we just need to check if all degrees in the induced subgraph are equal.
        # Degree of u in induced subgraph = number of neighbors of u that are in subset.
        
        degrees = []
        for u in subset:
            deg = 0
            mask = adj[u]
            # Count bits in mask that correspond to vertices in subset
            # We can precompute a mask for the subset
            # For efficiency in this small k, we can just iterate
            for v in subset:
                if v == u:
                    continue
                if mask & (1 << v):
                    deg += 1
            degrees.append(deg)
        
        # Check if all degrees are equal
        if all(d == degrees[0] for d in degrees):
            return True
        return False

    def find_violation(adj, n):
        """
        Find any subset of size 6 that induces a regular subgraph.
        Returns the subset if found, else None.
        """
        vertices = list(range(n))
        # We can sample random subsets if n is large, but for n=20, C(20,6) = 38760, which is small.
        # We can iterate through all subsets of size 6.
        for subset in itertools.combinations(vertices, 6):
            if is_regular_induced(subset, adj):
                return subset
        return None

    def count_violations(adj, n):
        """
        Count the number of subsets of size 6 that induce a regular subgraph.
        """
        vertices = list(range(n))
        count = 0
        for subset in itertools.combinations(vertices, 6):
            if is_regular_induced(subset, adj):
                count += 1
        return count

    # Initialize a random graph
    best_adj = None
    best_violations = float('inf')
    
    # Try a few random initializations
    for _ in range(5):
        adj = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        violations = count_violations(adj, target_n)
        if violations < best_violations:
            best_violations = violations
            best_adj = adj[:]
            
        if violations == 0:
            # Found a valid graph immediately
            edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if best_adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': target_n, 'edges': edges}

    # Local search
    current_adj = best_adj[:]
    current_violations = best_violations
    
    # Precompute all subsets of size 6 for faster access? 
    # C(20,6) is ~38k. Storing them might be memory heavy if done repeatedly, but we can generate them on the fly or cache.
    # Let's just iterate.
    
    while time.time() < deadline:
        # Find a violation
        vertices = list(range(target_n))
        violation = None
        # Randomize the order of checking to avoid local minima
        random.shuffle(vertices)
        for subset in itertools.combinations(vertices, 6):
            if is_regular_induced(subset, current_adj):
                violation = subset
                break
        
        if violation is None:
            # Valid graph found
            edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if current_adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': target_n, 'edges': edges}
        
        # Pick a random edge in the violation to flip
        u, v = random.sample(violation, 2)
        
        # Flip the edge
        if current_adj[u] & (1 << v):
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
        else:
            current_adj[u] |= (1 << v)
            current_adj[v] |= (1 << u)
            
        # Check if this move improved the situation (optional, but helps convergence)
        # For simplicity, we just accept the move if it doesn't drastically increase violations, 
        # or just accept all moves (random walk). 
        # Let's do a simple hill climbing: accept if violations decrease or stay same.
        new_violations = count_violations(current_adj, target_n)
        if new_violations > current_violations and random.random() > 0.1:
            # Revert if it got worse (with some probability to allow escaping local minima)
            if current_adj[u] & (1 << v):
                current_adj[u] ^= (1 << v)
                current_adj[v] ^= (1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
            # Revert means we keep the old state
        else:
            current_violations = new_violations
            
        # If we get stuck, try a larger perturbation
        if current_violations > 0 and random.random() < 0.05:
             # Flip a random edge anywhere
             u, v = random.sample(range(target_n), 2)
             if current_adj[u] & (1 << v):
                 current_adj[u] ^= (1 << v)
                 current_adj[v] ^= (1 << u)
             else:
                 current_adj[u] |= (1 << v)
                 current_adj[v] |= (1 << u)
             current_violations = count_violations(current_adj, target_n)

    # If we run out of time, return the best graph we found (even if invalid? No, must be valid).
    # The task says "A valid graph on n vertices certifies G(6) > n".
    # If we return an invalid graph, the score is 0.
    # So we must return a valid graph. If we didn't find one, we should return a smaller valid graph.
    # We know n=5 is valid. We can try to build up to 19, 18, etc.
    
    # Fallback: Return a known small valid graph
    # Let's try to quickly verify if we have a valid graph in best_adj
    if count_violations(best_adj, target_n) == 0:
        edges = []
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if best_adj[i] & (1 << j):
                    edges.append([i, j])
        return {'n': target_n, 'edges': edges}
    
    # If no valid graph of size 20 found, try size 19, 18...
    for n_try in range(target_n - 1, 4, -1):
        if time.time() > deadline - 5:
            break
        # Quick check: can we find a valid graph of size n_try?
        # Use the same logic but with smaller n.
        # This might be slow, so we limit attempts.
        found = False
        for _ in range(20):
            adj = [0] * n_try
            for i in range(n_try):
                for j in range(i + 1, n_try):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            if count_violations(adj, n_try) == 0:
                edges = []
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n_try, 'edges': edges}
                found = True
                break
        
    # Ultimate fallback: n=5
    return {'n': 5, 'edges': []}
