import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    def find_violation(adj, n):
        """Find one subset of size >= 6 that induces a regular subgraph.
        Returns the subset as a bitmask, or None if no violation found."""
        # Enumerate all masks of size >= 6
        # For efficiency, iterate by size and use Gosper's hack or simple enumeration
        for size in range(6, n + 1):
            # Generate all combinations of 'size' vertices from n
            # Use itertools for clarity, but for n=20 this is manageable
            from itertools import combinations
            for combo in combinations(range(n), size):
                mask = 0
                for v in combo:
                    mask |= (1 << v)
                # Check if induced subgraph on mask is regular
                degs = set()
                for v in combo:
                    deg = (adj[v] & mask).bit_count()
                    degs.add(deg)
                    if len(degs) > 1:
                        break
                if len(degs) <= 1:
                    return mask
        return None
    
    def is_valid(adj, n):
        """Check if graph has no regular induced subgraph of size >= 6."""
        return find_violation(adj, n) is None
    
    def flip_edge(adj, u, v):
        """Flip edge (u,v) in adjacency list."""
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    
    def local_search(adj, n, max_iter=200):
        """Local search: repeatedly find a violation and flip an edge in it."""
        adj = adj[:]
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            violation = find_violation(adj, n)
            if violation is None:
                return adj, True  # Valid!
            
            # Get the vertices in the violating subset
            vertices = []
            m = violation
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                vertices.append(v)
                m ^= lsb
            
            # Try flipping edges within the violating subset
            # Pick a random pair from the subset
            random.shuffle(vertices)
            found_improvement = False
            for i in range(len(vertices)):
                for j in range(i + 1, len(vertices)):
                    u, v = vertices[i], vertices[j]
                    # Save state
                    old_adj = adj[:]
                    flip_edge(adj, u, v)
                    
                    # Check if this flip removes the violation (or at least doesn't add new ones)
                    # For speed, just check if the original violation is gone
                    # and do a quick check for any new violation
                    new_violation = find_violation(adj, n)
                    if new_violation is None:
                        return adj, True  # Found a valid graph!
                    elif new_violation != violation:
                        # The violation changed, which might be progress
                        # Accept with some probability (simulated annealing-like)
                        if random.random() < 0.3:
                            found_improvement = True
                            break
                    else:
                        # Same violation, revert
                        adj = old_adj
                        continue
                if found_improvement:
                    break
            
            if not found_improvement:
                # No improvement in this iteration, try a random edge flip
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    flip_edge(adj, u, v)
        
        return adj, False
    
    # Try n=20 with multiple restarts
    n = 20
    best_adj = None
    best_valid = False
    
    for seed in range(50):
        if time.time() > deadline - 10:
            break
        
        random.seed(seed)
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Quick check: is it already valid?
        if is_valid(adj, n):
            best_adj = adj
            best_valid = True
            break
        
        # Do local search
        adj_ls, valid = local_search(adj, n, max_iter=100)
        if valid:
            best_adj = adj_ls
            best_valid = True
            break
    
    if best_valid and best_adj is not None:
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if best_adj[i] & (1 << j):
                    edges.append([i, j])
        return {'n': n, 'edges': edges}
    
    # Fallback: try n=19
    n = 19
    for seed in range(30):
        if time.time() > deadline - 5:
            break
        random.seed(seed)
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        if is_valid(adj, n):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        
        adj_ls, valid = local_search(adj, n, max_iter=80)
        if valid:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj_ls[i] & (1 << j):
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
    
    # Last resort: small valid graph
    return {'n': 5, 'edges': []}
