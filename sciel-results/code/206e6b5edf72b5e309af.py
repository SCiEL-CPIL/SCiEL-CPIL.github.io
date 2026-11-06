import time
import random

def _count_k4(adj, n):
    """Count K4s in the graph defined by adj (list of int bitmasks)."""
    total_triangles = 0
    for u in range(n):
        nu = adj[u]
        # Extract neighbors
        neighbors = []
        for i in range(n):
            if (nu >> i) & 1:
                neighbors.append(i)
        
        # Count triangles in neighbors
        # A triangle (v,w,x) in neighbors means u,v,w,x is a K4
        for i in range(len(neighbors)):
            v = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                w = neighbors[j]
                if (adj[v] >> w) & 1:
                    # v and w are connected.
                    # Check if there is a third neighbor x connected to both v and w?
                    # No, the loop finds edges (v,w) in the induced subgraph on N(u).
                    # Wait.
                    # A triangle in N(u) is a set of 3 vertices {v,w,x} such that all pairs are edges.
                    # My loop only checks if (v,w) is an edge. It doesn't check (v,x) or (w,x).
                    # So this counts edges in N(u), not triangles.
                    pass
                    
    # Correct way to count triangles in N(u):
    # For each pair (v,w) in N(u) that is an edge, count common neighbors x in N(u) such that x > w?
    # No, just count all triangles.
    # For each u:
    #   tri_count = 0
    #   for each edge (v,w) in N(u):
    #       common = adj[v] & adj[w] & adj[u]
    #       tri_count += popcount(common)
    #   total_triangles += tri_count
    #   # Each triangle in N(u) is counted once?
    #   # Triangle {v,w,x} in N(u).
    #   # Pairs: (v,w), (v,x), (w,x).
    #   # For pair (v,w), common includes x.
    #   # For pair (v,x), common includes w.
    #   # For pair (w,x), common includes v.
    #   # So each triangle is counted 3 times.
    #   # So triangles_in_Nu = tri_count // 3.
    
    k4_count = 0
    for u in range(n):
        nu = adj[u]
        tri_count = 0
        neighbors = [i for i in range(n) if (nu >> i) & 1]
        for i in range(len(neighbors)):
            v = neighbors[i]
            for j in range(i + 1, len(neighbors)):
                w = neighbors[j]
                if (adj[v] >> w) & 1:
                    common = adj[v] & adj[w] & adj[u]
                    tri_count += common.bit_count()
        k4_count += tri_count // 3
        
    return k4_count // 4

def _has_independent_set_6(adj, n):
    """Check if there is an independent set of size 6."""
    # Order vertices by degree (ascending? descending?)
    # For independent set, picking high degree vertices first might prune better?
    # If we pick a high degree vertex, we exclude many neighbors.
    # Let's sort by degree descending.
    degrees = [adj[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    # Precompute masks in order?
    # We will use a recursive function with a mask of available vertices.
    
    def dfs(available_mask, current_size):
        if current_size == 6:
            return True
        if available_mask == 0:
            return False
        # Prune: if current_size + popcount(available_mask) < 6
        if current_size + available_mask.bit_count() < 6:
            return False
            
        # Pick a vertex from available_mask
        # Use the order to pick the "best" vertex?
        # Simple: pick the first set bit in the mask, but mapped to original index?
        # Better: iterate through the global order, find the first vertex in available_mask.
        v = -1
        for idx in order:
            if (available_mask >> idx) & 1:
                v = idx
                break
        if v == -1:
            return False
            
        # Branch 1: Include v
        # New available: available_mask & ~adj[v] & ~(1 << v)
        if dfs(available_mask & ~adj[v] & ~(1 << v), current_size + 1):
            return True
            
        # Branch 2: Exclude v
        if dfs(available_mask & ~(1 << v), current_size):
            return True
            
        return False
        
    full_mask = (1 << n) - 1
    return dfs(full_mask, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Try n=36 down to 17
    for n in range(36, 16, -1):
        if time.time() > deadline - 10:
            break
            
        # SA Parameters
        T = 1.0
        cool = 0.99
        best_valid = None
        
        # Init
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Calculate initial cost
        def get_cost(adj, n):
            k4 = _count_k4(adj, n)
            if k4 > 0:
                return k4
            if _has_independent_set_6(adj, n):
                return 1
            return 0
        
        current_cost = get_cost(adj, n)
        if current_cost == 0:
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
        best_cost = current_cost
        best_adj = adj[:]
        
        steps = 0
        max_steps = 50000 # Adjust based on time
        
        while time.time() < deadline - 5:
            # Pick a random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
                
            # Flip
            if (adj[u] >> v) & 1:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                new_adj = adj
                # Revert for calculation? No, calculate cost, then revert if worse
            else:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                new_adj = adj
                
            new_cost = get_cost(new_adj, n)
            
            # Accept if better, or with prob exp(-delta/T)
            delta = new_cost - current_cost
            if delta < 0:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = new_adj[:]
                    if best_cost == 0:
                        edges = []
                        for i in range(n):
                            for j in range(i + 1, n):
                                if (best_adj[i] >> j) & 1:
                                    edges.append([i, j])
                        return {'n': n, 'edges': edges}
            else:
                # Prob accept
                if random.random() < pow(2.718, -delta / T):
                    current_cost = new_cost
                else:
                    # Revert
                    if (new_adj[u] >> v) & 1:
                        new_adj[u] &= ~(1 << v)
                        new_adj[v] &= ~(1 << u)
                    else:
                        new_adj[u] |= (1 << v)
                        new_adj[v] |= (1 << u)
                        
            T *= cool
            steps += 1
            
        # If loop finished without finding valid, try next n?
        # Or keep best_adj?
        # We only return valid graphs.
        
    # Fallback
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
