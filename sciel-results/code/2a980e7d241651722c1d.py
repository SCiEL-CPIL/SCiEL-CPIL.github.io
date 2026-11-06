import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Try to find a valid graph on n vertices
    # The best known is 39. We aim for 39.
    for n in [39, 38, 37, 36]:
        result = try_construct(n, deadline)
        if result:
            return result
            
    # Fallback to a known valid smaller graph if all else fails
    # Ten disjoint triangles: n=30, alpha=10
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}

def try_construct(n, deadline):
    # Bitmask representation for neighbors
    neighbors = [0] * n
    
    # Helper to check if adding edge (u, v) creates a C4
    def can_add_edge(u, v):
        if (neighbors[u] >> v) & 1:
            return False # Already connected
        if neighbors[u] & neighbors[v]:
            return False # Common neighbor exists -> C4
        return True

    def add_edge(u, v):
        neighbors[u] |= (1 << v)
        neighbors[v] |= (1 << u)

    # Helper to check if there is an independent set of size k
    # Returns True if found, False if proven not to exist or timeout
    # We use Branch and Bound with reduction.
    def has_independent_set_of_size(k):
        # k is the target size, e.g., 11
        # If we find an independent set of size k, return True.
        # If we exhaust the search and don't find one, return False.
        
        best_size = 0 # Not really needed if we just want to find size k, but good for tracking
        
        def bnb(vertices, current_size):
            nonlocal best_size
            if time.time() > deadline:
                raise TimeoutError()
            
            if current_size >= k:
                return True
            
            if not vertices:
                return False
            
            # Pruning: Even if we take all remaining, we can't reach k
            if current_size + vertices.bit_count() < k:
                return False
            
            # Reduction: Isolated vertices
            # If a vertex has no neighbors in 'vertices', it must be in any maximal independent set?
            # No, it's independent of all others in S. So we can include it.
            # Including it increases current_size by 1 and removes it from vertices.
            changed = True
            while changed:
                changed = False
                temp = vertices
                while temp:
                    v = (temp & -temp).bit_length() - 1
                    temp &= temp - 1
                    if (neighbors[v] & vertices) == 0:
                        # v is isolated in S
                        current_size += 1
                        vertices &= ~(1 << v)
                        changed = True
                        if current_size >= k:
                            return True
                        if current_size + vertices.bit_count() < k:
                            return False
                        break # Restart the while loop to check new state
            
            if not vertices:
                return False
            
            # Pick a vertex to branch on
            # Heuristic: Pick vertex with highest degree in 'vertices' to prune better?
            # Or lowest index?
            # Let's pick the one with highest degree in the induced subgraph.
            # This is O(k^2).
            max_deg = -1
            pick = -1
            temp = vertices
            while temp:
                v = (temp & -temp).bit_length() - 1
                temp &= temp - 1
                deg = (neighbors[v] & vertices).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    pick = v
            
            # Branch 1: Include pick
            # New vertices = vertices without pick and without N(pick)
            new_vertices = vertices & ~(neighbors[pick] | (1 << pick))
            if bnb(new_vertices, current_size + 1):
                return True
            
            # Branch 2: Exclude pick
            new_vertices = vertices & ~(1 << pick)
            if bnb(new_vertices, current_size):
                return True
                
            return False

        try:
            return bnb((1 << n) - 1, 0)
        except TimeoutError:
            return None # Timeout

    # Randomized Greedy Construction
    random.seed(42) # Fixed seed for reproducibility in debugging, but maybe random is better for exploration?
    # Let's use time-based seed for diversity
    random.seed(int(time.time() * 1000) % 100000)
    
    for attempt in range(100): # More attempts
        if time.time() > deadline:
            break
            
        neighbors = [0] * n
        edge_count = 0
        
        # 1. Greedy Random Edge Addition
        potential_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(potential_edges)
        
        for u, v in potential_edges:
            if can_add_edge(u, v):
                add_edge(u, v)
                edge_count += 1
            if time.time() > deadline:
                break
        
        # 2. Local Search to reduce alpha
        improved = True
        while improved:
            if time.time() > deadline:
                break
                
            # Check if alpha < 11 (i.e., no independent set of size 11)
            # If has_independent_set_of_size(11) is False, then alpha <= 10.
            # If it is True, we need to add more edges.
            # If it is None (timeout), we assume it's still bad and try to add edges blindly?
            # Or break?
            
            has_11 = has_independent_set_of_size(11)
            
            if has_11 is False:
                # Success! No independent set of size 11.
                edges = []
                for i in range(n):
                    mask = neighbors[i]
                    while mask:
                        j = (mask & -mask).bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask &= mask - 1
                return {'n': n, 'edges': edges}
            
            if has_11 is None:
                # Timeout in checker. Assume we are not done.
                # We can try to add edges based on a greedy independent set.
                pass

            # Find a large independent set to target
            # We need a set to add edges within.
            # Use a greedy independent set.
            def find_greedy_independent_set(vertices):
                remaining = vertices
                s = set()
                while remaining:
                    # Pick vertex with min degree in remaining
                    min_deg = float('inf')
                    pick = -1
                    temp = remaining
                    while temp:
                        v = (temp & -temp).bit_length() - 1
                        temp &= temp - 1
                        deg = (neighbors[v] & remaining).bit_count()
                        if deg < min_deg:
                            min_deg = deg
                            pick = v
                    s.add(pick)
                    remaining &= ~(neighbors[pick] | (1 << pick))
                return s
            
            S = find_greedy_independent_set((1 << n) - 1)
            
            if len(S) < 11:
                # Greedy found < 11, but B&B said there IS one of size 11.
                # This is a conflict. B&B is exact, Greedy is not.
                # If B&B says True, there is one.
                # We should try to find it or just add edges randomly in the graph?
                # Let's add edges between random pairs that are non-adjacent.
                pass

            added = False
            S_list = list(S)
            random.shuffle(S_list)
            
            # Add edges within S
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    u = S_list[i]
                    v = S_list[j]
                    if can_add_edge(u, v):
                        add_edge(u, v)
                        added = True
                        if time.time() > deadline:
                            break
                if time.time() > deadline:
                    break
                    
            if not added:
                # No more edges can be added in S.
                # Try adding edges elsewhere?
                # Or restart?
                improved = False
            else:
                improved = True
                
    return None
