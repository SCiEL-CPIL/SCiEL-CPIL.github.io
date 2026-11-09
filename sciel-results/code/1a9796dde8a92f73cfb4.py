import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    def is_valid(n, edges):
        """Check if a graph has no regular induced subgraph on 6+ vertices."""
        # Build adjacency matrix as list of bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        # Check all subsets of size k >= 6
        for k in range(6, n + 1):
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                # Check if all vertices in subset have the same degree within the subset
                verts = list(subset)
                d0 = (adj[verts[0]] & mask).bit_count()
                regular = True
                for v in verts[1:]:
                    d = (adj[v] & mask).bit_count()
                    if d != d0:
                        regular = False
                        break
                if regular:
                    return False
        return True

    def count_new_violations(current_adj, n_current, new_vertex_conn, target_n=6):
        """
        Count how many regular induced subgraphs of size exactly 6 would be created
        by adding a new vertex with the given connections to existing vertices.
        
        current_adj: list of bitmasks for existing vertices
        n_current: number of existing vertices
        new_vertex_conn: bitmask of which existing vertices the new vertex connects to
        """
        violations = 0
        
        # The new vertex is vertex n_current
        # Any subset of size 5 from existing vertices + the new vertex forms a size-6 subset
        # We need to check if such a subset is regular
        
        # For each 5-subset of existing vertices, check if adding the new vertex makes it regular
        # To be efficient, we only need to check subsets where the 5 existing vertices
        # have degrees that could potentially become regular with the new vertex
        
        for subset5 in combinations(range(n_current), 5):
            mask5 = 0
            for v in subset5:
                mask5 |= (1 << v)
            
            # Degrees of the 5 vertices within the 5-subset
            degs = []
            for v in subset5:
                d = (current_adj[v] & mask5).bit_count()
                degs.append(d)
            
            # The new vertex's degree within this 6-subset is:
            # number of connections from new vertex to the 5 vertices
            new_deg = 0
            for v in subset5:
                if (new_vertex_conn >> v) & 1:
                    new_deg += 1
            
            # For the 6-subset to be regular, all 6 vertices must have the same degree
            # The 5 existing vertices have degrees: degs[i] + (1 if connected to new vertex else 0)
            # Wait, no. The degree of an existing vertex v within the 6-subset is:
            # (number of neighbors of v among the other 4 existing vertices in subset5)
            # + (1 if v is connected to the new vertex)
            # But degs[i] as computed above is the degree within the 5-subset, which includes
            # connections to all other 4 vertices in subset5. So we need to add 1 if connected to new vertex.
            
            # Let's recompute properly:
            # For vertex v in subset5, its degree in the 6-subset = 
            #   (adj[v] & mask5).bit_count()  [connections to other 4 in subset5]
            #   + (1 if (new_vertex_conn >> v) & 1 else 0)  [connection to new vertex]
            
            all_same = True
            target_deg = None
            for i, v in enumerate(subset5):
                d_in_6 = (current_adj[v] & mask5).bit_count()
                if (new_vertex_conn >> v) & 1:
                    d_in_6 += 1
                if target_deg is None:
                    target_deg = d_in_6
                elif d_in_6 != target_deg:
                    all_same = False
                    break
            
            if all_same:
                # Now check if the new vertex's degree matches
                if new_deg == target_deg:
                    violations += 1
        
        return violations

    def greedy_build(target_n, seed):
        """
        Greedily build a graph on target_n vertices by adding one vertex at a time.
        For each new vertex, try all possible connection patterns to existing vertices
        and pick the one that minimizes new regular 6-subsets.
        """
        random.seed(seed)
        
        adj = []  # list of bitmasks
        edges = []
        
        for i in range(target_n):
            if i == 0:
                adj.append(0)
                continue
            
            n_current = i
            best_conn = 0
            best_violations = float('inf')
            
            # Try all possible connection patterns to existing vertices
            # There are 2^i possible patterns, but for i up to 20 that's too many
            # Instead, use a heuristic: try random patterns and the all-zero/all-one patterns
            
            # For efficiency, we'll try a limited set of patterns
            patterns_to_try = set()
            
            # Always try no edges and all edges
            patterns_to_try.add(0)
            patterns_to_try.add((1 << n_current) - 1)
            
            # Try some random patterns
            num_random = min(20, (1 << n_current))
            for _ in range(num_random):
                patterns_to_try.add(random.getrandbits(n_current))
            
            # Try patterns with low density and high density
            for density in [0.1, 0.3, 0.5, 0.7, 0.9]:
                conn = 0
                for j in range(n_current):
                    if random.random() < density:
                        conn |= (1 << j)
                patterns_to_try.add(conn)
            
            # Evaluate each pattern
            for conn in patterns_to_try:
                violations = count_new_violations(adj, n_current, conn)
                if violations < best_violations:
                    best_violations = violations
                    best_conn = conn
            
            # Add the new vertex with the best connection pattern
            adj.append(best_conn)
            for j in range(n_current):
                if (best_conn >> j) & 1:
                    edges.append([j, i])
            
            # Check time
            if time.time() > deadline:
                break
        
        return {'n': len(adj), 'edges': edges}

    # Try greedy construction with different seeds
    best_result = None
    best_n = 0
    
    for seed in range(50):
        if time.time() > deadline:
            break
        
        result = greedy_build(22, seed)
        n = result['n']
        
        if n > best_n:
            best_n = n
            best_result = result
        
        # If we got a full 22-vertex graph, verify it
        if n == 22:
            if is_valid(22, result['edges']):
                return result
        
        # If we got 21, verify it
        if n == 21:
            if is_valid(21, result['edges']):
                if best_n < 21:
                    best_n = 21
                    best_result = result
                # Don't return immediately; try to get 22
    
    # If we have a valid result, return it
    if best_result is not None and best_n >= 6:
        # Final verification for safety
        if is_valid(best_n, best_result['edges']):
            return best_result
    
    # Fallback: try random graphs for smaller n
    for target_n in [20, 19, 18, 17]:
        if time.time() > deadline:
            break
        for seed in range(20):
            if time.time() > deadline:
                break
            random.seed(seed)
            edges = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.5:
                        edges.append([i, j])
            if is_valid(target_n, edges):
                return {'n': target_n, 'edges': edges}
    
    # Last resort: return a small valid graph
    # Empty graph on 5 vertices is valid (no 6-vertex subsets)
    return {'n': 5, 'edges': []}
