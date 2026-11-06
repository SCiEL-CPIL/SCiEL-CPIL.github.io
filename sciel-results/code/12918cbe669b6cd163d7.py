import time
import random

def construct():
    """
    Return a C4-free graph with alpha < 11, maximizing n.
    
    Strategy: Use randomized greedy construction starting from a C4-free base graph,
    iteratively attempting to add vertices by sampling random neighbor sets that
    maintain C4-freeness and keep alpha < 11.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # First, build a good base: polarity graph of PG(2,5) with 31 vertices
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, -1, 5)
                    points.append((1, (y * inv_x) % 5, (z * inv_x) % 5))
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    points.append((0, 1, (z * inv_y) % 5))
                else:
                    points.append((0, 0, 1))
    points = list(set(points))
    n_base = len(points)
    
    def bilinear(u, v):
        return (u[0] * v[2] + u[1] * v[1] + u[2] * v[0]) % 5
    
    adj_base = [0] * n_base
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if bilinear(points[i], points[j]) == 0:
                adj_base[i] |= (1 << j)
                adj_base[j] |= (1 << i)
    
    # Fast C4 check using bitmasks
    def is_c4_free(n, adj):
        for i in range(n):
            for j in range(i + 1, n):
                common = adj[i] & adj[j]
                if common & (common - 1):  # more than 1 bit set
                    return False
        return True
    
    # Fast independence number check (returns True if alpha >= 11)
    def has_independent_set_of_size_11(n, adj):
        # Use a greedy + backtracking hybrid
        # First try greedy to find a large independent set
        def greedy_is_size(n, adj):
            # Greedy: repeatedly pick vertex with min degree in remaining graph
            remaining = set(range(n))
            is_set = set()
            while remaining:
                # Find vertex with min degree in remaining subgraph
                min_deg = float('inf')
                best_v = None
                for v in remaining:
                    deg = len([u for u in remaining if u != v and (adj[v] >> u) & 1])
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                if best_v is None:
                    break
                is_set.add(best_v)
                remaining.remove(best_v)
                # Remove all neighbors of best_v from remaining
                neighbors = [u for u in remaining if (adj[best_v] >> u) & 1]
                for u in neighbors:
                    remaining.remove(u)
            return len(is_set)
        
        greedy_size = greedy_is_size(n, adj)
        if greedy_size < 11:
            # Greedy didn't find 11, but there might still be one
            # Use a more thorough search
            pass
        
        # More thorough: try to find independent set of size 11 using backtracking
        # with pruning
        found = [False]
        
        def backtrack(candidates, current_size, current_set):
            if found[0]:
                return
            if current_size >= 11:
                found[0] = True
                return
            if not candidates:
                return
            if current_size + len(candidates) < 11:
                return
            
            # Pick vertex with highest degree in candidate subgraph for better pruning
            best_v = None
            best_deg = -1
            for v in candidates:
                deg = sum(1 for u in candidates if u != v and (adj[v] >> u) & 1)
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            
            if best_v is None:
                return
            
            # Try including best_v
            new_candidates = [u for u in candidates if u != best_v and not ((adj[best_v] >> u) & 1)]
            backtrack(new_candidates, current_size + 1, current_set | {best_v})
            
            if found[0]:
                return
            
            # Try excluding best_v
            backtrack([u for u in candidates if u != best_v], current_size, current_set)
        
        backtrack(list(range(n)), 0, set())
        return found[0]
    
    # Start with base graph
    best_n = n_base
    best_adj = adj_base[:]
    best_edges = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if adj_base[i] & (1 << j):
                best_edges.append([i, j])
    
    # Try to extend the graph by adding vertices
    # We'll try multiple random strategies
    
    target_n = 43  # Max possible
    max_extend = 43 - n_base
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Reset to base
        adj = adj_base[:]
        n = n_base
        
        # Try to add vertices one by one
        for new_idx in range(max_extend):
            if time.time() > deadline:
                break
            
            candidate_n = n + 1
            candidate_adj = adj + [0]
            
            # The new vertex (index n) needs to be connected to some subset of existing vertices
            # such that:
            # 1. No C4 is created
            # 2. alpha remains < 11
            
            # For C4-freeness: if new vertex connects to vertices a and b,
            # then a and b must not have a common neighbor (other than the new vertex)
            # Actually, more precisely: for any two neighbors a, b of the new vertex,
            # N(a) ∩ N(b) must be empty (in the original graph)
            
            # Let's try different connection patterns
            # Pattern 1: connect to no vertices (isolated)
            # Pattern 2: connect to 1 vertex
            # Pattern 3: connect to 2 vertices that don't share neighbors
            # etc.
            
            success = False
            
            # Try random subsets of neighbors
            # Start with small subsets
            
            # First try: isolated vertex
            if is_c4_free(candidate_n, candidate_adj):
                if not has_independent_set_of_size_11(candidate_n, candidate_adj):
                    success = True
                    adj = candidate_adj
                    n = candidate_n
                    continue
            
            # Try connecting to 1 random vertex
            for _ in range(10):
                if time.time() > deadline:
                    break
                v = random.randint(0, n - 1)
                candidate_adj = adj + [0]
                candidate_adj[v] |= (1 << n)
                candidate_adj[n] |= (1 << v)
                
                if is_c4_free(candidate_n, candidate_adj):
                    if not has_independent_set_of_size_11(candidate_n, candidate_adj):
                        success = True
                        adj = candidate_adj
                        n = candidate_n
                        break
            
            if success:
                continue
            
            # Try connecting to 2 vertices that don't share common neighbors
            for _ in range(20):
                if time.time() > deadline:
                    break
                v1 = random.randint(0, n - 1)
                v2 = random.randint(0, n - 1)
                if v1 == v2:
                    continue
                
                # Check if v1 and v2 share a common neighbor
                common = adj[v1] & adj[v2]
                if common:
                    continue
                
                candidate_adj = adj + [0]
                candidate_adj[v1] |= (1 << n)
                candidate_adj[v2] |= (1 << n)
                candidate_adj[n] |= (1 << v1) | (1 << v2)
                
                if is_c4_free(candidate_n, candidate_adj):
                    if not has_independent_set_of_size_11(candidate_n, candidate_adj):
                        success = True
                        adj = candidate_adj
                        n = candidate_n
                        break
            
            if not success:
                break
        
        # Update best if we did better
        if n > best_n:
            best_n = n
            best_adj = adj[:]
            best_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i] & (1 << j):
                        best_edges.append([i, j])
    
    # Final verification
    if not is_c4_free(best_n, best_adj):
        # Fall back to base
        best_n = n_base
        best_adj = adj_base[:]
        best_edges = []
        for i in range(n_base):
            for j in range(i + 1, n_base):
                if adj_base[i] & (1 << j):
                    best_edges.append([i, j])
    
    return {'n': best_n, 'edges': best_edges}
