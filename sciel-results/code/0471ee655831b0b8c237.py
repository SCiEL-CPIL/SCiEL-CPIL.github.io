import random
import time

def construct():
    n = 16
    # 1. Base Graph: Grötzsch Graph (Mycielski of C5)
    # Vertices 0-4: C5 cycle
    # Vertices 5-9: Copies of 0-4
    # Vertex 10: Root
    base_edges = []
    # C5
    for i in range(5):
        base_edges.append((i, (i + 1) % 5))
    # Mycielski step
    for i in range(5):
        u = i
        v = (i + 1) % 5
        base_edges.append((u, 5 + v))
        base_edges.append((v, 5 + u))
    for i in range(5):
        base_edges.append((5 + i, 10))
        
    # Initialize adjacency for base graph
    adj = {i: set() for i in range(n)}
    for u, v in base_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    # Candidate edges: all edges involving vertices 11-15
    candidates = []
    for u in range(11, 16):
        for v in range(16):
            if v > u:
                # Skip if already in base (none are, as base is 0-10)
                # But check if u,v adjacent in base? No, base has no 11-15.
                candidates.append((u, v))
                
    # Note: candidates includes edges between 11-15 and 0-10, and 11-15 among themselves.
    # Total candidates: 
    # 5 * 11 (11-15 to 0-10) + 10 (11-15 among themselves) = 55 + 10 = 65.
    
    def is_triangle_free(edge_set):
        # edge_set is a set of tuples (u, v) with u < v
        # Check all edges
        # For efficiency, we can check during addition, but for GA repair, we need full check.
        # n=16 is small.
        # Build adj
        adj_check = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj_check[u].add(v)
            adj_check[v].add(u)
        for u in range(n):
            for v in adj_check[u]:
                if u < v:
                    common = adj_check[u] & adj_check[v]
                    if common:
                        return False
        return True

    def repair(edge_set):
        # Remove edges that create triangles
        # Iterate until no change
        changed = True
        while changed:
            changed = False
            adj_temp = {i: set() for i in range(n)}
            for u, v in edge_set:
                adj_temp[u].add(v)
                adj_temp[v].add(u)
            
            new_edge_set = set()
            for u, v in edge_set:
                if u > v: u, v = v, u
                # Check if adding (u,v) creates triangle
                # It creates triangle if N(u) and N(v) share a common neighbor
                # But we are removing from existing set.
                # Simpler: check if (u,v) is part of a triangle in the CURRENT set.
                common = adj_temp[u] & adj_temp[v]
                # If common, there is w such that (u,w) and (v,w) are in set.
                # We need to remove one. Let's remove (u,v) if it's "newest" or just arbitrary.
                # To be deterministic, remove (u,v) if u > 10 or v > 10?
                # Or just remove it.
                if common:
                    # Remove (u,v)
                    # We mark it for removal
                    pass # Logic below
                else:
                    new_edge_set.add((u, v))
            
            # This simple pass might not remove all if multiple triangles share edges.
            # Better: 
            # For each edge, if it forms a triangle, remove it.
            # Repeat.
            
            # Re-implement repair more carefully
            # We want to keep as many edges as possible.
            # This is a Maximum Independent Set on the conflict graph of triangles?
            # Given small n, we can just greedily remove edges that are in triangles.
            
            # Let's use a different repair:
            # Start with all edges.
            # While there is a triangle, remove one of its edges.
            # To maximize edges, we should remove the edge that is in the most triangles?
            # Or the edge with highest "cost"?
            # Here all edges have same cost.
            
            # Simple repair:
            # 1. Build adj.
            # 2. Find all triangles.
            # 3. Remove an edge from each triangle.
            # 4. Repeat.
            
            # For speed in GA, we can just check if the set is triangle-free.
            # If not, remove a random edge that is part of a triangle.
            
            if not is_triangle_free(edge_set):
                # Find a triangle
                adj_check = {i: set() for i in range(n)}
                for u, v in edge_set:
                    adj_check[u].add(v)
                    adj_check[v].add(u)
                removed = False
                for u in range(n):
                    if removed: break
                    for v in adj_check[u]:
                        if u < v and (u, v) in edge_set:
                            common = adj_check[u] & adj_check[v]
                            if common:
                                # Remove (u, v)
                                edge_set.discard((u, v))
                                removed = True
                                break
                if not removed:
                    break # Should not happen if is_triangle_free is consistent
            else:
                break
                
        return edge_set

    def greedy_fill(edge_set):
        # Add any candidate edge that doesn't create a triangle
        adj_check = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj_check[u].add(v)
            adj_check[v].add(u)
            
        for u, v in candidates:
            if (u, v) not in edge_set:
                # Check if adding (u,v) creates triangle
                if not (adj_check[u] & adj_check[v]):
                    edge_set.add((u, v))
                    adj_check[u].add(v)
                    adj_check[v].add(u)
        return edge_set

    # GA Setup
    pop_size = 50
    time_limit = 100
    start_time = time.time()
    
    # Initialize population
    population = []
    for _ in range(pop_size):
        # Random subset of candidates
        ind = set()
        for c in candidates:
            if random.random() < 0.5:
                ind.add(c)
        ind = repair(ind)
        ind = greedy_fill(ind)
        population.append(ind)
        
    best_ind = max(population, key=len)
    
    while time.time() - start_time < time_limit:
        new_population = []
        # Elitism
        new_population.append(best_ind)
        
        for _ in range(pop_size - 1):
            # Selection
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            # Crossover
            child = set()
            for c in candidates:
                if random.random() < 0.5:
                    child.add(c if c in p1 else c if c in p2 else c) # Simple: pick from p1 or p2
                else:
                    if c in p1: child.add(c)
                    elif c in p2: child.add(c)
            
            # Mutation
            for c in candidates:
                if random.random() < 0.1:
                    if c in child:
                        child.discard(c)
                    else:
                        child.add(c)
            
            # Repair and Fill
            child = repair(child)
            child = greedy_fill(child)
            
            new_population.append(child)
            
        population = new_population
        current_best = max(population, key=len)
        if len(current_best) > len(best_ind):
            best_ind = current_best
            
    # Convert best_ind to list of lists
    result = [[u, v] for u, v in best_ind]
    return result
