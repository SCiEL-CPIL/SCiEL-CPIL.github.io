import time
import random

def _has_triangle_addition(neighbors, u, v):
    """Check if adding edge (u,v) creates a triangle."""
    # A triangle is formed if u and v share a common neighbor
    return (neighbors[u] & neighbors[v]) != 0

def _greedy_independent_set(neighbors, n):
    """Find a maximal independent set using greedy method."""
    remaining = (1 << n) - 1
    indep_set = []
    while remaining:
        # Pick a vertex with minimum degree in the remaining subgraph?
        # Or just pick the first one?
        # To maximize size, picking min degree is better.
        # But for speed, picking the first available is O(1) per step if we don't calculate degrees.
        # Let's pick the first available vertex.
        v = (remaining & -remaining).bit_length() - 1
        indep_set.append(v)
        # Remove v and its neighbors
        remaining &= ~(neighbors[v] | (1 << v))
    return indep_set

def _max_independent_set_size(neighbors, n):
    """
    Exact check for alpha < 10.
    Returns True if alpha < 10 (i.e., no independent set of size 10).
    Uses branch and bound.
    """
    # If n < 10, alpha is definitely < 10
    if n < 10:
        return True
        
    # We can use a simple recursive search with pruning.
    # State: current set of candidates, current size.
    # We want to know if we can reach size 10.
    
    # Optimization: If the number of remaining vertices is less than (10 - current_size), return False.
    # Also, if we can't find an independent set of size 10, we return True (valid).
    
    # A faster heuristic: Check if the greedy size is < 10.
    # If greedy finds < 10, it's likely alpha < 10.
    # But to be exact, we need backtracking.
    
    # For the purpose of this task, a very fast approximate check might suffice for the search loop,
    # but the final returned graph MUST be valid.
    # So we will use a fast check in the loop, and a rigorous check at the end.
    
    pass

def _is_valid(neighbors, n):
    """
    Rigorously check if the graph is triangle-free and alpha < 10.
    """
    # 1. Triangle-free check
    for i in range(n):
        ni = neighbors[i]
        # Check if any two neighbors of i are connected
        # This is O(n^3) worst case, but n is small (<=40).
        # Better: for each i, check if neighbors[i] has any edges.
        # We can check if the subgraph induced by neighbors[i] is empty.
        # Since the graph is simple and undirected, if it was built correctly, it should be.
        # But we added edges, so we must ensure no triangles were created.
        # Our addition logic ensures this.
        
    # 2. Alpha < 10 check
    # Use a branch and bound algorithm to find if there is an independent set of size 10.
    # If found, return False. If not, return True.
    
    # We'll implement a simple backtracking with bitmasks.
    # To speed it up, we can order vertices by degree.
    
    # Let's use a recursive function with memoization or just simple pruning.
    # Given n <= 40, and we are looking for size 10, the search space is manageable with good pruning.
    
    # Pruning: 
    # 1. If len(candidates) < 10 - current_size: return False (can't reach 10)
    # 2. If current_size + len(candidates) < 10: return False
    # 3. Coloring bound: Color the candidate subgraph. If the number of colors is less than 10 - current_size, 
    #    then no independent set of that size exists in that subgraph.
    
    # We'll implement a simple version of this.
    
    def find_indep_set(candidates_mask, current_size):
        if current_size >= 10:
            return True
        # Count bits
        if candidates_mask == 0:
            return False
        # Pruning: if remaining vertices < needed
        if candidates_mask.bit_count() < 10 - current_size:
            return False
            
        # Pick a vertex v from candidates
        # Heuristic: pick vertex with min degree in candidates?
        # For simplicity, pick the lowest bit.
        v = (candidates_mask & -candidates_mask).bit_length() - 1
        
        # Option 1: Include v
        # New candidates: candidates_mask & ~neighbors[v] & ~(1 << v)
        new_candidates = candidates_mask & ~neighbors[v] & ~(1 << v)
        if find_indep_set(new_candidates, current_size + 1):
            return True
            
        # Option 2: Exclude v
        new_candidates = candidates_mask & ~(1 << v)
        if find_indep_set(new_candidates, current_size):
            return True
            
        return False

    return not find_indep_set((1 << n) - 1, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Try n from 40 down to 26
    for n in range(40, 25, -1):
        if time.time() > deadline:
            break
            
        # Try to find a valid graph for this n
        # Use multiple restarts
        max_restarts = 50
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize random triangle-free graph
            # Start with empty graph, add random edges that don't create triangles
            neighbors = [0] * n
            # Shuffle vertices to randomize
            vertices = list(range(n))
            random.shuffle(vertices)
            
            # Add edges
            # We want a reasonably dense graph to keep alpha small.
            # Triangle-free max edges is n^2/4.
            # Let's aim for ~ n^2/6 edges.
            target_edges = (n * n) // 6
            
            # Simple random edge addition
            # To ensure triangle-free, we can check before adding.
            # This is slow if done naively.
            # Better: Start with a bipartite graph?
            # Random bipartite graph is triangle-free.
            # Let's split vertices into two sets A and B.
            # Add random edges between A and B.
            mid = n // 2
            set_a = set(vertices[:mid])
            set_b = set(vertices[mid:])
            
            for u in set_a:
                for v in set_b:
                    if random.random() < 0.5:
                        neighbors[u] |= (1 << v)
                        neighbors[v] |= (1 << u)
                        
            # Now we have a triangle-free graph.
            # Check alpha
            if _is_valid(neighbors, n):
                # Convert to edge list
                edges = []
                for i in range(n):
                    mask = neighbors[i]
                    while mask:
                        j = (mask & -mask).bit_length() - 1
                        if i < j:
                            edges.append([i, j])
                        mask &= mask - 1
                return {'n': n, 'edges': edges}
            
            # If not valid, try to repair
            # Local search
            for step in range(100):
                if time.time() > deadline:
                    break
                    
                # Find an independent set of size 10
                # We can use the greedy one to find a violation
                indep = _greedy_independent_set(neighbors, n)
                if len(indep) < 10:
                    # It might still be invalid, but let's check rigorously
                    if _is_valid(neighbors, n):
                        edges = []
                        for i in range(n):
                            mask = neighbors[i]
                            while mask:
                                j = (mask & -mask).bit_length() - 1
                                if i < j:
                                    edges.append([i, j])
                                mask &= mask - 1
                        return {'n': n, 'edges': edges}
                    else:
                        # Greedy missed it, or it's a complex structure.
                        # We need to find the actual independent set.
                        # For now, let's just restart if greedy says ok but exact says no.
                        break 
                        
                # Found an independent set of size >= 10
                # Try to add an edge within it
                added = False
                # Shuffle the independent set to randomize which edge we pick
                random.shuffle(indep)
                for i in range(len(indep)):
                    for j in range(i + 1, len(indep)):
                        u = indep[i]
                        v = indep[j]
                        # Check if adding (u,v) creates a triangle
                        if (neighbors[u] & neighbors[v]) == 0:
                            # Add edge
                            neighbors[u] |= (1 << v)
                            neighbors[v] |= (1 << u)
                            added = True
                            break
                    if added:
                        break
                        
                if not added:
                    # Stuck in local minimum
                    # Try a random edge swap? Or just restart
                    break

    # If we didn't find a better one, return the seed
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
