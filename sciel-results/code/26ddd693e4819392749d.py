import random
import time

def construct():
    deadline = time.time() + 110  # 10s margin for safety
    n = 20
    num_colors = 20
    
    best_edges = None
    best_count = 0
    
    # Pre-compute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def greedy_build(seed, initial_edges=None, order_type='random'):
        """
        Builds a graph using greedy edge addition.
        If initial_edges is provided, it starts with those edges and tries to add more.
        """
        rng = random.Random(seed)
        
        # Initialize adjacency and neighbors
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        # If we have initial edges, add them first
        if initial_edges:
            for e in initial_edges:
                u, v, c = e
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
            # The set of edges to consider for addition should be the remaining ones
            existing_set = set((min(u,v), max(u,v)) for u,v,_ in edges)
            remaining_edges = [e for e in all_edges if e not in existing_set]
        else:
            remaining_edges = all_edges[:]
            
        # Determine order of remaining edges
        if order_type == 'random':
            rng.shuffle(remaining_edges)
        elif order_type == 'natural':
            pass
        elif order_type == 'reverse':
            remaining_edges.reverse()
            
        for (u, v) in remaining_edges:
            # Check if edge can be added
            # Find all length-3 paths u-a-b-v
            # If any path has non-distinct colors, edge is impossible.
            # Otherwise, forbidden colors are the colors on those paths.
            
            impossible = False
            forbidden_colors = set()
            
            # Iterate a in neighbors[u]
            for a in neighbors[u]:
                if a == v:
                    continue
                ca = adj[u][a]
                # Iterate b in neighbors[v]
                for b in neighbors[v]:
                    if b == u or b == a:
                        continue
                    # Check if (a,b) is an edge
                    if adj[a][b] != -1:
                        cb = adj[a][b]
                        cv = adj[b][v]
                        # Check distinctness of ca, cb, cv
                        if ca == cb or cb == cv or ca == cv:
                            impossible = True
                            break
                        # Add to forbidden
                        forbidden_colors.add(ca)
                        forbidden_colors.add(cb)
                        forbidden_colors.add(cv)
                if impossible:
                    break
            
            if not impossible:
                # Find a valid color
                # Randomize color order to avoid bias and help with diversity in restarts
                colors = list(range(num_colors))
                rng.shuffle(colors)
                
                placed = False
                for c in colors:
                    if c not in forbidden_colors:
                        # Place edge
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
                        placed = True
                        break
        
        return edges

    # Phase 1: Standard Random Restarts
    # Use the fast greedy method to find a good baseline
    order_types = ['random', 'natural', 'reverse']
    num_restarts = 100 # Reduced to allow time for Phase 2
    
    for ot in order_types:
        for i in range(num_restarts):
            if time.time() > deadline:
                break
            edges = greedy_build(i, None, ot)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges

    if best_edges is None:
        best_edges = [[0, v, 0] for v in range(1, 20)]
        best_count = len(best_edges)

    # Phase 2: Local Search / Iterative Refinement
    # Try to improve the best solution by removing a few edges and re-filling
    # This allows escaping local optima where the greedy order prevented certain edges from being added.
    
    current_best = best_edges[:]
    
    # We will run this loop until time runs out
    # Each iteration:
    # 1. Take current_best
    # 2. Remove k random edges (k=1 to 5)
    # 3. Re-run greedy on the remaining graph (adding the removed edges back into the pool)
    # 4. If new graph is better, update current_best
    
    max_refine_iterations = 5000
    refine_iter = 0
    
    while time.time() < deadline and refine_iter < max_refine_iterations:
        refine_iter += 1
        
        # Decide how many edges to remove
        # Smaller k means more local, larger k means more global perturbation
        k = random.randint(1, 5)
        
        # Remove k random edges from current_best
        if len(current_best) <= k:
            continue
            
        edges_to_remove = random.sample(current_best, k)
        remove_set = set((min(u,v), max(u,v)) for u,v,_ in edges_to_remove)
        
        # Filter current_best to keep only edges not in remove_set
        base_edges = [e for e in current_best if (min(e[0], e[1]), max(e[0], e[1])) not in remove_set]
        
        # Re-run greedy build starting from base_edges
        # Use a new seed for the randomization in the greedy phase
        seed = refine_iter * 1000 + random.randint(0, 100)
        
        new_edges = greedy_build(seed, base_edges, 'random')
        
        if len(new_edges) > best_count:
            best_count = len(new_edges)
            best_edges = new_edges
            current_best = new_edges[:]
            
            # If we found a significantly better solution, we might want to restart the refinement
            # from this new point, which we are effectively doing by updating current_best.

    return best_edges
