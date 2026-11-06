import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n = 18
    
    def is_k4_free(adj):
        # Check for K4
        # A K4 exists if there are 4 vertices all connected to each other.
        # We can check by looking at triangles. For each triangle (u,v,w),
        # check if there is a vertex x connected to u, v, and w.
        # Or simply iterate all 4-combinations? 18C4 = 3060. Very fast.
        for combo in itertools.combinations(range(n), 4):
            u, v, w, x = combo
            if (adj[u][v] and adj[u][w] and adj[u][x] and
                adj[v][w] and adj[v][x] and
                adj[w][x]):
                return False
        return True

    def max_triangle_free_size(adj):
        # Find largest subset S such that G[S] is triangle-free.
        # This is equivalent to finding the largest subset that does not contain any triangle.
        # Let T be the set of all triangles in G.
        # We want max |S| such that S does not contain any t in T.
        
        # Precompute triangles
        triangles = []
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]: continue
                for w in range(v+1, n):
                    if adj[u][w] and adj[v][w]:
                        triangles.append((u, v, w))
                        
        if not triangles:
            return n
            
        best_size = 0
        
        # Branch and bound
        # We select vertices one by one.
        # State: current_set (list), available (set of vertices not yet considered)
        # Optimization: If current_set + available size <= best_size, prune.
        
        # To speed up, we can order vertices by degree or something?
        # Let's just use simple recursion.
        
        def dfs(current_set, available):
            nonlocal best_size
            if len(current_set) + len(available) <= best_size:
                return
            
            if not available:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                return
            
            # Heuristic: pick a vertex from available
            # To make it faster, we can pick a vertex involved in many triangles with current_set?
            # Or just pick the first one.
            # Let's pick a random one or the one with highest index to break symmetry?
            # Actually, standard B&B works fine for n=18.
            
            v = next(iter(available))
            new_available = available - {v}
            
            # Option 1: Exclude v
            dfs(current_set, new_available)
            
            # Option 2: Include v
            # Check if adding v creates a triangle with current_set
            valid = True
            for u in current_set:
                if not adj[v][u]:
                    continue
                for w in current_set:
                    if w <= u: # avoid duplicates, w > u
                        continue
                    if adj[u][w] and adj[v][w]:
                        # Triangle (u, v, w)
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                dfs(current_set + [v], new_available)
                
        dfs([], set(range(n)))
        return best_size

    def random_k4_free_graph():
        adj = [[False]*n for _ in range(n)]
        # Start with empty graph, add random edges, remove if K4 created
        # Or start with random graph, remove K4s.
        # Let's start with a random graph with p=0.5
        for u in range(n):
            for v in range(u+1, n):
                if random.random() < 0.5:
                    adj[u][v] = adj[v][u] = True
        
        # Remove edges that create K4s
        # Iterate until no K4s?
        # Better: Iterate all edges, if removing it breaks a K4, remove it?
        # Actually, if we add edges randomly, we might create many K4s.
        # Let's use a repair strategy:
        # While not K4-free:
        #   Find a K4
        #   Remove a random edge from that K4
        
        while not is_k4_free(adj):
            # Find a K4
            found = False
            for combo in itertools.combinations(range(n), 4):
                u, v, w, x = combo
                if (adj[u][v] and adj[u][w] and adj[u][x] and
                    adj[v][w] and adj[v][x] and
                    adj[w][x]):
                    # Remove a random edge from this K4
                    edges = [(u,v), (u,w), (u,x), (v,w), (v,x), (w,x)]
                    e = random.choice(edges)
                    adj[e[0]][e[1]] = adj[e[1]][e[0]] = False
                    found = True
                    break
            if not found:
                break # Should not happen if is_k4_free is correct
                
        return adj

    best_graph = None
    best_score = 18 # Max possible
    
    # Initial random search
    for i in range(50):
        if time.time() > deadline: break
        g = random_k4_free_graph()
        score = max_triangle_free_size(g)
        if score < best_score:
            best_score = score
            best_graph = g
            
    # Local Search
    # We will perform local search starting from the best graph found
    # and also from random graphs if time permits.
    
    def local_search(start_graph, steps=1000):
        nonlocal best_score, best_graph
        current_graph = [row[:] for row in start_graph]
        current_score = max_triangle_free_size(current_graph)
        
        for _ in range(steps):
            if time.time() > deadline: break
            if current_score == 1: # Can't do better than 1? No, min is 1? 
                # Actually, if graph is complete, score is 3 (triangle). 
                # If graph has no triangles, score is 18.
                # We want to minimize score.
                # The minimum possible score for a graph on 18 vertices is at least 3? 
                # No, if the graph is a matching, it's triangle free, so score 18.
                # We want to force triangles everywhere.
                # The minimum score is related to the Turan number.
                # If we can make every 4-set contain a triangle, score is 3.
                # Can we make every 5-set contain a triangle?
                # Let's just search.
                pass
                
            # Pick a random edge to flip
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            
            # Try flipping
            original_state = current_graph[u][v]
            current_graph[u][v] = not original_state
            current_graph[v][u] = current_graph[u][v]
            
            if not is_k4_free(current_graph):
                # Revert
                current_graph[u][v] = original_state
                current_graph[v][u] = original_state
                continue
                
            new_score = max_triangle_free_size(current_graph)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_graph = [row[:] for row in current_graph]
            else:
                # Revert with some probability (Simulated Annealing style)
                # Or just revert to keep it simple (Hill Climbing)
                # Let's use a simple acceptance rule: accept if better, else revert.
                # To escape local optima, we can sometimes accept worse moves.
                if random.random() > 0.1: # 10% chance to accept worse
                    current_graph[u][v] = original_state
                    current_graph[v][u] = original_state
                # else keep the worse move (restart from this point)
                # But we need to track current_score if we keep it.
                # Let's just stick to Hill Climbing for simplicity and speed, 
                # but do multiple restarts.
                current_graph[u][v] = original_state
                current_graph[v][u] = original_state

    # Run local search on the best graph found
    if best_graph:
        local_search(best_graph, steps=5000)
        
    # If time remains, do more random restarts + local search
    while time.time() < deadline - 5:
        g = random_k4_free_graph()
        score = max_triangle_free_size(g)
        if score < best_score:
            best_score = score
            best_graph = g
            local_search(best_graph, steps=1000)
        else:
            # Try local search from this random graph anyway
            local_search(g, steps=500)
            
    if best_graph is None:
        # Fallback to seed
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
        
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_graph[u][v]:
                edges.append([u, v])
    return edges
