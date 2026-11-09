import time
from random import Random

def construct():
    def instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        # The seed program uses 4*n*d attempts. We must match this exactly.
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def solve_strong_coloring(n, edges):
        # Precompute adjacency list
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        m = len(edges)
        if m == 0:
            return []
            
        # Precompute conflicts for each edge
        # conflict[i] is a set of indices j such that edge j conflicts with edge i
        conflicts = [set() for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                
                # Check if they share a vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                    continue # Already conflicting, no need to check joined-by-edge
                    
                # Check if joined by an edge
                # Vertices of edge i are u1, v1. Vertices of edge j are u2, v2.
                # Check if any vertex in i is adjacent to any vertex in j.
                if (u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]):
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                    
        # Backtracking solver
        # We want to minimize the number of colors.
        # The conjecture says max 5*d^2/4.
        # Let's estimate a good upper bound. 
        # Greedy usually uses <= 2*Delta^2.
        # Let's try increasing the color limit until we find a solution.
        
        # Initial guess for max colors
        # For d=4, bound is 20. d=5, bound is 31. d=6, bound is 45.
        # We can start from a lower bound like max degree + 1 or similar, but 
        # strong edge coloring number is usually higher.
        
        # Let's just run backtracking with a limit.
        # We'll try limits starting from a reasonable lower bound up to the conjectured bound.
        
        # Lower bound: Max degree + 1? No, strong edge coloring is harder.
        # A simple lower bound is the size of the largest star (degree+1) but also 
        # consider triangles etc. 
        # Let's just start from 1 and go up? No, too slow if optimal is 20.
        
        # Better: Use the greedy result as an upper bound, then try to improve?
        # Or just try the conjectured bound directly. If it fails, increase.
        # But the conjecture is an upper bound, so it should work.
        
        # Let's calculate the greedy coloring first to get a safe upper bound.
        def greedy_coloring():
            color = [-1] * m
            for i in range(m):
                used = set()
                for j in conflicts[i]:
                    if color[j] != -1:
                        used.add(color[j])
                c = 0
                while c in used:
                    c += 1
                color[i] = c
            return max(color) + 1
            
        upper_bound = greedy_coloring()
        
        # The conjectured bound is 5*d^2/4.
        # We can try to see if a solution exists with fewer colors.
        # But finding the *minimum* might be hard.
        # However, for these small graphs, backtracking is fast.
        
        # Let's try to find the minimum k such that a coloring with k colors exists.
        # We know k <= upper_bound.
        
        # Heuristic for variable ordering:
        # Pick the uncolored edge with the fewest available colors.
        
        def backtrack(color, k_limit, deadline):
            # color is a list of length m, -1 if uncolored
            # Find uncolored edge
            uncolored = [i for i in range(m) if color[i] == -1]
            if not uncolored:
                return True
                
            # Pick variable with minimum remaining values
            best_i = -1
            best_available = None
            min_avails = k_limit + 1
            
            # Sort uncolored edges by some heuristic? 
            # To speed up, we can just scan. m is small (< 240).
            for i in uncolored:
                used_colors = set()
                for j in conflicts[i]:
                    if color[j] != -1:
                        used_colors.add(color[j])
                        
                avail = 0
                # We don't need to construct the set of available, just count or check min
                # Actually, we need to know which colors are available to try them.
                # But for selection, we just need the count.
                for c in range(k_limit):
                    if c not in used_colors:
                        avail += 1
                        if avail > min_avails:
                            break # No need to count more if already worse
                            
                if avail < min_avails:
                    min_avails = avail
                    best_i = i
                    best_available = used_colors # Store the used set to reuse?
                    # Note: used_colors depends on current coloring state, which is fixed for this iteration
                    if avail == 0:
                        break # Fail fast
                
            if best_i == -1:
                return False # Should not happen if uncolored not empty
                
            if min_avails == 0:
                return False
                
            # Try colors for best_i
            # Order of colors: try 0, 1, ...
            # We can also sort colors by how they constrain others, but simple order is fine.
            
            used_in_best = set()
            for j in conflicts[best_i]:
                if color[j] != -1:
                    used_in_best.add(color[j])
                    
            for c in range(k_limit):
                if c not in used_in_best:
                    color[best_i] = c
                    if backtrack(color, k_limit, deadline):
                        return True
                    color[best_i] = -1
                    if time.time() > deadline:
                        return False # Timeout
            return False

        # We want the minimum number of colors.
        # Let's try k from a lower bound to upper_bound.
        # Lower bound: max degree + 1 is not sufficient. 
        # Let's just start from 1? No.
        # Let's try the conjectured bound first? 
        # If the conjecture holds, the optimal is <= bound.
        # It is possible the optimal is much lower.
        
        # Let's try to find the exact minimum.
        # Since m is small, we can binary search or linear search.
        # Linear search from lower_bound to upper_bound.
        
        # A good lower bound for strong edge coloring is the maximum degree + 1? 
        # Actually, for a star K_{1,d}, strong edge coloring needs d+1 colors.
        # So max_degree + 1 is a lower bound.
        
        max_deg = max(len(adj[u]) for u in range(n))
        lower_bound = max_deg + 1
        
        # Also, consider triangles. If there is a triangle, edges need distinct colors?
        # In a triangle, edges (u,v), (v,w), (w,u). 
        # (u,v) and (v,w) share v -> diff colors.
        # (v,w) and (w,u) share w -> diff colors.
        # (w,u) and (u,v) share u -> diff colors.
        # So 3 colors needed for triangle edges.
        
        # Let's just iterate k from lower_bound to upper_bound.
        
        start_time = time.time()
        total_deadline = start_time + 10.0 # 10 seconds per graph
        
        for k in range(lower_bound, upper_bound + 1):
            color = [-1] * m
            # Simple greedy initialization might help? No, backtracking from scratch is fine.
            
            if backtrack(color, k, total_deadline):
                return color
                
        # If we didn't find a solution (should not happen if upper_bound is valid),
        # return the greedy coloring
        return greedy_coloring_result # Need to store this? 
        # Wait, the function returns a list of colors.
        # Let's just return the result of the successful backtracking.
        
        # If loop finishes without return, something went wrong.
        # Fall back to greedy
        return [greedy_coloring() for _ in range(m)] # This is wrong, greedy_coloring returns int.

    # Refactor solve_strong_coloring to be cleaner

    def solve_strong_coloring_v2(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        m = len(edges)
        if m == 0:
            return []
            
        conflicts = [set() for _ in range(m)]
        
        for i in range(m):
            u1, v1 = edges[i]
            for j in range(i + 1, m):
                u2, v2 = edges[j]
                
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                    continue
                    
                if (u2 in adj[u1] or v2 in adj[u1] or u2 in adj[v1] or v2 in adj[v1]):
                    conflicts[i].add(j)
                    conflicts[j].add(i)
                    
        def get_greedy_colors():
            color = [-1] * m
            for i in range(m):
                used = set()
                for j in conflicts[i]:
                    if color[j] != -1:
                        used.add(color[j])
                c = 0
                while c in used:
                    c += 1
                color[i] = c
            return color
            
        greedy_sol = get_greedy_colors()
        upper_bound = max(greedy_sol) + 1
        
        max_deg = max(len(adj[u]) for u in range(n))
        lower_bound = max_deg + 1
        
        def backtrack(color, k_limit, deadline):
            uncolored = [i for i in range(m) if color[i] == -1]
            if not uncolored:
                return True
                
            best_i = -1
            min_avails = k_limit + 1
            
            for i in uncolored:
                used_colors = set()
                for j in conflicts[i]:
                    if color[j] != -1:
                        used_colors.add(color[j])
                        
                avail = 0
                for c in range(k_limit):
                    if c not in used_colors:
                        avail += 1
                        if avail > min_avails:
                            break
                            
                if avail < min_avails:
                    min_avails = avail
                    best_i = i
                    if avail == 0:
                        break
                
            if best_i == -1:
                return False
                
            if min_avails == 0:
                return False
                
            used_in_best = set()
            for j in conflicts[best_i]:
                if color[j] != -1:
                    used_in_best.add(color[j])
                    
            for c in range(k_limit):
                if c not in used_in_best:
                    color[best_i] = c
                    if backtrack(color, k_limit, deadline):
                        return True
                    color[best_i] = -1
                    if time.time() > deadline:
                        return False
            return False

        start_time = time.time()
        total_deadline = start_time + 10.0
        
        # Try to find minimal k
        for k in range(lower_bound, upper_bound + 1):
            color = [-1] * m
            if backtrack(color, k, total_deadline):
                return color
                
        # Fallback to greedy
        return greedy_sol

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        colors = solve_strong_coloring_v2(len(spec[0]), edges)
        out.append(colors)
    return out
