import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # We will try to find a valid coloring for n=18, then 17, then 16...
    # But 17 is known to exist. 18 is the target.
    
    def check_c6(n, colors, u, v, c):
        # Check if adding edge (u,v) with color c creates a C6 in color c subgraph
        # We need to check if there is a path of length 4 from u to v in G_c
        # Path u-a-b-d-e-v (5 edges in path? No, path length 4 means 4 edges: u-a, a-b, b-e, e-v? No.)
        # Cycle: u-a-b-d-e-v-u.
        # Edges: (u,a), (a,b), (b,d), (d,e), (e,v), (v,u).
        # Path from u to v in G_c \ {(u,v)} of length 5?
        # No, the cycle has 6 edges. The path from u to v not using (u,v) has 5 edges.
        # u -> a -> b -> d -> e -> v.
        # So we need a path of length 5 from u to v.
        
        # Wait, my previous analysis said path length 4.
        # Let's re-verify.
        # C6 vertices: 0,1,2,3,4,5.
        # Edges: 0-1, 1-2, 2-3, 3-4, 4-5, 5-0.
        # If we add edge 5-0 (u,v), the rest is path 0-1-2-3-4-5.
        # Length of path 0 to 5 is 5 edges.
        # So we need to check if there is a path of length 5 from u to v in G_c.
        
        # This is more expensive.
        # Path length 5: u-a-b-c-d-v.
        # a in N(u), v in N(v).
        # Path a-b-c-d of length 3.
        # So we need path of length 3 between a and d.
        # a-b-c-d.
        # b in N(a), c in N(b) cap N(d).
        
        # So:
        # For a in N(u):
        #   For d in N(v):
        #     Check if dist(a,d) == 3 in G_c.
        
        # Dist 3 check:
        # Exists b in N(a) such that N(b) intersects N(d).
        # And the path must be simple.
        
        neighbors_u = [x for x in range(n) if x != u and x != v and colors[x*n + u] == c or colors[u*n + x] == c]
        # Note: colors is stored for i<j. 
        # Let's use a matrix or a set of edges for current color c for speed.
        # Since we are backtracking, maintaining a list of edges per color is good.
        
        pass

    # This logic is getting complex to implement correctly in a short time with a single file.
    # Let's use a simpler, known valid coloring for n=17 if search fails or is slow.
    
    # Known construction for n=17:
    # Vertices 0..16.
    # This is a known result.
    # I will hardcode a valid coloring for n=17 if I can find/generate one, 
    # or use a fast random restart local search.
    
    # Given the "Fresh Design" directive, I should implement the search properly.
    
    import random
    
    def solve(n):
        # colors[i*n + j] for i < j
        # Use a list of size n*n, initialize to -1
        colors = [-1] * (n*n)
        
        # Symmetry breaking
        # 0-1 is color 0
        colors[0*n + 1] = 0
        # 0-2 is color 0 or 1
        # 0-3 is color 0, 1, 2
        # ...
        
        # We need a fast validity check.
        # We will maintain adj lists for each color.
        adj = [[set() for _ in range(n)] for _ in range(4)]
        
        def is_valid(u, v, c):
            # Check if adding (u,v) to color c creates C6
            # We need path of length 5 from u to v in adj[c]
            # u-a-b-d-e-v
            # a in adj[c][u]
            # v in adj[c][v]
            # Path a-b-d-e of length 3.
            
            # Optimize: 
            # For each a in adj[c][u]:
            #   For each d in adj[c][v]:
            #     Check if there is a path of length 3 from a to d.
            #     Path a-b-e-d.
            #     b in adj[c][a]
            #     e in adj[c][b] intersect adj[c][d]
            
            # To make this faster, we can precompute "distance 3" reachability?
            # Or just iterate.
            # n=17, deg ~ 8.
            # 8 * 8 * (8*8) = 4096 ops per edge check.
            # 136 edges. 136 * 4096 = 550k ops per node in search tree.
            # Search tree might be large.
            
            for a in adj[c][u]:
                if a == v: continue
                for d in adj[c][v]:
                    if d == u: continue
                    if a == d: continue # Path a-b-a-d? No, a!=d
                    # Check path a-b-e-d
                    # b in adj[c][a]
                    for b in adj[c][a]:
                        if b == u or b == v or b == d: continue
                        # e in adj[c][b] and e in adj[c][d]
                        # Also e != a, e != u, e != v
                        common = adj[c][b].intersection(adj[c][d])
                        # Filter e
                        for e in common:
                            if e == a or e == u or e == v or e == b or e == d:
                                continue
                            return False
            return True

        def backtrack(idx):
            if time.time() > deadline:
                return None
            
            if idx == n*n:
                return colors[:]
            
            u = idx // n
            v = idx % n
            
            if u >= v or colors[idx] != -1:
                return backtrack(idx + 1)
            
            # Symmetry breaking for vertex 0
            # If u == 0:
            #   if v == 1: only try 0
            #   if v == 2: try 0, 1
            #   if v == 3: try 0, 1, 2
            #   ...
            
            # Simple symmetry: 
            # The first edge (0,1) is 0.
            # For u=0, the colors of (0,1), (0,2)... should be non-decreasing?
            # No, that restricts too much.
            # Just fix (0,1)=0.
            # And for u=0, v>1, we can restrict the set of colors tried
            # based on previous colors of (0, k) for k<v.
            # E.g., if (0,1)=0, (0,2) can be 0 or 1.
            # (0,3) can be 0, 1, 2.
            
            max_color = 0
            if u == 0:
                # Find max color used so far for (0, k) with k < v
                for k in range(1, v):
                    if colors[k*n] != -1: # (0,k) is at index k*n
                        if colors[k*n] > max_color:
                            max_color = colors[k*n]
                # We can try colors up to max_color + 1
                # But we also have 4 colors total.
                
            for c in range(4):
                if u == 0:
                    # Enforce symmetry breaking
                    if v == 1:
                        if c != 0: continue
                    elif v == 2:
                        if c > 1: continue
                    elif v == 3:
                        if c > 2: continue
                    else:
                        # c can be anything, but let's say we don't introduce a new color 
                        # unless necessary? No, we have 4 colors.
                        # Let's just allow all, but the previous constraints handle the first few.
                        pass

                if not is_valid(u, v, c):
                    continue
                
                colors[idx] = c
                adj[c][u].add(v)
                adj[c][v].add(u)
                
                res = backtrack(idx + 1)
                if res is not None:
                    return res
                
                colors[idx] = -1
                adj[c][u].remove(v)
                adj[c][v].remove(u)
                
            return None
        
        return backtrack(0)

    best_n = 0
    best_colors = []
    
    # Try 18, then 17
    for n in [18, 17]:
        if time.time() > deadline - 5:
            break
        res = solve(n)
        if res is not None:
            # Verify
            # ...
            best_n = n
            best_colors = res
            break
            
    if best_n == 0:
        # Fallback to seed
        n = 9
        return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
        
    # Format output
    out_colors = []
    for i in range(best_n):
        for j in range(i+1, best_n):
            out_colors.append(best_colors[i*best_n + j])
            
    return {'n': best_n, 'colors': out_colors}
