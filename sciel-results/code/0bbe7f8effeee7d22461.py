import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin for 120s limit
    
    def verify(colors, n):
        """Verify no monochromatic C4."""
        if len(colors) != n*(n-1)//2:
            return False
        # Build color-specific adjacency
        color_adj = [ [set() for _ in range(n)] for _ in range(5) ]
        edge_idx = 0
        for i in range(n):
            for j in range(i+1, n):
                c = colors[edge_idx]
                edge_idx += 1
                color_adj[c][i].add(j)
                color_adj[c][j].add(i)
        
        for c in range(5):
            for i in range(n):
                for j in range(i+1, n):
                    common = len(color_adj[c][i] & color_adj[c][j])
                    if common >= 2:
                        return False
        return True

    def greedy_packing(n, seed_val):
        """
        Greedy set-cover/packing algorithm.
        Assigns edges one by one to the color that minimizes the number of
        C4s that would be created by adding that edge to that color.
        """
        random.seed(seed_val)
        num_edges = n*(n-1)//2
        
        # Precompute edge endpoints
        endpoints = []
        for i in range(n):
            for j in range(i+1, n):
                endpoints.append((i,j))
        
        # Track neighbors for each vertex in each color
        # nbrs[v][c] = set of neighbors of v in color c
        nbrs = [ [set() for _ in range(5)] for _ in range(n) ]
        
        # Track C4 potential: for each color c and pair (u,v),
        # the number of common neighbors in color c.
        # We need to know, for a candidate edge (u,v) in color c,
        # how many C4s would be formed.
        # A C4 is formed if there exists x such that (u,x) and (v,x) are both in color c.
        # So the "cost" of adding edge (u,v) to color c is the number of common neighbors of u and v in color c.
        # If this count is >= 1, adding the edge creates at least one C4.
        # Actually, it creates C(count, 1) = count C4s? No.
        # A C4 is a cycle of length 4. u-v-x-y-u.
        # If we add edge (u,v) to color c, and there are k common neighbors of u and v in color c,
        # then for each common neighbor x, we have a triangle u-v-x.
        # This doesn't directly form a C4 unless there's another path.
        # Wait, the condition is "no two vertices with two common neighbours in one colour".
        # So if u and v have common neighbors x1, x2 in color c, then {u,v,x1,x2} contains a C4?
        # The C4 would be u-x1-v-x2-u.
        # Yes. So the number of C4s created by adding edge (u,v) to color c
        # is C(k, 2) where k is the number of common neighbors of u and v in color c BEFORE adding the edge.
        
        # To make this efficient, we can track for each pair (u,v) and color c,
        # the number of common neighbors. But that's O(n^2 * 5) memory, which is fine for n=26.
        # However, updating it is O(n) per edge addition.
        
        # Let's use a different metric: the "potential" of a color class is the sum of C(k,2) over all pairs.
        # But we want to assign edges greedily.
        # For each unassigned edge (u,v), for each color c, compute the number of common neighbors of u and v in color c.
        # Choose the color c that minimizes this number.
        # If the minimum is 0, great, no C4 is created.
        # If the minimum is > 0, we are forced to create C4s, but we pick the least bad option.
        # This is a heuristic; it doesn't guarantee a valid coloring, but it might find one if n is small enough.
        
        # To speed up, we can precompute the common neighbor counts.
        # CN[c][u][v] = number of common neighbors of u and v in color c.
        # Initially all 0.
        # When we add edge (u,v) to color c:
        #   For each x in nbrs[u][c] (excluding v), CN[c][x][v] += 1, CN[c][v][x] += 1
        #   For each x in nbrs[v][c] (excluding u), CN[c][x][u] += 1, CN[c][u][x] += 1
        #   And CN[c][u][v] += 1 (u and v are now neighbors, but they don't count as common neighbors of each other for C4 purposes? 
        #   Wait, the condition is about pairs of vertices having two common neighbors.
        #   The pair (u,v) themselves: if they have common neighbors, that's what matters.
        #   Adding edge (u,v) doesn't change the common neighbors of u and v directly, 
        #   but it changes the common neighbors of (u,x) and (v,x) for all x.
        
        # Let's refine the cost function.
        # The total number of C4s in color c is sum_{u<v} C(CN[c][u][v], 2).
        # When we add edge (u,v) to color c, how does the total C4 count change?
        # CN[c][u][v] increases by 1 (since u and v are now adjacent, but wait, CN counts common neighbors, not adjacency).
        # Actually, CN[c][a][b] is the number of vertices x such that (a,x) and (b,x) are in color c.
        # Adding edge (u,v) to color c means (u,v) is in color c.
        # This affects CN[c][u][v]? No, CN[c][u][v] counts common neighbors of u and v.
        # The edge (u,v) itself is not a neighbor.
        # So adding edge (u,v) to color c does NOT change CN[c][u][v].
        # It DOES change CN[c][u][x] for all x such that (v,x) is in color c (x != u).
        #   Because now u and x have a new common neighbor: v.
        # So CN[c][u][x] += 1 for each x in nbrs[v][c] \ {u}.
        # Similarly, CN[c][v][x] += 1 for each x in nbrs[u][c] \ {v}.
        
        # The change in total C4 count for color c is:
        # For each x in nbrs[v][c] \ {u}:
        #   CN[c][u][x] goes from k to k+1.
        #   Change in C(k,2) is C(k+1,2) - C(k,2) = k.
        # For each x in nbrs[u][c] \ {v}:
        #   CN[c][v][x] goes from k to k+1.
        #   Change in C(k,2) is k.
        
        # So the cost of adding edge (u,v) to color c is:
        #   sum_{x in nbrs[v][c] \ {u}} CN[c][u][x] + sum_{x in nbrs[u][c] \ {v}} CN[c][v][x]
        
        # This is O(n) to compute per candidate edge-color pair.
        # With 5 colors and O(n^2) edges, total work is O(n^3 * 5) = O(n^3).
        # For n=26, n^3 = 17576, times 5 is ~88k operations. Very fast.
        
        # Let's implement this.
        
        CN = [ [ [0]*n for _ in range(n) ] for _ in range(5) ]
        # CN[c][u][v] = number of common neighbors of u and v in color c
        
        colors = [0] * num_edges
        edge_idx = 0
        
        # To make it more effective, we can process edges in a specific order.
        # Random order might be good to avoid bias.
        edge_order = list(range(num_edges))
        random.shuffle(edge_order)
        
        for e_idx in edge_order:
            u, v = endpoints[e_idx]
            
            best_c = -1
            best_cost = float('inf')
            
            for c in range(5):
                # Compute cost of adding (u,v) to color c
                cost = 0
                # For x in nbrs[v][c] \ {u}: cost += CN[c][u][x]
                for x in nbrs[v][c]:
                    if x != u:
                        cost += CN[c][u][x]
                # For x in nbrs[u][c] \ {v}: cost += CN[c][v][x]
                for x in nbrs[u][c]:
                    if x != v:
                        cost += CN[c][v][x]
                
                if cost < best_cost:
                    best_cost = cost
                    best_c = c
            
            # Assign edge to best_c
            colors[e_idx] = best_c
            c = best_c
            
            # Update CN
            # For x in nbrs[v][c] \ {u}: CN[c][u][x] += 1, CN[c][x][u] += 1
            for x in nbrs[v][c]:
                if x != u:
                    CN[c][u][x] += 1
                    CN[c][x][u] += 1
            # For x in nbrs[u][c] \ {v}: CN[c][v][x] += 1, CN[c][x][v] += 1
            for x in nbrs[u][c]:
                if x != v:
                    CN[c][v][x] += 1
                    CN[c][x][v] += 1
            
            # Update nbrs
            nbrs[u][c].add(v)
            nbrs[v][c].add(u)
        
        # Verify
        if verify(colors, n):
            return colors
        return None

    def try_greedy(n, num_restarts=10):
        """Try greedy packing with multiple random edge orderings."""
        for r in range(num_restarts):
            if time.time() > deadline - 5:
                break
            res = greedy_packing(n, seed_val=1000 + r)
            if res is not None:
                return res
        return None

    # Try greedy packing for n from 26 down to 12
    for n in range(26, 11, -1):
        if time.time() > deadline - 10:
            break
        
        # Allocate time based on n
        # Greedy is fast, so we can do many restarts
        num_restarts = 20 if n <= 20 else 10
        res = try_greedy(n, num_restarts)
        if res is not None:
            return {'n': n, 'colors': res}
    
    # Fallback: trivial coloring for small n
    n = 11
    num_edges = n*(n-1)//2
    colors = [i % 5 for i in range(num_edges)]
    return {'n': n, 'colors': colors}
