import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 148.0  # Safety margin before 150s limit
    
    def time_check():
        return time.time() > deadline

    # Target n=17
    # If we can't find n=17, we will try n=16, then n=15, etc.
    # But n=17 is the known best. We will focus on finding a valid n=17.
    
    # We will use a greedy initialization + local search (Simulated Annealing)
    # for n=17.
    
    n = 17
    # Number of edges
    m = n * (n - 1) // 2
    
    # Edge indices: map (i, j) to index 0..m-1
    # We need to efficiently map between edge index and (i, j)
    # For n=17, m=136.
    
    # Precompute edge endpoints
    edges = []
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
            edge_index[(i, j)] = idx
            idx += 1
            
    # Initialize colors: random or greedy?
    # Greedy: For each edge, pick a color that minimizes the number of C6s formed.
    # But checking C6s for every edge assignment is expensive if done naively.
    # However, for n=17, we can maintain a count of C6s for each color.
    # Actually, a simpler initialization is random, then repair.
    # Let's try a greedy "girth" heuristic:
    # Assign colors such that each color class has high girth.
    # A graph with girth >= 7 has no C3, C4, C5, C6.
    # We want to avoid C6 specifically.
    
    # Let's use a standard local search:
    # 1. Random initialization
    # 2. Calculate initial violations (number of monochromatic C6s)
    # 3. Simulated Annealing:
    #    - Pick a random edge
    #    - Change its color
    #    - Calculate delta in violations
    #    - Accept if delta < 0 or with probability exp(-delta/T)
    
    # To calculate delta efficiently, we need to know how many C6s are affected by changing an edge.
    # A C6 is a cycle of length 6.
    # Changing edge (u, v) affects C6s that contain (u, v).
    # If we remove (u, v) from color c, we break all C6s of color c containing (u, v).
    # If we add (u, v) to color c', we form new C6s of color c' containing (u, v).
    
    # Precompute for each edge (u, v), the list of "potential" C6s?
    # A C6 is defined by 6 vertices.
    # This is complex. Let's use a simpler violation count:
    # Count all monochromatic C6s in the graph.
    # For n=17, the number of C6s is manageable?
    # Number of ways to pick 6 vertices from 17: C(17, 6) = 12376.
    # For each set of 6 vertices, there are (6-1)!/2 = 60 cycles.
    # Total C6s in K_17: 12376 * 60 = 742,560.
    # This is small enough to check all of them every time?
    # No, doing this for every step in SA (millions of steps) is too slow.
    
    # We need incremental updates.
    # Let's maintain a count of C6s for each color.
    # When an edge (u, v) changes color from c_old to c_new:
    # 1. Decrease count for c_old by the number of C6s of color c_old that use (u, v).
    # 2. Increase count for c_new by the number of C6s of color c_new that use (u, v).
    
    # To count C6s of color c that use edge (u, v):
    # We need to find paths of length 4 from u to v using only edges of color c, 
    # such that the vertices are distinct and the path doesn't use (u, v) internally?
    # Actually, a C6 containing (u, v) corresponds to a path of length 5 from u to v 
    # in the graph G_c - {(u, v)}? No.
    # A C6 is a cycle u-v-...-u.
    # If (u, v) is in the cycle, the rest of the cycle is a path of length 4 from v to u 
    # in G_c - {(u, v)}?
    # Let's say the cycle is u, v, x, y, z, w, u.
    # Edges: (u,v), (v,x), (x,y), (y,z), (z,w), (w,u).
    # All must be color c.
    # So we need to count the number of paths of length 4 from v to u in G_c, 
    # using vertices {v, x, y, z, w, u} distinct, and not using edge (u, v).
    
    # This can be done by matrix multiplication or just iterating.
    # Given n=17, we can just iterate over possible intermediate vertices.
    # Path v -> x -> y -> z -> w -> u.
    # x, y, z, w must be distinct and not u, v.
    
    # Let's implement a function count_c6s_using_edge(u, v, color)
    # that counts how many C6s of the given color use edge (u, v).
    
    # We will maintain adjacency matrices for each color.
    # adj[c][i][j] = 1 if edge (i, j) has color c.
    
    # Let's try this approach.
    
    def initialize_adjacency(colors):
        adj = [[[0]*n for _ in range(n)] for _ in range(4)]
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            adj[c][u][v] = 1
            adj[c][v][u] = 1
        return adj

    def count_c6s_using_edge(u, v, adj, c):
        """
        Count C6s of color c that contain edge (u, v).
        Cycle: u - v - x - y - z - w - u
        We need to count paths v -> x -> y -> z -> w -> u in G_c
        such that x, y, z, w are distinct and not u, v.
        """
        count = 0
        # x is a neighbor of v (color c), x != u
        # y is a neighbor of x (color c), y != u, y != v
        # z is a neighbor of y (color c), z != u, z != v, z != x
        # w is a neighbor of z (color c), w != u, w != v, w != x, w != y
        # and w is a neighbor of u (color c)
        
        # To speed up, we can precompute neighbor lists for each color.
        # But neighbor lists change.
        # For n=17, iterating 17^4 is ~83k operations.
        # This is fast enough if called a few times per step.
        # But in SA, we might call it many times.
        # Let's optimize:
        # Iterate x in neighbors(v, c) \ {u}
        # Iterate y in neighbors(x, c) \ {u, v}
        # Iterate z in neighbors(y, c) \ {u, v, x}
        # Iterate w in neighbors(z, c) \ {u, v, x, y}
        # Check if w is neighbor of u (color c)
        
        # We need fast neighbor access.
        # adj[c] is a matrix.
        
        for x in range(n):
            if x == u or x == v: continue
            if adj[c][v][x] == 0: continue
            for y in range(n):
                if y == u or y == v or y == x: continue
                if adj[c][x][y] == 0: continue
                for z in range(n):
                    if z == u or z == v or z == x or z == y: continue
                    if adj[c][y][z] == 0: continue
                    for w in range(n):
                        if w == u or w == v or w == x or w == y or w == z: continue
                        if adj[c][z][w] == 0: continue
                        if adj[c][w][u] == 1:
                            count += 1
        return count

    # This is O(n^4) per call. 17^4 = 83521.
    # If we do 1000 steps, that's 80 million ops. Might be slow in Python.
    # Let's see if we can optimize.
    # Or just use a simpler heuristic:
    # Instead of exact C6 count, use a proxy: number of paths of length 4 between u and v?
    # Or just count "potential" C6s?
    
    # Alternative:
    # Use a different local search:
    # Minimize the sum of squares of degrees? No.
    # Minimize the number of C4s? C4s are easier to count?
    # Actually, let's just try the exact count but with a time limit.
    # If it's too slow, we fall back to n=16 or lower.
    
    # Let's try a different initialization:
    # Greedy:
    # Start with empty graph.
    # For each edge (u, v):
    #   Try each color c.
    #   Calculate cost = number of C6s formed.
    #   Pick color with min cost.
    # This is O(m * 4 * n^4) = 136 * 4 * 83k = 45 million.
    # This is doable in < 1s in C, but maybe 5-10s in Python.
    # Let's try it.
    
    # Actually, let's just use Random Initialization + Simulated Annealing.
    # It's more robust.
    
    def get_violations(adj):
        total = 0
        for c in range(4):
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[c][u][v]:
                        total += count_c6s_using_edge(u, v, adj, c)
        # Each C6 is counted 6 times (once for each edge in the cycle)
        # So total / 6 is the number of C6s.
        return total // 6

    # This is O(m * n^4) = 136 * 83k = 11 million.
    # Too slow for every step.
    
    # We need incremental updates.
    # Let's maintain `violation_count` for each color.
    # `violation_count[c]` = number of C6s of color c.
    
    # When edge (u, v) changes from c_old to c_new:
    # 1. Calculate k_old = count_c6s_using_edge(u, v, adj, c_old)
    # 2. Remove edge (u, v) from adj[c_old]
    # 3. violation_count[c_old] -= k_old
    #    BUT: removing the edge might break other C6s?
    #    No, k_old is the number of C6s that *use* this edge.
    #    So those C6s are broken.
    # 4. Add edge (u, v) to adj[c_new]
    # 5. Calculate k_new = count_c6s_using_edge(u, v, adj, c_new)
    #    (Note: adj[c_new] now includes (u, v), but count_c6s_using_edge
    #     looks for paths v->...->u, so it doesn't use (u, v) internally.
    #     So we can calculate it before or after adding, as long as we
    #     don't count (u, v) as part of the path.
    #     My implementation of count_c6s_using_edge does NOT use (u, v)
    #     in the path v->x->y->z->w->u. It only checks if (w, u) is an edge.
    #     So it is safe to calculate after adding the edge, as long as
    #     we don't allow x, y, z, w to be u or v.
    #     Wait, if (u, v) is in adj, does it affect the path?
    #     The path is v-x-y-z-w-u.
    #     The edge (u, v) is not in the path.
    #     So it's fine.
    # 6. violation_count[c_new] += k_new
    
    # This allows O(1) update of total violations, with O(n^4) cost per move.
    # Since we only do one move at a time, this is acceptable.
    
    # Let's implement this.
    
    best_n_found = 0
    best_colors_found = None
    
    # Try n=17
    for target_n in [17, 16, 15]:
        if time_check():
            break
            
        n = target_n
        m = n * (n - 1) // 2
        edges = []
        edge_index = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                edge_index[(i, j)] = idx
                idx += 1
        
        # Initialize
        colors = [random.randint(0, 3) for _ in range(m)]
        adj = initialize_adjacency(colors)
        
        # Calculate initial violations
        violation_counts = [0] * 4
        for c in range(4):
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[c][u][v]:
                        violation_counts[c] += count_c6s_using_edge(u, v, adj, c)
        # Divide by 6
        for c in range(4):
            violation_counts[c] //= 6
            
        total_violations = sum(violation_counts)
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.999
        steps = 0
        
        while not time_check() and T > 0.01:
            # Pick random edge
            e_idx = random.randint(0, m - 1)
            u, v = edges[e_idx]
            c_old = colors[e_idx]
            
            # Pick new color
            c_new = random.randint(0, 3)
            if c_new == c_old:
                continue
                
            # Calculate delta
            # Remove from c_old
            k_old = count_c6s_using_edge(u, v, adj, c_old)
            # Note: k_old is the number of C6s using (u, v) in color c_old.
            # These C6s will be broken.
            
            # Add to c_new
            # We need to count C6s formed in c_new.
            # The edge (u, v) is not yet in adj[c_new].
            # So we can count paths in current adj[c_new].
            k_new = count_c6s_using_edge(u, v, adj, c_new)
            
            delta = k_new - k_old
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                # Accept
                colors[e_idx] = c_new
                adj[c_old][u][v] = 0
                adj[c_old][v][u] = 0
                adj[c_new][u][v] = 1
                adj[c_new][v][u] = 1
                
                violation_counts[c_old] -= k_old
                violation_counts[c_new] += k_new
                
                total_violations = sum(violation_counts)
                
                if total_violations == 0:
                    # Found a valid coloring!
                    best_n_found = n
                    best_colors_found = colors[:]
                    break
                    
            T *= cooling
            steps += 1
            
        if best_n_found == n:
            break
            
    if best_n_found > 0:
        return {'n': best_n_found, 'colors': best_colors_found}
        
    # Fallback
    # Return a trivial coloring for n=5 (no C6 possible)
    n = 5
    colors = [0] * (n * (n - 1) // 2)
    return {'n': n, 'colors': colors}

# We need to make sure count_c6s_using_edge is efficient.
# The current implementation is O(n^4).
# For n=17, it's ~80k ops.
# If we do 10000 steps, that's 800M ops. Might be too slow in Python.
# Let's optimize count_c6s_using_edge.

def count_c6s_using_edge_optimized(u, v, adj, c):
    count = 0
    # x is neighbor of v
    # y is neighbor of x
    # z is neighbor of y
    # w is neighbor of z and u
    
    # Precompute neighbors?
    # adj[c] is a list of lists.
    
    # Let's just use the simple loop but with local variables for speed.
    adj_c = adj[c]
    
    for x in range(n):
        if x == u or x == v: continue
        if adj_c[v][x] == 0: continue
        for y in range(n):
            if y == u or y == v or y == x: continue
            if adj_c[x][y] == 0: continue
            for z in range(n):
                if z == u or z == v or z == x or z == y: continue
                if adj_c[y][z] == 0: continue
                for w in range(n):
                    if w == u or w == v or w == x or w == y or w == z: continue
                    if adj_c[z][w] == 0: continue
                    if adj_c[w][u] == 1:
                        count += 1
    return count

# We need to replace the inner function with this optimized one.
# But `n` is not global in the function.
# We can pass `n` or use a closure.

def construct():
    start_time = time.time()
    deadline = 148.0
    
    def time_check():
        return time.time() > deadline

    # We will use a global-like variable for n inside the helper
    # Or just define the helper inside construct.
    
    best_n_found = 0
    best_colors_found = None
    
    for target_n in [17, 16, 15]:
        if time_check():
            break
            
        n = target_n
        m = n * (n - 1) // 2
        edges = []
        edge_index = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                edge_index[(i, j)] = idx
                idx += 1
        
        def initialize_adjacency(colors):
            adj = [[[0]*n for _ in range(n)] for _ in range(4)]
            for idx, (u, v) in enumerate(edges):
                c = colors[idx]
                adj[c][u][v] = 1
                adj[c][v][u] = 1
            return adj

        def count_c6s_using_edge(u, v, adj, c):
            count = 0
            adj_c = adj[c]
            for x in range(n):
                if x == u or x == v: continue
                if adj_c[v][x] == 0: continue
                for y in range(n):
                    if y == u or y == v or y == x: continue
                    if adj_c[x][y] == 0: continue
                    for z in range(n):
                        if z == u or z == v or z == x or z == y: continue
                        if adj_c[y][z] == 0: continue
                        for w in range(n):
                            if w == u or w == v or w == x or w == y or w == z: continue
                            if adj_c[z][w] == 0: continue
                            if adj_c[w][u] == 1:
                                count += 1
            return count

        # Initialize
        colors = [random.randint(0, 3) for _ in range(m)]
        adj = initialize_adjacency(colors)
        
        # Calculate initial violations
        violation_counts = [0] * 4
        for c in range(4):
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[c][u][v]:
                        violation_counts[c] += count_c6s_using_edge(u, v, adj, c)
        for c in range(4):
            violation_counts[c] //= 6
            
        total_violations = sum(violation_counts)
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.999
        steps = 0
        
        while not time_check() and T > 0.01:
            e_idx = random.randint(0, m - 1)
            u, v = edges[e_idx]
            c_old = colors[e_idx]
            
            c_new = random.randint(0, 3)
            if c_new == c_old:
                continue
                
            k_old = count_c6s_using_edge(u, v, adj, c_old)
            k_new = count_c6s_using_edge(u, v, adj, c_new)
            
            delta = k_new - k_old
            
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                colors[e_idx] = c_new
                adj[c_old][u][v] = 0
                adj[c_old][v][u] = 0
                adj[c_new][u][v] = 1
                adj[c_new][v][u] = 1
                
                violation_counts[c_old] -= k_old
                violation_counts[c_new] += k_new
                
                total_violations = sum(violation_counts)
                
                if total_violations == 0:
                    best_n_found = n
                    best_colors_found = colors[:]
                    break
                    
            T *= cooling
            steps += 1
            
        if best_n_found == n:
            break
            
    if best_n_found > 0:
        return {'n': best_n_found, 'colors': best_colors_found}
        
    n = 5
    colors = [0] * (n * (n - 1) // 2)
    return {'n': n, 'colors': colors}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
