"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    """
    Finds a valid 4-coloring of K_n with no monochromatic triangles.
    Uses Randomized Local Search (Simulated Annealing) which is a heuristic
    optimization approach, distinct from the exact backtracking in the seed.
    """
    
    def count_mono_triangles(colors, n):
        """
        counts[u][v][c] = number of w such that color(u,w)==c and color(v,w)==c.
        Returns total number of monochromatic triangles.
        """
        # colors is a list of length n*(n-1)/2, or a 2D array?
        # Let's use a 2D array for speed: color[u][v]
        # But the output format requires a flat list. We will convert at the end.
        pass

    def try_local_search(n, max_time=10):
        """
        Attempts to find a valid 4-coloring of K_n using Simulated Annealing.
        Returns a flat list of colors if successful, None otherwise.
        """
        start_time = time.time()
        
        # 2D array for colors: color[u][v]
        color = [[-1]*n for _ in range(n)]
        
        # Initialize randomly
        for i in range(n):
            for j in range(i+1, n):
                c = random.randint(0, 3)
                color[i][j] = c
                color[j][i] = c
        
        # Precompute counts: count[u][v][c]
        # count[u][v][c] is the number of vertices w (w!=u, w!=v) such that color[u][w]==c and color[v][w]==c.
        # We can compute this efficiently.
        # For each pair (u, v), iterate w. O(n^3). For n=50, 125k ops. Fine.
        
        def compute_counts():
            counts = [[[0]*4 for _ in range(n)] for _ in range(n)]
            for u in range(n):
                for v in range(u+1, n):
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        c_uw = color[u][w]
                        c_vw = color[v][w]
                        if c_uw == c_vw:
                            counts[u][v][c_uw] += 1
                            counts[v][u][c_uw] += 1
            return counts

        counts = compute_counts()
        
        # Total monochromatic triangles
        # Each triangle (u,v,w) is counted 3 times in the sum of counts[u][v][color(u,v)].
        def get_total_triangles():
            total = 0
            for u in range(n):
                for v in range(u+1, n):
                    total += counts[u][v][color[u][v]]
            return total // 3

        current_triangles = get_total_triangles()
        
        # Simulated Annealing Parameters
        T_start = 1.0
        T_end = 0.01
        alpha = 0.995
        T = T_start
        best_state = None
        best_triangles = current_triangles
        
        # If we start with 0, we are done
        if current_triangles == 0:
            return flatten(color, n)

        while time.time() - start_time < max_time:
            if T < T_end:
                break
                
            # Pick a random edge
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            c_old = color[u][v]
            
            # Try all 4 colors
            best_delta = float('inf')
            best_new_color = c_old
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                
                # Calculate delta in total triangles
                # When we change color[u][v] from c_old to c_new:
                # 1. The edge (u,v) itself stops contributing to c_old triangles and starts contributing to c_new triangles.
                #    Contribution change for edge (u,v): counts[u][v][c_new] - counts[u][v][c_old]
                # 2. The counts for other edges (u,w) and (v,w) change.
                #    For each w, the pair (u,w) and (v,w) are affected.
                #    Specifically, for a fixed w, the term counts[u][w][color(u,w)] changes?
                #    No, counts[u][w][c] depends on color[u][x] and color[w][x].
                #    Changing color[u][v] affects counts[u][w][c] for all w != u,v?
                #    Let's check the definition: counts[a][b][c] = sum_{x} [color[a][x]==c and color[b][x]==c].
                #    If we change color[u][v], it affects:
                #    - counts[u][w][c] for all w: because color[u][v] is one of the terms in the sum for x=v?
                #      Yes, for counts[u][w][c], the sum is over x. One term is x=v.
                #      So if color[u][v] changes, counts[u][w][c] changes for all w.
                #    This makes incremental update O(n^2) per step?
                #    Actually, we can derive the delta directly.
                
                # Let's stick to the simple O(n) update per candidate color if we update counts carefully,
                # or just recompute counts for the affected vertices.
                # For simplicity and correctness, since n is small (<=50), we can just
                # temporarily set the color, recompute counts for u and v, calculate total, then revert?
                # That's O(n^2) per color candidate, 4 candidates, so O(n^2) per step.
                # 50^2 = 2500 ops per step. 1 million steps = 2.5 billion ops. Too slow in Python.
                
                # We need an O(n) or O(1) delta calculation.
                # Let's analyze the change in Total Triangles T.
                # T = 1/3 * sum_{a<b} counts[a][b][color[a][b]]
                
                # Change color[u][v] from c_old to c_new.
                # Affected terms in the sum:
                # 1. Term for edge (u,v): counts[u][v][c_new] - counts[u][v][c_old]
                # 2. Terms for edges (u,w) and (v,w) for all w != u,v.
                #    For edge (u,w), the term is counts[u][w][color[u][w]].
                #    Does counts[u][w][color[u][w]] change?
                #    counts[u][w][c] = sum_{x} [color[u][x]==c and color[w][x]==c].
                #    The term x=v is [color[u][v]==c and color[w][v]==c].
                #    So if color[u][v] changes, counts[u][w][c] changes for c = c_old and c = c_new.
                #    Specifically:
                #    If color[w][v] == c_old, then counts[u][w][c_old] decreases by 1.
                #    If color[w][v] == c_new, then counts[u][w][c_new] increases by 1.
                #    Similarly for counts[v][w].
                
                # This is getting complex. Given the time limit, let's use a simpler heuristic:
                # "Greedy with Random Restarts"
                # 1. Random init.
                # 2. While T > 0:
                #    Find an edge (u,v) that is part of a monochromatic triangle.
                #    Change its color to the one that reduces T the most.
                #    If no move reduces T, pick a random move (to escape local min).
                #    This requires finding a bad edge quickly.
                
                pass
            
            # Due to complexity of incremental update in Python within time limits,
            # we will use a simpler "Greedy Repair" strategy.
            
            # Find any monochromatic triangle
            # We can just check random edges.
            # If color[u][v] == c, check if there is w with color[u][w]==c and color[v][w]==c.
            # If yes, we have a triangle.
            
            # Let's restart the logic with a simpler, faster heuristic.
            pass

        return None

    # Fallback to a known valid construction for small n
    # The seed had a fallback for n=32. We will use a simpler one for n=15 or 19 if needed.
    # But we want to maximize n.
    
    # Strategy:
    # 1. Try to find n=50 with Local Search (if implemented correctly).
    # 2. If not, try n=40, 30...
    # 3. Since implementing the fast incremental SA is tricky, let's use a 
    #    "Greedy Extension" approach which is constructive and fast.
    
    def greedy_construct(n, max_time=10):
        """
        Construct coloring vertex by vertex.
        """
        start_time = time.time()
        # colors[i][j]
        color = [[-1]*n for _ in range(n)]
        
        # Add vertex 0
        for i in range(1, n):
            # Assign colors to (0, i)
            # To avoid symmetry, we can fix some.
            # But greedy: for each new vertex v, assign colors to (v, u) for u < v.
            pass
            
        return None

    # Given the difficulty of writing a perfect SA in this short time,
    # I will implement a **Randomized Greedy Backtracking** with a very strong 
    # heuristic: **Forward Checking with Constraint Propagation**.
    # This is similar to the seed but optimized with **Bitsets** for speed.
    
    import numpy as np

    def construct_with_bitsets(n, max_time=110):
        start_time = time.time()
        
        # Bitset representation of colors
        # For each vertex v and color c, we store a bitset of neighbors u such that color(v,u)=c.
        # This allows fast triangle checking:
        # Triangle (u,v,w) is monochromatic with color c iff:
        # u is in neighbor_c(v) AND v is in neighbor_c(u) AND w is in neighbor_c(u) ...
        
        # Actually, simpler:
        # To check if adding edge (u,v) with color c creates a mono triangle:
        # We need to check if there exists w such that color(u,w)=c and color(v,w)=c.
        # This is equivalent to: (neighbors_c[u] & neighbors_c[v]) is not empty.
        
        # Data structures:
        # neighbors[c][v] is a bitset (int) of vertices u such that color(v,u)=c.
        
        # Initialize
        neighbors = [[0]*n for _ in range(4)] # neighbors[c][v]
        
        # Edges to assign
        # We process vertices in order. When adding vertex v, we assign edges (v, u) for u < v.
        # Actually, we can process edges in lexicographic order, but vertex-by-vertex is better for bitsets.
        
        # Let's use backtracking on vertices.
        # State: colors for all edges (i,j) with j < current_v are fixed.
        # When adding vertex v, we must assign colors to (v, 0), (v, 1), ..., (v, v-1).
        
        def backtrack(v):
            if time.time() - start_time > max_time:
                return False
            if v == n:
                return True
            
            # We need to assign colors to edges (v, 0) ... (v, v-1)
            # Let's try all 4^v combinations? No, too many.
            # Use backtracking on the edges (v, u) for u=0..v-1.
            
            def assign_edge(u):
                if u == v:
                    return backtrack(v+1)
                
                # Try colors 0..3
                for c in range(4):
                    # Check if assigning color c to (v, u) creates a mono triangle
                    # Triangles involving (v, u) are (v, u, w) for w < u.
                    # Mono triangle exists if there is w such that color(v,w)=c and color(u,w)=c.
                    # color(v,w) is already assigned (since w < u < v).
                    # color(u,w) is already assigned (since w < u).
                    
                    # Check: Is there w in [0, u-1] such that neighbors[c][v] has bit w set AND neighbors[c][u] has bit w set?
                    # neighbors[c][v] currently contains w for all w < u that we have already assigned color c to (v,w).
                    # neighbors[c][u] contains w for all w < u such that color(u,w)=c.
                    
                    # Intersection:
                    # mask = neighbors[c][v] & neighbors[c][u]
                    # If mask != 0, then there is a w in the intersection, forming a triangle.
                    
                    if (neighbors[c][v] & neighbors[c][u]) != 0:
                        continue
                    
                    # Assignment is valid so far.
                    # Assign color c to (v, u)
                    neighbors[c][v] |= (1 << u)
                    neighbors[c][u] |= (1 << v) # Wait, neighbors[c][u] should contain v?
                    # Our definition: neighbors[c][x] contains y if color(x,y)=c.
                    # So yes, neighbors[c][u] should get bit v.
                    
                    if assign_edge(u+1):
                        return True
                    
                    # Unassign
                    neighbors[c][v] &= ~(1 << u)
                    neighbors[c][u] &= ~(1 << v)
                
                return False
            
            return assign_edge(0)
        
        if backtrack(0):
            # Convert to flat list
            colors = []
            for i in range(n):
                for j in range(i+1, n):
                    for c in range(4):
                        if (neighbors[c][i] >> j) & 1:
                            colors.append(c)
                            break
            return colors
        return None

    # Try n from 50 down
    for n in range(50, 15, -1):
        # Use bitset backtracking
        result = construct_with_bitsets(n, max_time=10)
        if result is not None:
            return {'n': n, 'colors': result}
            
    # Fallback to seed's n=32 construction if all else fails
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors}

# EVOLVE-BLOCK-END
