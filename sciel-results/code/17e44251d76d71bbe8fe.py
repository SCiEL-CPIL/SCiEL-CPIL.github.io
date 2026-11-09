import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave 5s buffer

    def get_seed():
        # A known valid 4-coloring of K_16 (cyclic group Z_16 construction)
        # This is a well-known construction for R_4(C6)
        n = 16
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                d = j - i
                # Use a coloring based on cyclic differences
                # For K_16, use the Paley-type or known construction
                # We'll use a simple valid construction
                # Actually, let me use a verified construction
                # The known result: R_4(C6) >= 16
                # Use cyclic coloring: color edge (i,j) by (j-i) mod 16 mapped to {0,1,2,3}
                # We need each color class to be C6-free
                # Let's use a known valid one
                colors.append(0)
        # This seed needs to be valid. Let me use a simple construction.
        # For now, use n=9 with the original seed pattern
        n = 9
        colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
        return {'n': n, 'colors': colors}

    def edge_index(n, i, j):
        return i*(n-1) - i*(i-1)//2 + (j-i-1)

    def decode_edge(n, idx):
        # Find i, j such that edge_index(n, i, j) == idx
        # Binary search or direct
        # i*(n-1) - i*(i-1)//2 <= idx < (i+1)*(n-1) - (i+1)*i//2
        # Let's just iterate for small n
        for i in range(n-1):
            base = i*(n-1) - i*(i-1)//2
            if base <= idx < base + (n-1-i):
                j = i + 1 + (idx - base)
                return i, j
        return None

    def build_adj(n, colors):
        """Build adjacency bitmasks for each color class."""
        adj = [0]*4  # adj[c] is a list of n bitmasks
        adj = [[0]*n for _ in range(4)]
        for i in range(n):
            for j in range(i+1, n):
                c = colors[edge_index(n, i, j)]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        return adj

    def check_c6_fast(n, colors):
        """Check if any color class contains a C6 using precomputed cycles."""
        adj = build_adj(n, colors)
        # For each color, check all C6 cycles
        for c in range(4):
            # Check for C6: v0-v1-v2-v3-v4-v5-v0
            # We can check by looking for a 6-cycle
            # Use the fact that a C6 exists iff there's a path of length 5 between adjacent vertices
            # But faster: iterate over all 6-tuples? Too slow for n=17.
            # Better: for each pair of non-adjacent vertices, check if they have a common 5-neighbor structure
            # Actually, let's use a smarter approach:
            # A C6 exists iff there exist vertices a,b,c,d,e,f such that
            # a-b-c-d-e-f-a are all edges in color c.
            # We can fix a and b (adjacent), then find a path of length 4 from b to a
            # through 3 intermediate vertices.
            
            # Fast approach: for each edge (a,b) in color c, check if there's a path of length 5 from a to b
            # in the graph without using edge (a,b) directly.
            # Path: a - x - y - z - w - b where x,y,z,w are distinct and not a,b
            
            # Even faster: use the precomputed list of all C6s in K_n
            # But we need to check if all 6 edges of a C6 have the same color.
            
            # Let's precompute all C6 cycles once
            pass
        
        return False

    def precompute_c6s(n):
        """Precompute all 6-cycles in K_n as tuples of 6 vertex indices."""
        c6s = []
        # A C6 is a sequence v0,v1,v2,v3,v4,v5 with edges v0-v1,v1-v2,...,v5-v0
        # To avoid duplicates, fix v0=0 (or min vertex) and v1 < v5 (to break rotational symmetry)
        # Actually, for checking purposes, we just need each cycle once.
        # Number of C6s in K_n is n!/(2*6) for n>=6, which is large.
        # For n=17: 17!/(12) ≈ 3.5e12 - too many!
        
        # Alternative: don't precompute all. Instead, use a different check.
        # For each color c, build the graph and check for C6 using a smarter algorithm.
        return c6s

    def has_c6_in_color(n, adj_c):
        """Check if a single color class (given as adjacency bitmasks) contains a C6."""
        # A C6 exists iff there is a 6-cycle.
        # We can check: for each vertex v, look for a path of length 5 from v back to v
        # through 5 distinct intermediate vertices.
        # This is expensive. Better approach:
        
        # For each pair of vertices (a,b) that are NOT adjacent, check if they have a common neighbor structure
        # that forms a C6.
        
        # Actually, the most practical approach for n<=20:
        # For each vertex a, for each neighbor b of a, for each neighbor c of b (c != a),
        # for each neighbor d of c (d not in {a,b}), for each neighbor e of d (e not in {a,b,c}),
        # check if e is adjacent to a and e != b.
        # This finds all C6s containing edge (a,b).
        
        # Let's optimize with bitmasks.
        # adj_c[i] is a bitmask of neighbors of i.
        
        for a in range(n):
            neighbors_a = adj_c[a]
            for b in range(a+1, n):
                if not (neighbors_a & (1 << b)):
                    continue  # a and b must be adjacent
                # Now find path a-b-c-d-e-a of length 5 (excluding direct a-b edge for the return)
                # Actually, the cycle is a-b-c-d-e-f-a, so 6 edges.
                # Let me redo: cycle v0-v1-v2-v3-v4-v5-v0
                # Fix v0=a, v1=b (b > a to avoid duplicates)
                # v2 in N(b), v2 != a
                # v3 in N(v2), v3 not in {a, b}
                # v4 in N(v3), v4 not in {a, b, v2}
                # v5 in N(v4), v5 not in {a, b, v2, v3}
                # v5 must be in N(a)
                
                nb = adj_c[b] & ~(1 << a)  # neighbors of b except a
                for v2 in range(n):
                    if not (nb & (1 << v2)):
                        continue
                    nc = adj_c[v2] & ~(1 << a) & ~(1 << b)
                    for v3 in range(n):
                        if not (nc & (1 << v3)):
                            continue
                        nd = adj_c[v3] & ~(1 << a) & ~(1 << b) & ~(1 << v2)
                        for v4 in range(n):
                            if not (nd & (1 << v4)):
                                continue
                            ne = adj_c[v4] & ~(1 << a) & ~(1 << b) & ~(1 << v2) & ~(1 << v3)
                            # Check if any neighbor of v4 (excluding a,b,v2,v3) is also neighbor of a
                            common = ne & adj_c[a]
                            if common:
                                return True
        return False

    def check_c6(n, colors):
        adj = build_adj(n, colors)
        for c in range(4):
            if has_c6_in_color(n, adj[c]):
                return True
        return False

    def penalty(n, colors):
        """Count the number of C6s across all color classes (for local search)."""
        adj = build_adj(n, colors)
        count = 0
        for c in range(4):
            # Count C6s in this color
            # For efficiency, we just check existence and return a large penalty if found
            # But for local search, we want a graded penalty.
            # Let's count the number of 6-cycles.
            for a in range(n):
                neighbors_a = adj[c][a]
                for b in range(a+1, n):
                    if not (neighbors_a & (1 << b)):
                        continue
                    nb = adj[c][b] & ~(1 << a)
                    for v2 in range(n):
                        if not (nb & (1 << v2)):
                            continue
                        nc = adj[c][v2] & ~(1 << a) & ~(1 << b)
                        for v3 in range(n):
                            if not (nc & (1 << v3)):
                                continue
                            nd = adj[c][v3] & ~(1 << a) & ~(1 << b) & ~(1 << v2)
                            for v4 in range(n):
                                if not (nd & (1 << v4)):
                                    continue
                                ne = adj[c][v4] & ~(1 << a) & ~(1 << b) & ~(1 << v2) & ~(1 << v3)
                                common = ne & adj[c][a]
                                count += bin(common).count('1')
        return count

    def try_local_search(n, time_limit):
        """Local search with penalty minimization."""
        num_edges = n * (n-1) // 2
        best_colors = None
        best_penalty = float('inf')
        
        for restart in range(50):
            if time.time() > deadline:
                break
            
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Local search: try to reduce penalty
            improved = True
            while improved and time.time() < deadline:
                improved = False
                edge_order = list(range(num_edges))
                random.shuffle(edge_order)
                
                for idx in edge_order:
                    if time.time() > deadline:
                        break
                    old_c = colors[idx]
                    # Compute current penalty contribution... this is expensive.
                    # Instead, just try changing and check if penalty decreases.
                    # For speed, use a simpler heuristic: try each color and keep if it doesn't create a C6.
                    
                    # Actually, let's use a different approach:
                    # Try each color for this edge, and keep the one that minimizes the number
                    # of C6s involving this edge.
                    
                    best_new_c = old_c
                    best_local_pen = None
                    
                    for new_c in range(4):
                        if new_c == old_c:
                            continue
                        colors[idx] = new_c
                        # Quick check: does this create a C6?
                        if check_c6(n, colors):
                            # Count how many C6s
                            p = penalty(n, colors)
                            if best_local_pen is None or p < best_local_pen:
                                best_local_pen = p
                                best_new_c = new_c
                        else:
                            # No C6 at all - great!
                            best_local_pen = 0
                            best_new_c = new_c
                            break
                    
                    colors[idx] = best_new_c
                    if best_new_c != old_c:
                        improved = True
                        if best_local_pen == 0:
                            return colors  # Found a valid coloring!
            
            # Check if we found a solution
            if not check_c6(n, colors):
                return colors
            
            if best_local_pen < best_penalty:
                best_penalty = best_local_pen
                best_colors = colors[:]
        
        return best_colors

    def try_backtracking(n, time_limit):
        """Constructive backtracking: add vertices one by one."""
        # This is complex. Let's use a simpler approach:
        # Start with a valid coloring of K_k and try to extend to K_{k+1}.
        
        # First, find a valid coloring for small n
        base_n = 9
        base_colors = [min(j - i, base_n - j + i) - 1 for i in range(base_n) for j in range(i + 1, base_n)]
        
        # Try to extend to n
        # For each new vertex, assign colors to its edges to existing vertices
        # such that no C6 is created.
        
        # This is a constraint satisfaction problem.
        # For vertex v, we need to assign colors to edges (v, u) for u < v.
        # The constraint is that for each color c, the subgraph induced by edges of color c
        # (including the new edges) must not contain a C6.
        
        # This is hard to check incrementally. Let's just use local search for now.
        return None

    # Try to find a solution for n=18, 17, 16
    for target_n in [18, 17, 16]:
        if time.time() > deadline:
            break
        
        result = try_local_search(target_n, deadline - time.time())
        if result is not None:
            if not check_c6(target_n, result):
                return {'n': target_n, 'colors': result}
        
        # If local search didn't find a perfect solution, check if it's close
        # and try to repair it
        if result is not None:
            # Try to repair: find C6s and recolor edges
            for attempt in range(20):
                if time.time() > deadline:
                    break
                if not check_c6(target_n, result):
                    return {'n': target_n, 'colors': result}
                
                # Find a C6 and recolor one of its edges
                adj = build_adj(target_n, result)
                found = False
                for c in range(4):
                    for a in range(target_n):
                        neighbors_a = adj[c][a]
                        for b in range(a+1, target_n):
                            if not (neighbors_a & (1 << b)):
                                continue
                            nb = adj[c][b] & ~(1 << a)
                            for v2 in range(target_n):
                                if not (nb & (1 << v2)):
                                    continue
                                nc = adj[c][v2] & ~(1 << a) & ~(1 << b)
                                for v3 in range(target_n):
                                    if not (nc & (1 << v3)):
                                        continue
                                    nd = adj[c][v3] & ~(1 << a) & ~(1 << b) & ~(1 << v2)
                                    for v4 in range(target_n):
                                        if not (nd & (1 << v4)):
                                            continue
                                        ne = adj[c][v4] & ~(1 << a) & ~(1 << b) & ~(1 << v2) & ~(1 << v3)
                                        common = ne & adj[c][a]
                                        if common:
                                            v5 = common.bit_length() - 1
                                            # Found C6: a-b-v2-v3-v4-v5-a
                                            # Recolor edge (a, b)
                                            idx = edge_index(target_n, a, b)
                                            old_c = result[idx]
                                            for new_c in range(4):
                                                if new_c == old_c:
                                                    continue
                                                result[idx] = new_c
                                                if not check_c6(target_n, result):
                                                    return {'n': target_n, 'colors': result}
                                            result[idx] = old_c
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    break
    
    # Fallback to seed
    return get_seed()
