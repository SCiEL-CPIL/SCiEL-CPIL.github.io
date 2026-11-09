"""Four-colourings of complete graphs without monochromatic hexagons (R_4(C6))."""
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START

def _build_adjacency(n, colors):
    """Build adjacency lists for each color class."""
    adj = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].add((i, j))
            idx += 1
    return adj

def _find_c6_in_color(adj_color):
    """Find a 6-cycle in a given color class. Returns the cycle as list of vertices or None.
    Uses BFS-based approach: for each edge, try to find a path of length 4 between endpoints."""
    # Build adjacency as dict of sets for faster lookup
    adj_dict = defaultdict(set)
    for (u, v) in adj_color:
        adj_dict[u].add(v)
        adj_dict[v].add(u)
    
    vertices = list(adj_dict.keys())
    
    # For each edge (u,v), try to find a path of length 4 from u to v
    # This would form a 6-cycle: u - a - b - c - d - v - u
    for (u, v) in adj_color:
        # BFS from u up to depth 4, trying to reach v
        # But we need simple paths, so use DFS with visited set
        # Limit search for efficiency
        if len(adj_dict) < 6:
            continue
        
        # Try to find path u -> a -> b -> c -> d -> v where all edges are in this color
        # and all vertices distinct
        def find_path(start, target, depth, max_depth, visited, path):
            if depth == max_depth:
                if target in adj_dict.get(start, set()):
                    return path + [target]
                return None
            for nxt in adj_dict.get(start, set()):
                if nxt not in visited and nxt != target:
                    result = find_path(nxt, target, depth + 1, max_depth, visited | {nxt}, path + [nxt])
                    if result is not None:
                        return result
            return None
        
        # We need a path of length 4 from u to v (4 intermediate vertices)
        # So we look for u -> a -> b -> c -> d -> v
        # That's 4 edges from u to v, meaning path length 4 (5 vertices including u and v)
        # Actually for a 6-cycle: u-v is one edge, and we need 5 more edges forming a path from v back to u
        # Or: the cycle is u, v, a, b, c, d, u - that's 6 edges
        # So we need a path of length 5 from u to v not using the direct edge, OR
        # equivalently, a path of length 4 from u to some neighbor of v... 
        
        # Simpler: a 6-cycle contains edge (u,v) and a path of length 5 from u to v.
        # Equivalently, it contains a path of length 4 from u to v where the path doesn't use edge (u,v).
        # Wait: 6-cycle = 6 edges. If one edge is (u,v), the other 5 edges form a path from u to v of length 5.
        # So we need: u - x1 - x2 - x3 - x4 - x5 - v, all in color, all distinct.
        # That's a path of length 5 (6 vertices) from u to v.
        
        # Alternative approach: enumerate all 6-tuples? Too slow.
        # Better: for each pair (u,v) in the color, check if there's a path of length 5.
        # Use BFS with depth limit.
        
        # Actually, let's use a different strategy: 
        # For each color class, check if it contains C6 by looking at all 6-vertex subsets? No, too many.
        
        # Use the fact that C6 has diameter 3. For each vertex pair at distance 3 in the color graph,
        # check if they're connected by two disjoint paths of length 2.
        
        # Let me use a simpler but potentially slower method: 
        # For each triple (a,b,c) forming a path a-b-c in the color, and each triple (d,e,f) forming path d-e-f,
        # check if {a,b,c,d,e,f} forms a C6. This is getting complicated.
        
        # Simplest correct approach: for each edge (u,v), do a bounded DFS to find a path of length 5 from u to v.
        pass
    
    # Let me use a cleaner approach: enumerate all 6-cycles by checking all combinations of 6 vertices
    # is too slow. Instead, use the following:
    # A graph contains C6 iff there exist vertices v1,...,v6 such that vi-v(i+1) are edges and v6-v1 is an edge.
    # We can do this by: for each pair (u,v) at graph distance exactly 3, check if there are two 
    # internally disjoint paths of length 2 between u and v.
    
    # Compute all-pairs shortest paths within this color class
    # For n up to 20, this is feasible
    
    # Build adjacency matrix
    n_local = n
    adj_mat = [[False] * n_local for _ in range(n_local)]
    for (u, v) in adj_color:
        adj_mat[u][v] = True
        adj_mat[v][u] = True
    
    # Compute distances
    INF = 999
    dist = [[INF] * n_local for _ in range(n_local)]
    for i in range(n_local):
        dist[i][i] = 0
        for j in range(n_local):
            if adj_mat[i][j]:
                dist[i][j] = 1
    
    # Floyd-Warshall
    for k in range(n_local):
        for i in range(n_local):
            if dist[i][k] == INF:
                continue
            for j in range(n_local):
                if dist[k][j] == INF:
                    continue
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    # For each pair (u,v) with dist[u][v] == 3, check if there are two internally disjoint paths of length 2
    # A path of length 2 from u to v is u-w-v where w is a common neighbor.
    # Two internally disjoint paths of length 2 means two distinct common neighbors w1, w2.
    for u in range(n_local):
        for v in range(u + 1, n_local):
            if dist[u][v] != 3:
                continue
            # Find common neighbors of u and v
            common = []
            for w in range(n_local):
                if w != u and w != v and adj_mat[u][w] and adj_mat[v][w]:
                    common.append(w)
            if len(common) >= 2:
                # We have u-w1-v and u-w2-v, but we need a 6-cycle: u-w1-?-w2-v-?
                # Actually, two paths of length 2 give a 4-cycle, not 6.
                # For a 6-cycle with diameter 3: u-a-b-c-d-e-u where dist(u,e)=3 via u-a-b-e or u-d-c-e
                # Let me reconsider.
                
                # A 6-cycle: v1-v2-v3-v4-v5-v6-v1
                # dist(v1,v4) = 3 (via v1-v2-v3-v4 and v1-v6-v5-v4)
                # So for pair (v1,v4) at distance 3, there are two paths of length 3 between them.
                # A path of length 3: v1-a-b-v4
                # So we need two internally disjoint paths of length 3 between u and v.
                
                # Find all paths of length 3 from u to v
                paths_len3 = []
                for a in range(n_local):
                    if a == u or a == v:
                        continue
                    if not adj_mat[u][a]:
                        continue
                    for b in range(n_local):
                        if b == u or b == v or b == a:
                            continue
                        if not adj_mat[a][b]:
                            continue
                        if adj_mat[b][v]:
                            paths_len3.append((a, b))
                
                # Check if any two paths are internally disjoint
                found = False
                for i in range(len(paths_len3)):
                    for j in range(i + 1, len(paths_len3)):
                        a1, b1 = paths_len3[i]
                        a2, b2 = paths_len3[j]
                        # Internally disjoint: {a1,b1} ∩ {a2,b2} = empty
                        if a1 != a2 and a1 != b2 and b1 != a2 and b1 != b2:
                            found = True
                            break
                    if found:
                        break
                if found:
                    return True
    return False

def _count_c6s(n, colors):
    """Count the number of monochromatic 6-cycles across all 4 colors."""
    adj = _build_adjacency(n, colors)
    count = 0
    for c in range(4):
        if _find_c6_in_color(adj[c]):
            count += 1
    return count

def _is_valid(n, colors):
    """Check if no color class contains a 6-cycle."""
    adj = _build_adjacency(n, colors)
    for c in range(4):
        if _find_c6_in_color(adj[c]):
            return False
    return True

def _initial_random(n):
    """Generate a random 4-coloring of K_n."""
    num_edges = n * (n - 1) // 2
    return [random.randint(0, 3) for _ in range(num_edges)]

def _edge_index(i, j, n):
    """Get the index of edge (i,j) in the lexicographic color list."""
    # Edges are ordered: (0,1),(0,2),...,(0,n-1),(1,2),(1,3),...
    # Index = sum_{k=0}^{i-1} (n-1-k) + (j - i - 1)
    return i * (2 * n - i - 1) // 2 + (j - i - 1)

def _get_color(colors, i, j, n):
    return colors[_edge_index(i, j, n)]

def _set_color(colors, i, j, n, c):
    colors[_edge_index(i, j, n)] = c

def _count_c6s_fast(n, colors):
    """Fast count of monochromatic C6s using adjacency matrices."""
    # Build adjacency lists per color
    adj = [set() for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c].add((i, j))
            idx += 1
    
    total = 0
    for c in range(4):
        if _find_c6_in_color(adj[c]):
            total += 1
    return total

def _simulated_annealing(n, time_limit):
    """Run simulated annealing to find a valid 4-coloring of K_n."""
    start_time = time.time()
    
    # Start with random coloring
    colors = _initial_random(n)
    
    # Initial evaluation
    current_cost = _count_c6s_fast(n, colors)
    
    best_colors = colors[:]
    best_cost = current_cost
    
    if best_cost == 0:
        return best_colors
    
    T = 5.0  # Initial temperature
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        if T < 0.001:
            break
        
        # Pick a random edge
        idx = random.randint(0, n * (n - 1) // 2 - 1)
        # Find which edge this is
        # Reverse of _edge_index
        # i*(2n-i-1)/2 <= idx < (i+1)*(2n-i-2)/2
        # Let's just iterate
        i, j = None, None
        running = 0
        for ii in range(n):
            count = n - 1 - ii
            if running + count > idx:
                i = ii
                j = ii + 1 + (idx - running)
                break
            running += count
        
        old_c = colors[idx]
        new_c = random.choice([c for c in range(4) if c != old_c])
        colors[idx] = new_c
        
        new_cost = _count_c6s_fast(n, colors)
        delta = new_cost - current_cost
        
        if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_colors = colors[:]
                if best_cost == 0:
                    return best_colors
        else:
            colors[idx] = old_c
        
        T *= cooling_rate
    
    return best_colors

def _hill_climbing(n, time_limit):
    """Simple hill-climbing: keep flipping edges if it reduces C6 count."""
    start_time = time.time()
    colors = _initial_random(n)
    current_cost = _count_c6s_fast(n, colors)
    best_colors = colors[:]
    best_cost = current_cost
    
    if best_cost == 0:
        return best_colors
    
    no_improve = 0
    while time.time() - start_time < time_limit and no_improve < 5000:
        idx = random.randint(0, n * (n - 1) // 2 - 1)
        old_c = colors[idx]
        new_c = random.choice([c for c in range(4) if c != old_c])
        colors[idx] = new_c
        new_cost = _count_c6s_fast(n, colors)
        
        if new_cost < current_cost:
            current_cost = new_cost
            no_improve = 0
            if new_cost < best_cost:
                best_cost = new_cost
                best_colors = colors[:]
                if best_cost == 0:
                    return best_colors
        else:
            colors[idx] = old_c
            no_improve += 1
    
    return best_colors

def _incremental_search():
    """Try to build valid colorings for increasing n."""
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Try n from 17 down, but first try to find valid ones for larger n
    # Actually, let's try n=17, 18, 19 with SA
    
    # Start with n=17
    for target_n in [17, 18, 19]:
        if time.time() > deadline - 30:
            break
        
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        # Try multiple restarts
        found = None
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            colors = _simulated_annealing(target_n, min(remaining / 3, 20))
            if _is_valid(target_n, colors):
                found = colors
                break
        
        if found is not None:
            return target_n, found
    
    # If we couldn't find n>=17, fall back to known n=9
    return 9, [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Keep 10s safety margin
    
    try:
        n, colors = _incremental_search()
    except Exception:
        n = 9
        colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Verify validity
    if not _is_valid(n, colors):
        # Fall back to known valid n=9
        n = 9
        colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
