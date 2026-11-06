import time
import random
import sys

_START_TIME = time.time()
_DEADLINE = 115.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _seed_32():
    """Greenwood-Gleason colouring of K32."""
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
            if i // 16 == j // 16:
                colors.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors.append(3)
    return {'n': 32, 'colors': colors}

def _compute_index(n, i, j):
    """Compute the index of pair (i,j) in lexicographic edge ordering."""
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def _build_adj(colors, n):
    """Build adjacency bitsets for a given coloring."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    return adj

def _count_mono_triangles(adj, n):
    """Count total monochromatic triangles using bitsets."""
    count = 0
    for c in range(4):
        for i in range(n):
            # Neighbors of i in color c, greater than i
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
            # For each neighbor j, count common neighbors k > j
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                count += common.bit_count()
    return count

def _find_one_mono_triangle(adj, n):
    """Find one monochromatic triangle, return (i, j, k, c) or None."""
    for c in range(4):
        for i in range(n):
            neigh = adj[c][i] & ~((1 << (i + 1)) - 1)
            j_mask = neigh
            while j_mask:
                j_bit = j_mask & (-j_mask)
                j = j_bit.bit_length() - 1
                j_mask ^= j_bit
                common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                if common:
                    k_bit = common & (-common)
                    k = k_bit.bit_length() - 1
                    return (i, j, k, c)
    return None

def _local_search(n, time_budget):
    """
    Local search to find a valid 4-coloring of K_n.
    Returns colors list if successful, None otherwise.
    """
    m = n * (n - 1) // 2
    start_time = time.time()
    
    max_restarts = 200
    for restart in range(max_restarts):
        if time.time() - start_time > time_budget:
            break
            
        # Initialize random coloring
        colors = [random.randint(0, 3) for _ in range(m)]
        adj = _build_adj(colors, n)
        
        # Local search loop
        while True:
            if time.time() - start_time > time_budget:
                return None
            
            tri = _find_one_mono_triangle(adj, n)
            if tri is None:
                return colors
            
            i, j, k, c = tri
            
            # Choose an edge to recolor. Randomly pick one of the 3 edges.
            edges = [(i, j), (i, k), (j, k)]
            edge = edges[random.randint(0, 2)]
            u, v = edge
            
            # Determine best new color
            best_c = -1
            min_new_tris = float('inf')
            
            for new_c in range(4):
                if new_c == c:
                    continue
                # Count new triangles formed by (u, v) with color new_c
                # These are vertices w such that (u, w) and (v, w) are color new_c
                new_tris = (adj[new_c][u] & adj[new_c][v]).bit_count()
                
                if new_tris < min_new_tris:
                    min_new_tris = new_tris
                    best_c = new_c
                    
            if best_c == -1:
                best_c = (c + 1) % 4
                
            # Recolor
            idx = _compute_index(n, u, v)
            colors[idx] = best_c
            
            # Update adj
            adj[c][u] &= ~(1 << v)
            adj[c][v] &= ~(1 << u)
            adj[best_c][u] |= (1 << v)
            adj[best_c][v] |= (1 << u)

def _greedy_construction(n, time_budget):
    """
    Greedy randomized construction.
    Iteratively adds vertices, selecting colors to minimize new triangles.
    """
    start_time = time.time()
    m = n * (n - 1) // 2
    colors = [0] * m
    
    # We build the graph vertex by vertex.
    # adj[c][i] bitmask of neighbors.
    adj = [[0] * n for _ in range(4)]
    
    # Precompute index offsets? Or just compute on the fly.
    # idx(i, j) = i*(n-1) - i*(i+1)//2 + (j-i-1)
    
    # Start with vertex 0. No edges.
    # Add vertex 1. Edge (0,1).
    # ...
    
    # To make it fast, we should probably just do local search from a greedy start.
    # Let's try a specific greedy heuristic:
    # For each new vertex v, assign colors to edges (u, v) for u < v.
    # Minimize the number of new triangles.
    
    # This is O(n^3 * 4) which is 50^3 * 4 = 500,000 operations. Very fast.
    
    # Let's implement this greedy construction.
    
    # Initialize
    for v in range(1, n):
        # Determine colors for edges (u, v) for u in 0..v-1
        # We want to minimize sum over u of (common neighbors of u and v in color c)
        
        # Actually, adding vertex v creates triangles (u, w, v) where u, w < v.
        # For each edge (u, v) assigned color c, it forms a triangle with (u, w) if (u, w) is also color c and w < v.
        # So for a fixed v, and for each u < v, if we assign color c to (u, v),
        # the cost is the number of w < u such that (u, w) is color c.
        
        # Let's define cost[u][c] = number of w < u such that color(u, w) == c.
        # Then for vertex v, we want to assign colors c_u to edges (u, v)
        # to minimize sum_{u < v} cost[u][c_u].
        
        # This is a minimum weight matching / assignment problem?
        # No, each edge (u, v) is independent in terms of cost calculation?
        # Yes! The cost of coloring (u, v) with c depends only on the existing edges (u, w) for w < u.
        # It does not depend on the colors of other edges (u', v) for u' != u.
        # Wait, is that true?
        # Triangle (u, w, v) requires (u, v), (u, w), (w, v) all same color.
        # If we are adding v, we are setting colors for (u, v) and (w, v).
        # So the color of (w, v) matters for the triangle (u, w, v).
        # So the costs are coupled.
        
        # Example: u=1, w=0, v=2.
        # Edges: (0,1), (0,2), (1,2).
        # If (0,1) is color 0.
        # We choose (0,2) and (1,2).
        # If (0,2)=0 and (1,2)=0, we have a triangle.
        # Cost of (0,2) being 0 depends on (0,1)? No, (0,1) is fixed.
        # The triangle is formed if (0,2)==(0,1) AND (1,2)==(0,1).
        # So the "cost" of (0,2) is 0 if we don't care about (1,2) yet?
        
        # This is a binary quadratic problem.
        # However, for small n, we can just do local search.
        # The greedy construction might be too complex to implement correctly and fast in this time.
        # Let's stick to local search but with a better initialization.
        
        # Initialization: Random.
        # Maybe "Greedy Random":
        # Start with empty graph.
        # Add vertices 0..n-1.
        # When adding v, for each u < v, pick color randomly?
        # Or pick color that minimizes immediate triangles?
        
        # Let's try: For each v, for each u < v, pick color c that minimizes
        # number of w < u such that (u, w) is c.
        # This ignores the coupling with (w, v), but might be a good heuristic.
        
        pass # We will implement this in the main construct if needed, but local search is safer.

def construct():
    try:
        best = _seed_32()
        best_n = 32
        
        # Try local search for larger n.
        # Targets: 50, 45, 40, 35.
        
        # Time budget allocation
        elapsed = time.time() - _START_TIME
        remaining = _DEADLINE - elapsed
        if remaining < 5:
            return best
            
        targets = [50, 45, 40, 35]
        
        # We have ~110s.
        # Let's try 50 with 60s.
        # If fail, try 45 with 30s.
        # If fail, try 40 with 10s.
        
        for target_n in targets:
            if target_n <= best_n:
                continue
            
            elapsed = time.time() - _START_TIME
            remaining = _DEADLINE - elapsed
            if remaining < 5:
                break
                
            # Allocate budget
            if target_n == 50:
                budget = min(remaining - 5, 60)
            elif target_n == 45:
                budget = min(remaining - 5, 30)
            else:
                budget = min(remaining - 5, 10)
                
            if budget < 2:
                break
                
            result = _local_search(target_n, budget)
            if result is not None:
                return {'n': target_n, 'colors': result}
                
        return best
    except Exception as e:
        try:
            return _seed_32()
        except:
            return {'n': 1, 'colors': []}

if __name__ == "__main__":
    res = construct()
    print(res['n'])
