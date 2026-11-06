import time
import random
import numpy as np

def _build_k32():
    """
    Construct the 32-vertex 4-coloring of K_32 without monochromatic triangles.
    This is based on the structure of GF(16) and cubic residues.
    
    The vertices are partitioned into two sets of 16: V0 and V1.
    Edges between V0 and V1 are all color 3.
    Edges within V0 and V1 are colored 0, 1, or 2 based on the difference in their indices 
    mapped through the multiplicative group of GF(16).
    """
    # Multiplication in GF(16) with primitive polynomial x^4 + x + 1 (0b10011)
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    # Generate log map for non-zero elements
    log_map = {}
    x = 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)
    
    # The cubic residues in GF(16)^* form a subgroup of size 5? 
    # Actually, the standard construction for R(3,3,3,3) > 32 uses a specific difference set or similar.
    # A known valid coloring for 32 vertices:
    # Vertices 0..15 (Block A) and 16..31 (Block B).
    # Edges (i, j) with i,j in A: color = (log_diff % 3) where diff is in multiplicative group.
    # But log is only for non-zero. 0 is identity? 
    # Let's use the explicit 32x32 matrix often cited in literature for this problem.
    # Since generating it perfectly from scratch in GF(16) is prone to off-by-one errors in the field arithmetic definition,
    # and I have a reliable seed in the prompt, I will implement the logic from the prompt's _build_k32 
    # but verify it. The prompt's logic:
    # if i//16 == j//16: colors.append(log_map[(i%16)^(j%16)] % 3)
    # else: colors.append(3)
    
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            if i // 16 == j // 16:
                # i%16 and j%16 are distinct in [0,15]
                diff = (i % 16) ^ (j % 16)
                # diff is non-zero
                colors.append(log_map[diff] % 3)
            else:
                colors.append(3)
                
    return np.array(colors, dtype=np.uint8)

_K32_BASE = _build_k32()

def _edge_index(i, j, n):
    """Calculate the flat index of edge (i, j) in a K_n graph (i < j)."""
    if i > j:
        i, j = j, i
    # Sum of edges in previous rows:
    # Row 0: n-1
    # Row 1: n-2
    # ...
    # Row i: n-1-i
    # Edges before row i: sum_{k=0}^{i-1} (n-1-k) = i*(n-1) - (i-1)*i//2
    # In row i, j is the (j-i-1)-th edge (0-indexed)
    return i * (n - 1) - (i * (i - 1)) // 2 + (j - i - 1)

def _validate(colors, n):
    """
    Check if the coloring has no monochromatic triangles.
    colors: array of length n*(n-1)//2
    """
    if n <= 3:
        return True
    
    # Build adjacency bitmasks for each color
    # adj[c][i] is a bitmask of neighbors of i in color c
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1

    for c in range(4):
        for i in range(n):
            nbrs = adj[c][i]
            # If any neighbor j of i (in color c) has a common neighbor with i (in color c), 
            # then (i, j, k) is a triangle.
            # We just need to check if adj[c][j] & nbrs is non-zero for any j in nbrs.
            temp = nbrs
            while temp:
                lsb = temp & (-temp)
                j = lsb.bit_length() - 1
                temp ^= lsb
                if adj[c][j] & nbrs:
                    return False
    return True

def _count_triangles(colors, n):
    """Count total monochromatic triangles."""
    if n < 3:
        return 0
    
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
    
    count = 0
    for c in range(4):
        for i in range(n):
            nbrs = adj[c][i]
            temp = nbrs
            while temp:
                lsb = temp & (-temp)
                j = lsb.bit_length() - 1
                temp ^= lsb
                if i < j:
                    # Common neighbors of i and j in color c
                    # These form triangles (i, j, k)
                    common = adj[c][j] & nbrs
                    count += common.bit_count()
    # Each triangle is counted 3 times (once per vertex)
    return count // 3

def _try_add_vertex(colors, n, deadline):
    """
    Try to add a new vertex (index n) to the graph.
    Returns (new_colors, new_n) if successful, else (None, n).
    """
    # Build adjacency for current n
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    # For the new vertex v = n, we need to assign colors to edges (n, u) for u in 0..n-1.
    # Let x_u be the color of edge (n, u).
    # Constraint: For any color c, the set of u such that x_u = c must be an independent set in G_c (the graph of color c).
    # That is, if x_u = c and x_w = c, then edge (u, w) must NOT be color c.
    
    # We can model this as a CSP.
    # Variables: x_0, ..., x_{n-1}
    # Domain: {0, 1, 2, 3}
    # Constraints: For all u < w, if x_u == x_w == c, then adj[c][u] must not have bit w set.
    
    # Backtracking search with forward checking
    x = [-1] * n
    
    def is_valid_assignment(u, c):
        # Check if assigning x_u = c conflicts with any already assigned x_w
        for w in range(n):
            if x[w] != -1 and x[w] == c:
                # Check if (u, w) is an edge of color c
                if (adj[c][u] >> w) & 1:
                    return False
        return True

    def backtrack(k):
        if time.time() > deadline - 2:
            return False
        if k == n:
            return True
            
        # Heuristic: Try colors in random order or most constrained first?
        # For simplicity and speed, try 0,1,2,3
        for c in range(4):
            if is_valid_assignment(k, c):
                x[k] = c
                if backtrack(k + 1):
                    return True
                x[k] = -1
        return False

    # To improve success rate, we can randomize the order of vertices or colors
    # But n is up to ~40-50, so backtracking might be slow if constraints are tight.
    # However, if we start from a valid 32-vertex graph, the constraints are structured.
    # Let's try a few random shuffles of the vertex order for the backtracking variable ordering.
    
    # Actually, the constraints are global. Shuffling variable order helps break symmetries or find solutions faster.
    order = list(range(n))
    
    for attempt in range(20):
        if time.time() > deadline - 2:
            break
        random.shuffle(order)
        
        # We need to map original index to order index in backtracking?
        # It's easier to just shuffle the colors tried or the vertex order.
        # Let's stick to simple backtracking on natural order first, as the structure of K32 is regular.
        
        x = [-1] * n
        if backtrack(0):
            # Construct new colors array
            new_colors = colors.copy()
            # Append colors for edges (n, u) for u=0..n-1
            # The order in the flat array for the new vertex n:
            # Edges (0, n), (1, n), ..., (n-1, n)
            # These come after all edges of K_n.
            # So we just append x[0], x[1], ..., x[n-1]
            new_colors = np.concatenate([colors, np.array(x, dtype=np.uint8)])
            return new_colors, n + 1
            
    return None, n

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Start with known good solution
    best_n = 32
    best_colors = _K32_BASE.copy()
    
    current_n = best_n
    current_colors = best_colors.copy()
    current_energy = 0 # _count_triangles(current_colors, current_n)
    
    # Precompute edges list for SA
    def get_edges(n):
        edges = []
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j, idx))
                idx += 1
        return edges
    
    edges = get_edges(current_n)
    
    temp = 1.0
    cooling = 0.995
    
    # Main Loop
    while time.time() < deadline - 5:
        
        # 1. If current energy is 0, try to extend
        if current_energy == 0:
            new_colors, new_n = _try_add_vertex(current_colors, current_n, deadline)
            if new_n > current_n:
                current_n = new_n
                current_colors = new_colors
                current_energy = _count_triangles(current_colors, current_n)
                if current_energy == 0:
                    # Update best
                    best_n = current_n
                    best_colors = current_colors.copy()
                    edges = get_edges(current_n)
                    # Restart temperature or keep? Keep.
                else:
                    # If extension failed to maintain 0 energy, we have a broken graph.
                    # We can try to fix it with SA, or revert.
                    # Reverting is safer for maintaining a valid baseline, but SA might fix it.
                    # Let's try to fix it with a short SA burst.
                    pass
            
        # 2. Local Search (Simulated Annealing) to reduce energy
        # If energy > 0, we MUST reduce it to extend further.
        # If energy == 0, we can still explore to find "better" structures? 
        # For this problem, any valid coloring is equal (score 1.0 based on n). 
        # So the only goal is to increase n.
        # Thus, SA is only useful if we are stuck (energy > 0) or if we want to explore 
        # the space of valid colorings to find one that allows easier extension.
        
        if current_energy > 0:
            # Perform SA step
            i, j, idx = random.choice(edges)
            old_c = current_colors[idx]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
                
            # Calculate Delta Energy
            # We need the current adjacency structure to calculate delta efficiently.
            # For n <= 62, building adj arrays is O(n^2) ~ 4000 ops. 
            # Doing this every step might be too slow if SA runs for 100k steps.
            # Better: Maintain adj arrays incrementally.
            
            # Let's maintain adj as list of lists of bitmasks.
            # This requires passing adj to the loop or making it global/local mutable.
            
            # To keep code simple and robust within time limits:
            # If n is small (< 40), full recount is ~ O(n^3) or O(n^2) with bitmasks.
            # _count_triangles is O(n^2) with bitmasks.
            # For n=50, n^2=2500. 2500 ops per step.
            # 100,000 steps * 2500 = 2.5e8 ops. Python is ~1e7-1e8 ops/sec.
            # This is too slow for 100k steps.
            
            # We need incremental delta.
            # Let's rebuild adj once per "epoch" or just accept slower SA if we do fewer steps.
            # Or, implement incremental delta properly.
            
            # Incremental Delta Implementation:
            # We need adj matrices for current colors.
            # Let's maintain `adj` globally in the construct function scope.
            
            if not hasattr(construct, '_adj_cache'):
                construct._adj_cache = None
            
            # Rebuild adj if colors changed significantly or n changed?
            # Simpler: Rebuild adj every 100 steps or if n changed.
            # Or, just use the fast O(n) delta formula which requires knowing neighbors.
            
            # Let's use the O(n) delta formula.
            # We need to know the colors of all other edges connected to i and j.
            # We can look them up from current_colors array.
            
            # Delta for changing edge (i, j) from c_old to c_new:
            # 1. Remove triangles in c_old containing (i, j).
            #    These are vertices k such that color(i, k) == c_old and color(j, k) == c_old.
            # 2. Add triangles in c_new containing (i, j).
            #    These are vertices k such that color(i, k) == c_new and color(j, k) == c_new.
            
            def get_color(u, v):
                if u > v:
                    u, v = v, u
                return current_colors[_edge_index(u, v, current_n)]
            
            delta = 0
            for k in range(current_n):
                if k == i or k == j:
                    continue
                ci_k = get_color(i, k)
                cj_k = get_color(j, k)
                
                # Contribution to c_old
                if ci_k == old_c and cj_k == old_c:
                    delta -= 1  # Remove a triangle
                # Contribution to c_new
                if ci_k == new_c and cj_k == new_c:
                    delta += 1  # Add a triangle
            
            new_energy = current_energy + delta
            
            if new_energy < current_energy or random.random() < np.exp(-delta / max(temp, 1e-10)):
                current_colors[idx] = new_c
                current_energy = new_energy
                if current_energy == 0:
                    # Update best if n is same? No, n is same. 
                    # But we might have a different structure that allows extension.
                    # We don't track "best structure" for extension, we just try to extend.
                    pass
            else:
                # Revert
                pass # current_colors[idx] was not changed
                
            temp *= cooling
            if temp < 0.01:
                temp = 0.01
            
        else:
            # Energy is 0.
            # We tried to extend and failed.
            # We can perturb the solution slightly to escape local minima in the "extensibility" landscape.
            # Pick a random edge and change it (accepting energy > 0) to explore neighbors.
            i, j, idx = random.choice(edges)
            old_c = current_colors[idx]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            current_colors[idx] = new_c
            current_energy = _count_triangles(current_colors, current_n)
            # Now energy > 0 (likely). The loop will try to reduce it.
            # This acts as a random restart of the local search around the current valid solution.
            
    # Final Validation
    if time.time() < deadline - 1:
        if not _validate(best_colors, best_n):
            # Fallback
            best_n = 32
            best_colors = _K32_BASE.copy()
            
    return {'n': best_n, 'colors': best_colors.tolist()}

if __name__ == '__main__':
    res = construct()
    print(f"n: {res['n']}")
