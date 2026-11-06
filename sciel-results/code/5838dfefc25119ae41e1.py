"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # We'll try to build a large 4-coloring of K_n with no monochromatic triangle.
    # Start from the known 32-vertex Greenwood-Gleason construction, then try to extend.

    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    # Build the 32-vertex base coloring
    log_map, x = {}, 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    def base_colors(n=32):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                if i // 16 == j // 16:
                    colors.append(log_map[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors

    def color_index(n, i, j):
        """Get index in colors array for edge (i,j), i<j."""
        # Lexicographic order: for each i, edges (i, i+1), (i, i+2), ..., (i, n-1)
        # Then (i+1, i+2), ...
        # Index = sum_{k=0}^{i-1} (n-1-k) + (j - i - 1)
        return i * (2*n - i - 1) // 2 + (j - i - 1)

    def has_mono_triangle(colors, n, tri):
        """Check if triangle (a,b,c) is monochromatic."""
        a, b, c = tri
        if a > b: a, b = b, a
        if b > c: b, c = c, b
        if a > b: a, b = b, a
        c1 = colors[color_index(n, a, b)]
        c2 = colors[color_index(n, a, c)]
        c3 = colors[color_index(n, b, c)]
        return c1 == c2 == c3

    def count_mono_triangles(colors, n):
        count = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    if has_mono_triangle(colors, n, (a,b,c)):
                        count += 1
        return count

    # Start with 32-vertex base
    n = 32
    colors = base_colors(n)
    best_n = n
    best_colors = colors[:]

    # Try to extend by adding vertices one at a time
    # For each new vertex, we need to assign colors to edges to all existing vertices
    # such that no monochromatic triangle is formed.
    
    # Greedy extension with backtracking
    def try_extend(colors, n, max_new=10):
        """Try to add up to max_new vertices. Returns (new_n, new_colors) or (n, colors)."""
        cur_colors = colors[:]
        cur_n = n
        
        for new_v in range(max_new):
            if time.time() > deadline:
                break
            success = False
            # Try all possible color assignments for edges from new vertex to existing
            # There are 4^n possible assignments, but we can prune heavily.
            # Use constraint propagation: for each pair (i,j) of existing vertices,
            # the color of (new, i) and (new, j) cannot both equal color(i,j)
            # if (i,j) already has that color (would form mono triangle with new).
            
            # More precisely: for each existing edge (i,j) with color c,
            # we cannot have color(new,i)=c AND color(new,j)=c.
            
            # This is a constraint satisfaction problem. Let's use a simple backtracking.
            # But 4^32 is too large. Instead, use a heuristic: try to find a valid
            # assignment by treating it as a graph coloring problem on the "constraint graph".
            
            # Alternative: use the fact that the base is highly structured.
            # Try to find a vertex in the "complement" structure that fits.
            
            # Let's try a simpler approach: for each possible "type" of new vertex,
            # check if it works. A type is determined by which color class each edge
            # belongs to. But that's still 4^n.
            
            # Instead, let's try: pick a random subset of existing vertices and try to
            # add a new vertex connected to only those, then see if we can extend.
            
            # Actually, let's try a different approach: use the known structure.
            # The 32-vertex coloring is two copies of the 16-vertex Clebsch-based coloring.
            # Maybe we can add a third copy with appropriate cross-edges.
            
            # For now, let's try a simple greedy: for the new vertex, assign each edge
            # color to minimize the number of potential mono triangles.
            
            # Count for each color c, how many pairs (i,j) would form a mono triangle
            # if we set color(new,i)=c and color(new,j)=c.
            # Actually, for each pair (i,j), if color(i,j)=c, then we can't have both
            # color(new,i)=c and color(new,j)=c.
            
            # This is equivalent to: the set of vertices i where color(new,i)=c must be
            # an independent set in the graph formed by edges of color c.
            
            # So we need to partition the n existing vertices into 4 sets S0,S1,S2,S3
            # such that each Sk is an independent set in the color-k graph.
            # This is a 4-coloring problem on a 4-partite-like structure, which is hard.
            
            # Let's try a heuristic: for each existing vertex i, determine which colors
            # are "available" based on constraints from other vertices.
            
            # Simple greedy: process vertices in order, for each vertex i, pick the color
            # that doesn't create a mono triangle with any already-assigned vertex.
            
            assignment = [-1] * cur_n
            order = list(range(cur_n))
            
            # Sort by "constraint degree": vertices with more edges to other vertices
            # should be assigned first? Or last? Let's try last (most constrained first).
            # Actually, let's just try in order and see.
            
            for i in order:
                available = [0, 1, 2, 3]
                for j in range(cur_n):
                    if j == i or assignment[j] == -1:
                        continue
                    # If assignment[j] = c, then color(new,i) cannot be c if color(i,j)=c
                    # because that would form triangle (new, i, j) with all color c.
                    if color_index(cur_n, min(i,j), max(i,j)) < len(cur_colors):
                        c_ij = cur_colors[color_index(cur_n, min(i,j), max(i,j))]
                        if assignment[j] == c_ij:
                            if c_ij in available:
                                available.remove(c_ij)
                
                if not available:
                    break
                # Pick the color that appears least among already-assigned vertices
                # (to balance)
                counts = [0, 0, 0, 0]
                for j in range(cur_n):
                    if assignment[j] != -1:
                        counts[assignment[j]] += 1
                min_count = min(counts[c] for c in available)
                best_color = min(available, key=lambda c: counts[c])
                assignment[i] = best_color
            
            # Verify the assignment
            if all(a != -1 for a in assignment):
                # Check for mono triangles involving new vertex
                valid = True
                for i in range(cur_n):
                    for j in range(i+1, cur_n):
                        c_ij = cur_colors[color_index(cur_n, i, j)]
                        if assignment[i] == c_ij and assignment[j] == c_ij:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Add the new vertex
                    for i in range(cur_n):
                        cur_colors.append(assignment[i])
                    cur_n += 1
                    success = True
                    best_n = max(best_n, cur_n)
                    best_colors = cur_colors[:]
                    continue
            else:
                break
        
        return cur_n, cur_colors

    # Try extension
    n, colors = try_extend(colors, n, max_new=20)
    
    if n > best_n:
        best_n = n
        best_colors = colors[:]

    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
