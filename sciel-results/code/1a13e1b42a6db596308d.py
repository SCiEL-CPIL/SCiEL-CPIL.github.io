"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    random.seed(42)
    
    # Base: Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16)
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
    
    # Build K32: two copies of K16 with colour 3 between them
    def build_k32_colors():
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return colors
    
    def build_k16_colors():
        colors = []
        for i in range(16):
            for j in range(i + 1, 16):
                colors.append(log[i ^ j] % 3)
        return colors
    
    best_n = 32
    best_colors = build_k32_colors()
    
    # Try to extend K32 to K33 by adding vertex 32
    # Vertex 32 connects to vertices 0-15 (copy A) and 16-31 (copy B)
    # For no monochromatic triangle:
    # - For each colour c, the set of vertices in A coloured c to 32 must be independent in A's c-graph
    # - Similarly for B
    # - Edges between A and B are all colour 3, so if 32 has colour 3 to some a in A and some b in B,
    #   triangle (32, a, b) would be monochromatic colour 3. So 32 cannot have colour 3 to both A and B.
    
    # Strategy: try adding vertices one by one, checking validity
    # Use local search to find valid extensions
    
    def check_no_triangle(colors, n):
        """Check if any colour class contains a triangle. Returns True if valid."""
        # For each pair (i,j) with colour c, check if there exists k such that
        # (i,k) and (j,k) also have colour c
        # Build adjacency per colour
        adj = [[set() for _ in range(4)] for _ in range(n)]
        for idx in range(n * (n - 1) // 2):
            # Convert linear index to (i,j)
            # idx = i*(2n-1-i)//2 + (j-i-1)
            # Invert:
            i = 0
            while i < n - 1:
                count = n - 1 - i
                if idx < count:
                    j = i + 1 + idx
                    break
                idx -= count
                i += 1
            else:
                return True  # shouldn't happen
            c = colors[len(colors) - 1 - idx] if False else None  # need proper mapping
        # This is getting complex; let's use a different approach
        # Build list of edges per colour
        edges_by_color = [set() for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                edges_by_color[c].add((i, j))
                idx += 1
        
        for c in range(4):
            for (i, j) in edges_by_color[c]:
                # Check if there's k < i or k > j such that (i,k) and (j,k) in edges_by_color[c]
                # k < i: need (k,i) and (k,j) in edges_by_color[c]
                # k > j: need (i,k) and (j,k) in edges_by_color[c]
                for k in range(i):
                    if (k, i) in edges_by_color[c] and (k, j) in edges_by_color[c]:
                        return False
                for k in range(j + 1, n):
                    if (i, k) in edges_by_color[c] and (j, k) in edges_by_color[c]:
                        return False
        return True
    
    # Better: incremental check
    def is_valid_incremental(base_colors, base_n, new_vertex_edges):
        """
        base_colors: colours for K_base_n
        new_vertex_edges: list of colours for edges from new vertex (index base_n) to vertices 0..base_n-1
        Check if adding this vertex creates any monochromatic triangle.
        """
        n = base_n + 1
        # Triangles involving the new vertex (n-1):
        # For each pair (i,j) with i<j<n-1, if color(i,j) == color(i, n-1) == color(j, n-1), triangle
        # Build: for each colour c, which vertices are connected to n-1 with colour c
        # Then for each such pair (i,j) both connected to n-1 with colour c, check if (i,j) also has colour c
        for c in range(4):
            verts_c = [i for i in range(base_n) if new_vertex_edges[i] == c]
            for a in range(len(verts_c)):
                for b in range(a + 1, len(verts_c)):
                    i, j = verts_c[a], verts_c[b]
                    # Get colour of edge (i,j) in base
                    # Linear index for (i,j): sum_{k=0}^{i-1} (n-1-k) + (j-i-1) = i*(2*base_n-1-i)//2 + (j-i-1)
                    idx = i * (2 * base_n - 1 - i) // 2 + (j - i - 1)
                    if base_colors[idx] == c:
                        return False
        return True
    
    def get_edge_color(colors, n, i, j):
        """Get colour of edge (i,j) in a K_n colouring stored in lex order."""
        idx = i * (2 * n - 1 - i) // 2 + (j - i - 1)
        return colors[idx]
    
    # Try to extend K32 to larger n
    current_n = 32
    current_colors = best_colors[:]
    
    # Try adding vertex 32
    # For each possible assignment of colours to edges from 32 to 0..31, check validity
    # The constraint: for each c, the set S_c = {i : color(32,i)=c} must be such that
    # no two vertices in S_c are connected by colour c in the base.
    
    # Precompute for K32: for each colour c, which pairs are NOT connected by c
    # Actually, we need: for each c, the graph G_c on {0..31} with edges of colour c.
    # S_c must be an independent set in G_c.
    
    # Let's try a greedy/random approach:
    # For vertex 32, pick colours for edges to 0..31 such that constraints are met.
    
    # Precompute conflict sets
    # For each vertex i and colour c, which vertices j have color(i,j) == c?
    # Then S_c must not contain any pair (i,j) where color(i,j)==c.
    
    # Simpler: try random assignments and check
    max_target = 50
    improved = False
    
    for target_n in range(33, min(max_target + 1, 63)):
        if time.time() > deadline - 5:
            break
        # Try to add vertex (current_n)
        # Random restarts
        found = False
        for attempt in range(5000):
            if time.time() > deadline - 5:
                break
            # Randomly assign colours to edges from new vertex to existing
            new_edges = [random.randint(0, 3) for _ in range(current_n)]
            if is_valid_incremental(current_colors, current_n, new_edges):
                # Valid! Add the vertex
                current_colors.extend(new_edges)
                current_n += 1
                found = True
                improved = True
                break
        
        if not found:
            # Try local search: start from a random assignment and flip colours to reduce conflicts
            if time.time() > deadline - 10:
                break
            new_edges = [random.randint(0, 3) for _ in range(current_n)]
            for step in range(20000):
                if time.time() > deadline - 5:
                    break
                # Count conflicts
                conflicts = []
                for c in range(4):
                    verts_c = [i for i in range(current_n) if new_edges[i] == c]
                    for a in range(len(verts_c)):
                        for b in range(a + 1, len(verts_c)):
                            i, j = verts_c[a], verts_c[b]
                            idx = i * (2 * current_n - 1 - i) // 2 + (j - i - 1)
                            if current_colors[idx] == c:
                                conflicts.append((i, j, c))
                if not conflicts:
                    current_colors.extend(new_edges)
                    current_n += 1
                    found = True
                    improved = True
                    break
                # Pick a random conflict and flip one of the two vertices' colour
                i, j, c = random.choice(conflicts)
                if random.random() < 0.5:
                    new_edges[i] = (new_edges[i] + random.randint(1, 3)) % 4
                else:
                    new_edges[j] = (new_edges[j] + random.randint(1, 3)) % 4
            
            if found:
                break
    
    if current_n > best_n:
        best_n = current_n
        best_colors = current_colors
    
    return {'n': best_n, 'colors': best_colors}
# EVOLVE-BLOCK-END
