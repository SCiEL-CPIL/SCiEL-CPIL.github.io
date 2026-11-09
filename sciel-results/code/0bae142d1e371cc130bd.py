import time
import random

def construct():
    deadline = time.time() + 110
    
    # GF(16) construction for 32 vertices
    # Polynomial x^4 + x + 1
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

    def get_color(i, j):
        if i // 16 == j // 16:
            return log[(i % 16) ^ (j % 16)] % 3
        else:
            return 3

    n0 = 32
    colors0 = []
    for i in range(n0):
        for j in range(i + 1, n0):
            colors0.append(get_color(i, j))

    def build_matrix(colors, n):
        mat = [[-1] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                mat[i][j] = mat[j][i] = colors[idx]
                idx += 1
        return mat

    def matrix_to_colors(mat, n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                colors.append(mat[i][j])
        return colors

    mat0 = build_matrix(colors0, n0)
    best_n = n0
    best_mat = [row[:] for row in mat0]

    def try_add_vertex_fast(mat, n):
        """
        Attempt to add a vertex to the graph defined by mat (size n).
        Returns a list of colors for the new vertex (length n) or None.
        Uses bitmasks for speed.
        """
        # domains[i] is a bitmask of valid colors for vertex i (0..n-1)
        # Initially all 4 colors (15)
        domains = [15] * n
        
        # Precompute conflict masks
        # For each vertex i, and color c, which vertices j cannot have color c if i has c?
        # Actually, if i has color c, then for any j where mat[i][j] == c, j cannot have color c.
        # We need to propagate this.
        
        # We are assigning colors to 0..n-1 such that no monochromatic triangle exists.
        # The new vertex is n.
        # We are solving for the colors of 0..n-1? No, 0..n-1 are fixed.
        # Wait, the seed program tries to ADD a vertex.
        # So 0..n-1 are fixed. We need to find colors for the NEW vertex n?
        # No, the new vertex connects to 0..n-1.
        # The colors of edges (n, i) are the variables.
        # Let x_i be the color of edge (n, i).
        # Constraint: For any i, j, if x_i == x_j == mat[i][j], then triangle (n, i, j) is monochromatic.
        # So, for each color c, the set of vertices i where x_i == c must be an independent set in the subgraph of G_c.
        # G_c is the graph of edges with color c in the existing n vertices.
        
        # So we need to assign x_0..x_{n-1} in {0,1,2,3} such that
        # for each c, {i | x_i == c} is an independent set in G_c.
        
        # This is a 4-SAT / Coloring problem on the "conflict" hypergraph?
        # No, it's simpler.
        # For each c, let S_c be the set of vertices assigned color c.
        # S_c must be independent in G_c.
        
        # We can use backtracking.
        # Order vertices by degree in the union of G_c?
        
        assign = [-1] * n
        
        # Precompute for each vertex i and color c, the set of neighbors in G_c.
        # neigh[i][c] is a bitmask of j such that mat[i][j] == c.
        neigh = [[0]*4 for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                c = mat[i][j]
                if c >= 0:
                    neigh[i][c] |= (1 << j)
        
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == n:
                return True
            
            i = pos
            # Try colors 0,1,2,3
            # Optimization: check if color c is possible
            # If we assign i color c, then no neighbor j of i in G_c can have color c.
            # So for all j in neigh[i][c], x_j != c.
            
            # We need to check if any neighbor j already has color c.
            # We maintain a bitmask 'used' for each color?
            # No, we just check assign[j].
            
            for c in [0, 1, 2, 3]:
                valid = True
                # Check against already assigned neighbors
                # If j is assigned and assign[j] == c and mat[i][j] == c, conflict.
                # We can precompute for each i and c, the list of neighbors.
                # But checking assign[j] is O(deg).
                
                # Better: maintain domains.
                # But for a single vertex addition, n is ~30-50.
                # Simple backtracking might be enough if ordered well.
                
                # Check validity
                for j in range(n):
                    if assign[j] == c and mat[i][j] == c:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                assign[i] = c
                if backtrack(pos + 1):
                    return True
                assign[i] = -1
                
            return False
        
        # Order by degree (highest first) to constrain early
        order = sorted(range(n), key=lambda x: -sum(1 for j in range(n) if mat[x][j] >= 0))
        
        # We need to map order index to actual vertex index
        # Modify backtrack to use order
        def backtrack_order(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                return True
            
            i = order[idx]
            for c in [0, 1, 2, 3]:
                valid = True
                for j in range(n):
                    if assign[j] == c and mat[i][j] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                assign[i] = c
                if backtrack_order(idx + 1):
                    return True
                assign[i] = -1
            return False

        if backtrack_order(0):
            return assign
        return None

    # Try to extend to 50
    for target in range(33, 52):
        if time.time() > deadline - 5:
            break
        
        current_n = best_n
        current_mat = [row[:] for row in best_mat]
        
        # Try to add one vertex
        assignment = try_add_vertex_fast(current_mat, current_n)
        
        if assignment is None:
            # Try random restarts or different orderings?
            # For now, if it fails, we stop.
            # To improve, we could try multiple random shuffles of the vertex order.
            break
        
        new_n = current_n + 1
        new_mat = [row[:] for row in current_mat]
        for i in range(current_n):
            new_mat[i].append(assignment[i])
        new_mat.append([-1] * current_n)
        for i in range(current_n):
            new_mat[current_n][i] = assignment[i]
        current_mat = new_mat
        current_n = new_n
        
        best_n = current_n
        best_mat = [row[:] for row in current_mat]

    best_colors = matrix_to_colors(best_mat, best_n)
    return {'n': best_n, 'colors': best_colors}
