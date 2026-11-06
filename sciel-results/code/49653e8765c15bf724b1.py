"""Construct a 4-colouring of a complete graph without monochromatic triangles."""
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Attempt to construct a large 4-colouring of K_n with no monochromatic triangles.
    Strategy:
    1. Start with a known 32-vertex construction (Greenwood-Gleason K16 doubled).
    2. Try to extend to 50 vertices by adding vertices one by one (or in batches),
       using a randomized local search to assign colors to new edges while avoiding
       monochromatic triangles.
    """
    
    # Known 32-vertex construction
    def _mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    _log, _x = {}, 1
    for e in range(15):
        _log[_x] = e
        _x = _mul(_x, 2)
    
    def _base_color(i, j):
        """Color for edges within the 32-vertex base construction."""
        if i // 16 == j // 16:
            return _log[(i % 16) ^ (j % 16)] % 3
        else:
            return 3
    
    # Start with the 32-vertex base
    n_base = 32
    # adj[i][j] = color for edge (i,j), -1 if not yet assigned
    adj = [[-1] * (n_base) for _ in range(n_base)]
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = _base_color(i, j)
            adj[i][j] = c
            adj[j][i] = c
    
    # Function to check if adding edge (u, v) with color c creates a mono-triangle
    def _creates_triangle(u, v, c):
        """Check if edge (u,v) with color c creates a monochromatic triangle."""
        # For each vertex w, check if (u,w) and (v,w) both have color c
        for w in range(len(adj)):
            if w == u or w == v:
                continue
            cuw = adj[u][w]
            cvw = adj[v][w]
            if cuw == c and cvw == c:
                return True
        return False
    
    # Try to extend from 32 to up to 50 vertices
    target_max = 50
    rng = random.Random(42)
    
    current_n = n_base
    
    # We'll try to add vertices one by one
    # For each new vertex, we need to assign colors to all edges from it to existing vertices
    # such that no monochromatic triangle is formed
    
    def _try_add_vertex(adj, current_n):
        """
        Try to add a new vertex (index current_n) to the coloring.
        Returns True if successful, False otherwise.
        Modifies adj in place if successful.
        """
        new_idx = current_n
        size = current_n + 1
        # Expand adj
        new_row = [-1] * size
        adj.append(new_row)
        for i in range(current_n):
            adj[i].append(-1)
        
        # For each existing vertex i, assign a color to edge (i, new_idx)
        # such that no mono-triangle is formed
        # We use backtracking with randomized ordering
        
        # For efficiency, we'll assign colors in a greedy+random way
        # For each vertex i, the possible colors are those c where 
        # for all j < i (already assigned), if adj[i][j] == c then adj[j][new_idx] != c
        # Actually, let's think more carefully.
        
        # When we assign color c to edge (i, new_idx), we need:
        # - For all j where adj[i][j] == c, we need adj[j][new_idx] != c
        #   (otherwise triangle (i, j, new_idx) would be mono-c)
        # - For all j where adj[j][new_idx] == c (already assigned), we need adj[i][j] != c
        #   (otherwise triangle (i, j, new_idx) would be mono-c)
        
        # We'll assign in order i = 0, 1, ..., current_n - 1
        # For each i, try each color 0-3, pick one that doesn't conflict
        
        def _backtrack(i, assignments):
            if i == current_n:
                return True
            
            # Try colors in random order
            colors = [0, 1, 2, 3]
            rng.shuffle(colors)
            
            for c in colors:
                # Check if assigning adj[i][new_idx] = c is valid
                valid = True
                for j in range(i):
                    # j is already assigned
                    if adj[i][j] == c and adj[j][new_idx] == c:
                        valid = False
                        break
                if not valid:
                    continue
                # Also check against already assigned vertices j < i
                # This is already covered above since we check all j < i
                
                # Assign
                adj[i][new_idx] = c
                adj[new_idx][i] = c
                
                if _backtrack(i + 1, assignments):
                    return True
                
                # Undo
                adj[i][new_idx] = -1
                adj[new_idx][i] = -1
            
            return False
        
        # Try multiple times with different random seeds if first fails
        for attempt in range(20):
            rng_local = random.Random(attempt)
            # We need to use rng_local for shuffling, but our _backtrack uses rng
            # Let's just try and see
            if _backtrack(0, None):
                return True
            # If failed, reset
            for i in range(current_n):
                adj[i][new_idx] = -1
                adj[new_idx][i] = -1
        
        # Failed to add
        # Remove the new vertex
        adj.pop()
        for i in range(current_n):
            adj[i].pop()
        return False
    
    # Try to extend up to target_max
    for target_n in range(n_base + 1, target_max + 1):
        success = _try_add_vertex(adj, current_n)
        if success:
            current_n = target_n
        else:
            break
    
    # Now build the final color list in lexicographic order
    n = current_n
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(adj[i][j])
    
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
