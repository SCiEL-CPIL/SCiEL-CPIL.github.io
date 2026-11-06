"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools
import time

def construct():
    best_n = 1
    best_colors = []
    
    # Try to find largest n with valid 4-coloring
    # Use backtracking with forward checking and symmetry breaking
    
    def try_construct(n, max_time=110):
        """Try to find a valid 4-coloring of K_n. Returns colors list or None."""
        start_time = time.time()
        
        # edges in lexicographic order
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                edges.append((i, j))
        
        num_edges = len(edges)
        edge_to_idx = {}
        for idx, (i, j) in enumerate(edges):
            edge_to_idx[(i, j)] = idx
            edge_to_idx[(j, i)] = idx
        
        # For each vertex, store its incident edges
        vertex_edges = {v: [] for v in range(n)}
        for idx, (i, j) in enumerate(edges):
            vertex_edges[i].append((idx, j))
            vertex_edges[j].append((idx, i))
        
        colors = [None] * num_edges
        assigned = [False] * num_edges
        
        # Symmetry breaking: fix first vertex's edges to break color symmetry
        # Fix edge (0,1)=0, (0,2)=1, (0,3)=2 if n>=4
        def apply_symmetry_breaking():
            if n >= 2:
                idx = edge_to_idx[(0,1)]
                colors[idx] = 0
                assigned[idx] = True
            if n >= 3:
                idx = edge_to_idx[(0,2)]
                colors[idx] = 1
                assigned[idx] = True
            if n >= 4:
                idx = edge_to_idx[(0,3)]
                colors[idx] = 2
                assigned[idx] = True
        
        apply_symmetry_breaking()
        
        # Check if a triangle (a,b,c) would be monochromatic if we assign color c to edge (a,b)
        # given current assignments
        def would_create_mono_triangle(a, b, c, color):
            """Check if assigning 'color' to edge (a,b) creates a monochromatic triangle."""
            for v in range(n):
                if v == a or v == b:
                    continue
                # Check edges (a,v) and (b,v)
                idx_av = edge_to_idx[(min(a,v), max(a,v))]
                idx_bv = edge_to_idx[(min(b,v), max(b,v))]
                
                if assigned[idx_av] and assigned[idx_bv]:
                    if colors[idx_av] == color and colors[idx_bv] == color:
                        return True
                elif assigned[idx_av]:
                    if colors[idx_av] == color:
                        # Edge (b,v) is unassigned, but if it's forced to be 'color', triangle forms
                        # We'll handle this in forward checking
                        pass
                elif assigned[idx_bv]:
                    if colors[idx_bv] == color:
                        pass
            return False
        
        # Forward checking: for each unassigned edge, determine which colors are still possible
        def get_possible_colors(idx, a, b):
            possible = set([0, 1, 2, 3])
            for v in range(n):
                if v == a or v == b:
                    continue
                idx_av = edge_to_idx[(min(a,v), max(a,v))]
                idx_bv = edge_to_idx[(min(b,v), max(b,v))]
                
                if assigned[idx_av] and assigned[idx_bv]:
                    # If both are assigned and same color, that color is forbidden for (a,b)
                    if colors[idx_av] == colors[idx_bv]:
                        possible.discard(colors[idx_av])
            return possible
        
        # Find next unassigned edge with minimum possible colors (MRV heuristic)
        def select_next_edge():
            best_idx = -1
            best_count = 5
            for idx in range(num_edges):
                if assigned[idx]:
                    continue
                a, b = edges[idx]
                possible = get_possible_colors(idx, a, b)
                if len(possible) < best_count:
                    best_count = len(possible)
                    best_idx = idx
                    if best_count == 0:
                        return idx
            return best_idx
        
        # Backtracking search
        def backtrack():
            if time.time() - start_time > max_time:
                return False
            
            idx = select_next_edge()
            if idx == -1:
                # All edges assigned
                return True
            
            a, b = edges[idx]
            possible = get_possible_colors(idx, a, b)
            
            # Randomize order for diversity
            color_list = list(possible)
            random.shuffle(color_list)
            
            for color in color_list:
                colors[idx] = color
                assigned[idx] = True
                
                # Check if this assignment creates a monochromatic triangle
                valid = True
                for v in range(n):
                    if v == a or v == b:
                        continue
                    idx_av = edge_to_idx[(min(a,v), max(a,v))]
                    idx_bv = edge_to_idx[(min(b,v), max(b,v))]
                    if assigned[idx_av] and assigned[idx_bv]:
                        if colors[idx_av] == color and colors[idx_bv] == color:
                            valid = False
                            break
                
                if valid:
                    if backtrack():
                        return True
                
                colors[idx] = None
                assigned[idx] = False
            
            return False
        
        if backtrack():
            return colors
        return None
    
    # Try from high n down to lower n
    for n in range(50, 15, -1):
        if time.time() - (time.time() - 115) > 110:
            break
        
        # Try multiple random restarts for each n
        found = None
        max_restarts = 5 if n > 35 else 10
        for restart in range(max_restarts):
            random.seed(restart * 12345 + n)
            result = try_construct(n, max_time=10)
            if result is not None:
                found = result
                break
        
        if found is not None:
            return {'n': n, 'colors': found}
    
    # Fallback: return the known n=32 construction
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
