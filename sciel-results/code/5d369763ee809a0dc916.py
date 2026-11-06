import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    # We aim to build a 5-coloring of K_n with no monochromatic C4.
    # Start with known small constructions and try to extend.
    # The Lazebnik-Woldar construction gives n=26, but we try to find something better
    # or at least match the seed's n=11 while trying to improve.
    
    # First, let's try a greedy/backtracking approach to extend a coloring.
    # We'll represent the coloring as a list of colors for each edge (i,j) with i<j.
    # For efficiency, we'll use a dictionary or array indexed by (i,j).
    
    # Let's start with a known valid small coloring and try to extend it.
    # For n=11, the seed uses a cyclic construction. Let's verify it's valid and try to extend.
    
    def make_colors_from_dict(color_dict, n):
        """Convert a dict {(i,j): color} to the required list format."""
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                colors.append(color_dict.get((i,j), 0))
        return colors
    
    def has_c4_in_color(color, n, color_dict):
        """Check if a specific color class contains a C4."""
        # For each pair of vertices (u,v), count common neighbors in this color.
        # If any pair has >= 2 common neighbors, there's a C4.
        # Actually, a C4 exists if there are two vertices with two common neighbors.
        # More precisely: for each color c, build adjacency, then check for C4.
        
        # For efficiency, let's check: for each pair (u,v), count common neighbors in color c.
        # If count >= 2, then (u,v) plus those two neighbors form a C4 (if u,v are not adjacent in c, 
        # but actually C4 is a cycle of length 4, so we need u-w-x-v-u where all edges are in color c).
        # A C4 in color c means there exist distinct u,v,w,x such that edges (u,w),(w,x),(x,v),(v,u) are all in color c.
        # This is equivalent to: there exist two vertices u,v with at least two common neighbors in color c.
        
        # Build adjacency for this color
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if color_dict.get((i,j)) == color:
                    adj[i].add(j)
                    adj[j].add(i)
        
        for u in range(n):
            for v in range(u+1, n):
                common = adj[u] & adj[v]
                if len(common) >= 2:
                    return True
        return False
    
    def is_valid(color_dict, n):
        """Check if the coloring is valid (no monochromatic C4)."""
        for c in range(5):
            if has_c4_in_color(c, n, color_dict):
                return False
        return True
    
    # Start with the seed construction for n=11
    def seed_coloring(n):
        color_dict = {}
        for i in range(n):
            for j in range(i+1, n):
                d = min(j - i, n - j + i)
                color_dict[(i,j)] = d - 1
        return color_dict
    
    # Try to extend from n=11 to larger n
    best_n = 11
    best_color_dict = seed_coloring(11)
    
    # Verify seed is valid
    if is_valid(best_color_dict, 11):
        best_n = 11
    else:
        # Fallback to a trivial valid coloring
        best_n = 1
        best_color_dict = {}
    
    # Try to extend by adding vertices one at a time
    current_n = best_n
    current_dict = dict(best_color_dict)
    
    while current_n < 29:
        if time.time() > deadline - 5.0:
            break
        
        new_n = current_n + 1
        new_vertex = current_n
        success = False
        
        # Try to assign colors to edges from new_vertex to existing vertices
        # Use backtracking
        def backtrack(idx, assignments):
            if time.time() > deadline - 3.0:
                return None
            if idx == new_vertex:
                # All edges assigned, check validity
                test_dict = dict(current_dict)
                for (i, j), c in assignments:
                    test_dict[(i,j)] = c
                if is_valid(test_dict, new_n):
                    return dict(assignments)
                return None
            
            # Try all 5 colors for edge (new_vertex, idx)
            for c in range(5):
                assignments[(new_vertex, idx)] = c
                result = backtrack(idx + 1, assignments)
                if result is not None:
                    return result
                del assignments[(new_vertex, idx)]
            return None
        
        result = backtrack(0, {})
        if result is not None:
            current_dict.update(result)
            current_n = new_n
            best_n = current_n
            best_color_dict = dict(current_dict)
        else:
            break
    
    # Convert to required format
    colors = make_colors_from_dict(best_color_dict, best_n)
    
    return {'n': best_n, 'colors': colors}
