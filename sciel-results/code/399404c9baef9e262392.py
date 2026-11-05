"""Rewritten program for R_4(C6): maximize n with a 4-colouring of K_n with no monochromatic C6."""
import time
import itertools
from collections import defaultdict

DEADLINE_MARGIN = 5.0  # seconds before the 150s wall-clock kill to stop

def _has_c6_bitset(adj, n):
    """
    Check if the graph defined by adjacency masks 'adj' contains a 6-cycle.
    adj[i] is an integer bitmask where bit j is 1 if edge (i,j) exists.
    n is the number of vertices.
    """
    for v0 in range(n):
        # v1 in N(v0)
        mask_v1 = adj[v0]
        while mask_v1:
            v1 = (mask_v1 & -mask_v1).bit_length() - 1
            mask_v1 &= mask_v1 - 1
            
            # v2 in N(v1) \ {v0}
            mask_v2 = adj[v1] & ~(1 << v0)
            while mask_v2:
                v2 = (mask_v2 & -mask_v2).bit_length() - 1
                mask_v2 &= mask_v2 - 1
                
                # v3 in N(v2) \ {v0, v1}
                mask_v3 = adj[v2] & ~(1 << v0) & ~(1 << v1)
                while mask_v3:
                    v3 = (mask_v3 & -mask_v3).bit_length() - 1
                    mask_v3 &= mask_v3 - 1
                    
                    # v4 in N(v3) \ {v0, v1, v2}
                    mask_v4 = adj[v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                    while mask_v4:
                        v4 = (mask_v4 & -mask_v4).bit_length() - 1
                        mask_v4 &= mask_v4 - 1
                        
                        # Check if there exists v5 such that:
                        # v5 in N(v4) AND v5 in N(v0)
                        # v5 not in {v0, v1, v2, v3, v4}
                        
                        # Candidates for v5
                        mask_v5 = adj[v4] & adj[v0]
                        # Exclude used vertices
                        used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                        mask_v5 &= ~used_mask
                        
                        if mask_v5:
                            return True
    return False


def _is_valid(colors, n):
    """Check if the coloring is valid (no mono C6)."""
    adj = [[0] * n for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
            idx += 1
            
    for c in range(4):
        if _has_c6_bitset(adj[c], n):
            return False
    return True


def _find_valid_coloring(n, time_limit=10.0):
    """
    Find a valid 4-coloring of K_n using local search with C6-counting heuristic.
    """
    import random
    random.seed(42)
    
    deadline = time.time() + time_limit
    num_edges = n * (n - 1) // 2
    
    def count_c6s(colors):
        """Count total number of monochromatic C6s in the coloring."""
        adj = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
                idx += 1
        
        total = 0
        for c in range(4):
            # Count C6s in color class c
            for v0 in range(n):
                mask_v1 = adj[c][v0]
                while mask_v1:
                    v1 = (mask_v1 & -mask_v1).bit_length() - 1
                    mask_v1 &= mask_v1 - 1
                    
                    mask_v2 = adj[c][v1] & ~(1 << v0)
                    while mask_v2:
                        v2 = (mask_v2 & -mask_v2).bit_length() - 1
                        mask_v2 &= mask_v2 - 1
                        
                        mask_v3 = adj[c][v2] & ~(1 << v0) & ~(1 << v1)
                        while mask_v3:
                            v3 = (mask_v3 & -mask_v3).bit_length() - 1
                            mask_v3 &= mask_v3 - 1
                            
                            mask_v4 = adj[c][v3] & ~(1 << v0) & ~(1 << v1) & ~(1 << v2)
                            while mask_v4:
                                v4 = (mask_v4 & -mask_v4).bit_length() - 1
                                mask_v4 &= mask_v4 - 1
                                
                                mask_v5 = adj[c][v4] & adj[c][v0]
                                used_mask = (1 << v0) | (1 << v1) | (1 << v2) | (1 << v3) | (1 << v4)
                                mask_v5 &= ~used_mask
                                if mask_v5:
                                    total += 1
        return total
    
    # Start with random coloring
    colors = [random.randint(0, 3) for _ in range(num_edges)]
    current_c6s = count_c6s(colors)
    
    if current_c6s == 0:
        return colors
    
    best_colors = colors[:]
    best_c6s = current_c6s
    
    # Local search: flip random edges, keep if it reduces C6 count
    for iteration in range(100000):
        if time.time() > deadline:
            break
        if best_c6s == 0:
            return best_colors
            
        # Pick a random edge to flip
        edge_idx = random.randint(0, num_edges - 1)
        old_color = colors[edge_idx]
        new_color = (old_color + 1 + random.randint(0, 2)) % 4
        
        colors[edge_idx] = new_color
        new_c6s = count_c6s(colors)
        
        if new_c6s <= best_c6s:
            best_c6s = new_c6s
            best_colors = colors[:]
        else:
            colors[edge_idx] = old_color
    
    if best_c6s == 0:
        return best_colors
    return None


def _construct_known_17():
    """
    Find a valid 4-coloring of K_17 using local search.
    Falls back to smaller n if not found.
    """
    import random
    random.seed(42)
    
    # Try n=17 first with local search
    result = _find_valid_coloring(17, time_limit=15.0)
    if result is not None:
        return {'n': 17, 'colors': result}
    
    # Try n=16
    result = _find_valid_coloring(16, time_limit=10.0)
    if result is not None:
        return {'n': 16, 'colors': result}
    
    # Try n=15
    result = _find_valid_coloring(15, time_limit=10.0)
    if result is not None:
        return {'n': 15, 'colors': result}
    
    # Try n=14
    result = _find_valid_coloring(14, time_limit=10.0)
    if result is not None:
        return {'n': 14, 'colors': result}
    
    # Try n=13
    result = _find_valid_coloring(13, time_limit=10.0)
    if result is not None:
        return {'n': 13, 'colors': result}
    
    # Try n=12
    result = _find_valid_coloring(12, time_limit=10.0)
    if result is not None:
        return {'n': 12, 'colors': result}
    
    # Try n=11
    result = _find_valid_coloring(11, time_limit=10.0)
    if result is not None:
        return {'n': 11, 'colors': result}
    
    # Try n=10
    result = _find_valid_coloring(10, time_limit=10.0)
    if result is not None:
        return {'n': 10, 'colors': result}
    
    # Try n=9
    result = _find_valid_coloring(9, time_limit=5.0)
    if result is not None:
        return {'n': 9, 'colors': result}
    
    # Try n=8
    result = _find_valid_coloring(8, time_limit=5.0)
    if result is not None:
        return {'n': 8, 'colors': result}
    
    # Try n=7
    result = _find_valid_coloring(7, time_limit=5.0)
    if result is not None:
        return {'n': 7, 'colors': result}
    
    # Try n=6
    result = _find_valid_coloring(6, time_limit=5.0)
    if result is not None:
        return {'n': 6, 'colors': result}
    
    # Try n=5
    result = _find_valid_coloring(5, time_limit=2.0)
    if result is not None:
        return {'n': 5, 'colors': result}
    
    # n=4 always works
    colors = [random.randint(0, 3) for _ in range(6)]
    return {'n': 4, 'colors': colors}


def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)
    
    # Get a valid baseline
    result = _construct_known_17()
    best_n = result['n']
    best_colors = result['colors']
    
    return {'n': best_n, 'colors': best_colors}
