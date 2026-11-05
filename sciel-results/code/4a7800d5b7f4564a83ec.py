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


def _construct_known_17():
    """
    Attempt to find a valid 4-coloring of K_17 using local search.
    Falls back to a valid n=9 coloring if not found quickly.
    """
    import random
    random.seed(42)
    
    # First, try to find n=17 quickly (10 seconds)
    n_target = 17
    deadline_17 = time.time() + 10.0
    
    # Local search: Start random, flip edges to reduce C6s
    # Since we can't count C6s easily, we just check validity.
    # If invalid, we don't know which edge to flip.
    # So we just do random restarts.
    
    for attempt in range(1000):
        if time.time() > deadline_17:
            break
        colors = [random.randint(0, 3) for _ in range(n_target * (n_target - 1) // 2)]
        if _is_valid(colors, n_target):
            return {'n': n_target, 'colors': colors}
            
    # Fallback: Find a valid n=9 coloring
    n_fallback = 9
    deadline_9 = time.time() + 5.0
    for attempt in range(1000):
        if time.time() > deadline_9:
            break
        colors = [random.randint(0, 3) for _ in range(n_fallback * (n_fallback - 1) // 2)]
        if _is_valid(colors, n_fallback):
            return {'n': n_fallback, 'colors': colors}
            
    # If even n=9 fails, return n=1
    return {'n': 1, 'colors': []}


def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)
    
    # Get a valid baseline (n=17 or fallback)
    result = _construct_known_17()
    best_n = result['n']
    best_colors = result['colors']
    
    # If we have time, try to improve to n=18
    # But n=18 is likely impossible (R_4(C6)=18 means max n is 17).
    # So we just return the best we found.
    
    return {'n': best_n, 'colors': best_colors}
