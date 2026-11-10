# EVOLVE-BLOCK-START
import itertools
import sys
sys.setrecursionlimit(10000)

def construct():
    # Precompute edge indices
    # edge_index[(i, j)] = idx
    # edges[idx] = (i, j)
    edges = []
    edge_index = {}
    idx = 0
    for i in range(16):
        for j in range(i + 1, 16):
            edge_index[(i, j)] = idx
            edges.append((i, j))
            idx += 1
    
    # Precompute K4s for each edge
    # k4s_for_edge[e_idx] = list of tuples of 6 edge indices
    k4s_for_edge = [[] for _ in range(120)]
    
    # Generate all K4s
    all_k4s = []
    for quad in itertools.combinations(range(16), 4):
        a, b, c, d = quad
        e_ab = edge_index[(a, b)]
        e_ac = edge_index[(a, c)]
        e_ad = edge_index[(a, d)]
        e_bc = edge_index[(b, c)]
        e_bd = edge_index[(b, d)]
        e_cd = edge_index[(c, d)]
        
        k4_edges = (e_ab, e_ac, e_ad, e_bc, e_bd, e_cd)
        all_k4s.append(k4_edges)
        
        for e_idx in k4_edges:
            k4s_for_edge[e_idx].append(k4_edges)
            
    # Backtracking search
    # Try K = 13, 14, 15
    for K in [13, 14, 15]:
        colours = [-1] * 120
        # Symmetry breaking:
        # Edge 0 (0,1) must be color 0
        # We will use a helper to determine allowed colors
        
        def backtrack(e_idx):
            if e_idx == 120:
                return True
            
            i, j = edges[e_idx]
            
            # Determine allowed colors
            # To break symmetry, we track the max color used so far.
            # However, since we are filling in order, we can just check
            # if we are introducing a new color, it must be max_used + 1.
            
            # Find max color used in colours[0:e_idx]
            # This is O(N) per step, which is 120*120 = 14400 ops total, negligible.
            max_used = -1
            for c in colours[:e_idx]:
                if c > max_used:
                    max_used = c
            
            # Allowed colors: 0 .. max_used
            # Plus potentially max_used + 1 if max_used + 1 < K
            
            start_c = 0
            end_c = max_used + 1
            if max_used + 1 < K:
                end_c = max_used + 2 # exclusive
            
            # Additional symmetry breaking for the first few edges?
            # The max_used logic is sufficient to fix the color permutation symmetry.
            
            for c in range(start_c, end_c):
                colours[e_idx] = c
                
                # Check constraints for K4s containing e_idx
                valid = True
                for k4 in k4s_for_edge[e_idx]:
                    # Check if all edges in this K4 are assigned
                    # e_idx is assigned now.
                    # Check the other 5.
                    all_assigned = True
                    distinct_colors = set()
                    for k in k4:
                        col = colours[k]
                        if col == -1:
                            all_assigned = False
                            break
                        distinct_colors.add(col)
                    
                    if all_assigned:
                        if len(distinct_colors) < 5:
                            valid = False
                            break
                
                if valid:
                    if backtrack(e_idx + 1):
                        return True
                
                colours[e_idx] = -1
            
            return False
        
        if backtrack(0):
            return colours
    
    # Fallback to seed if no solution found (should not happen)
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
# EVOLVE-BLOCK-END
