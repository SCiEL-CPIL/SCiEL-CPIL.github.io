"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _has_k222(adj_mask, n):
    """Check if any color class contains K2,2,2.
    adj_mask[c] is a list of bitmasks for each vertex in color class c.
    Returns True if K2,2,2 exists in either color class."""
    for c in (0, 1):
        masks = adj_mask[c]
        for i in range(n):
            for j in range(i+1, n):
                # Find common neighbors in color c
                common = masks[i] & masks[j]
                if common == 0:
                    continue
                # Check if there's a pair of common neighbors that also share 2 more common neighbors
                # K2,2,2: vertices {i,j} connected to {a,b} and {c,d} in color c
                # So we need: i,j both connected to a,b,c,d in color c
                # And a,b,c,d form K2,2 with i,j
                # Actually K2,2,2 means: partition into 3 sets of 2, all edges between sets.
                # So {i,j}, {a,b}, {c,d} where all 12 edges exist in color c.
                # i,j are in set 1. Need to find a,b,c,d such that:
                #   i,a i,b i,c i,d j,a j,b j,c j,d a,b a,c a,d b,c b,d c,d all in color c
                # This is equivalent to: in the subgraph induced by common neighbors of i,j,
                # find a K4 (clique of size 4) where those 4 vertices are all mutually connected in color c.
                # Wait, that's not quite right. Let me reconsider.
                
                # K2,2,2 is the complete tripartite graph with parts of size 2.
                # Vertices: A={a1,a2}, B={b1,b2}, C={c1,c2}
                # Edges: all edges between different parts, no edges within parts.
                # So for a fixed pair {i,j} to be one part, we need:
                #   4 other vertices {x1,x2,y1,y2} such that:
                #   - i and j are both connected to all of x1,x2,y1,y2 in color c
                #   - x1,x2 are NOT connected to each other in color c (same part)
                #   - y1,y2 are NOT connected to each other in color c (same part)
                #   - x1,x2 ARE connected to y1,y2 in color c (different parts)
                
                # So: common = N(i) ∩ N(j) in color c
                # From common, we need to find two disjoint non-edges (x1,x2) and (y1,y2)
                # such that all cross edges exist between {x1,x2} and {y1,y2} in color c.
                
                # This is expensive. Let me use a simpler check.
                # For efficiency, I'll check: does common contain a K4?
                # If common has a K4, then those 4 vertices plus i,j form K2,2,2? No.
                # If i,j connect to a,b,c,d and a,b,c,d form K4, then {i,j},{a,b},{c,d} is K2,2,2
                # only if a-b is NOT an edge (they're in the same part). But in K4, a-b IS an edge.
                # So that's wrong.
                
                # Let me think differently. The complement approach might be easier.
                # Actually, let me just do a direct check with bitmasks.
                
                # For each pair (x,y) in common that are NOT connected in color c:
                #   Check if there exist (u,v) in common, u≠x,y, v≠x,y, u≠v,
                #   such that u,v are not connected in color c,
                #   and all cross edges {x,y}×{u,v} exist in color c.
                
                # This is O(n^4) in the worst case which is too slow.
                
                # Alternative: precompute for each color c, for each vertex v, 
                # the set of non-neighbors. Then for common neighbors of i,j,
                # check if there's a matching of size 2 in the non-edge graph
                # where all cross edges exist.
                
                # Given time constraints, let me use a simpler heuristic check.
                pass
    return False

def _check_k222_colors(colors, n):
    """Check if either color class contains K2,2,2. Returns True if valid (no K2,2,2)."""
    # Build adjacency for each color
    adj = [[0] * n for _ in range(2)]
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            c = colors[idx]
            idx += 1
            adj[c][i] |= (1 << j)
            adj[c][j] |= (1 << i)
    
    # For each color, check for K2,2,2
    for c in (0, 1):
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[c][i] >> j) & 1:
                    continue  # i,j not connected in color c
                # Find common neighbors in color c
                common = adj[c][i] & adj[c][j]
                # Remove i and j from common (they shouldn't be there anyway)
                common &= ~(1 << i)
                common &= ~(1 << j)
                if common == 0:
                    continue
                # We need to find 4 vertices x1,x2,y1,y2 in common such that:
                # - x1-x2 is NOT an edge in color c
                # - y1-y2 is NOT an edge in color c
                # - All cross edges between {x1,x2} and {y1,y2} exist in color c
                # 
                # Iterate over pairs in common that are non-edges
                common_list = []
                mask = common
                while mask:
                    bit = mask & (-mask)
                    v = bit.bit_length() - 1
                    common_list.append(v)
                    mask ^= bit
                
                m = len(common_list)
                if m < 4:
                    continue
                
                # Find all non-edge pairs in common
                nonedge_pairs = []
                for a_idx in range(m):
                    a = common_list[a_idx]
                    for b_idx in range(a_idx+1, m):
                        b = common_list[b_idx]
                        if not (adj[c][a] >> b) & 1:
                            nonedge_pairs.append((a, b))
                
                if len(nonedge_pairs) < 2:
                    continue
                
                # Check if any two non-edge pairs are disjoint and have all cross edges
                for p1_idx in range(len(nonedge_pairs)):
                    x1, x2 = nonedge_pairs[p1_idx]
                    for p2_idx in range(p1_idx+1, len(nonedge_pairs)):
                        y1, y2 = nonedge_pairs[p2_idx]
                        # Check disjointness
                        if x1 == y1 or x1 == y2 or x2 == y1 or x2 == y2:
                            continue
                        # Check all cross edges exist in color c
                        if ((adj[c][x1] >> y1) & 1 and 
                            (adj[c][x1] >> y2) & 1 and
                            (adj[c][x2] >> y1) & 1 and
                            (adj[c][x2] >> y2) & 1):
                            return False  # Found K2,2,2
    return True  # No K2,2,2 found

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with Paley(29)
    p = 29
    squares = {x * x % p for x in range(1, p)}
    
    # Generate base colors
    base_colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                base_colors.append(0)
            else:
                base_colors.append(1)
    
    best_result = {'n': p, 'colors': base_colors[:]}
    
    # Try to extend to n=30 by adding a new vertex
    n_target = 30
    
    # For the new vertex 29, we need to choose colors for edges (29, j) for j=0..28
    # There are 2^29 possible colorings, so we need to search intelligently.
    
    # Strategy: use local search. Start with a random assignment for vertex 29's edges,
    # then iteratively flip edges to reduce K2,2,2 violations.
    
    def _count_violations(colors, n):
        """Count K2,2,2 violations (should be 0 for valid)."""
        if _check_k222_colors(colors, n):
            return 0
        return 1
    
    def _get_violation_count_fast(adj, n):
        """Fast violation count using precomputed adjacency."""
        for c in (0, 1):
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj[c][i] >> j) & 1:
                        continue
                    common = adj[c][i] & adj[c][j]
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    if common == 0:
                        continue
                    # Count how many K2,2,2 this pair participates in
                    # Simplified: just check if there are at least 4 common neighbors
                    # and if any non-edge pair structure exists
                    # For speed, just return 1 if any violation found
                    common_list = []
                    mask = common
                    while mask:
                        bit = mask & (-mask)
                        v = bit.bit_length() - 1
                        common_list.append(v)
                        mask ^= bit
                    m = len(common_list)
                    if m < 4:
                        continue
                    # Check for non-edge pairs
                    has_nonedge = False
                    for a_idx in range(m):
                        a = common_list[a_idx]
                        for b_idx in range(a_idx+1, m):
                            b = common_list[b_idx]
                            if not (adj[c][a] >> b) & 1:
                                has_nonedge = True
                                break
                        if has_nonedge:
                            break
                    if not has_nonedge:
                        continue
                    # Found potential issue, do full check
                    nonedge_pairs = []
                    for a_idx in range(m):
                        a = common_list[a_idx]
                        for b_idx in range(a_idx+1, m):
                            b = common_list[b_idx]
                            if not (adj[c][a] >> b) & 1:
                                nonedge_pairs.append((a, b))
                    for p1_idx in range(len(nonedge_pairs)):
                        x1, x2 = nonedge_pairs[p1_idx]
                        for p2_idx in range(p1_idx+1, len(nonedge_pairs)):
                            y1, y2 = nonedge_pairs[p2_idx]
                            if x1 == y1 or x1 == y2 or x2 == y1 or x2 == y2:
                                continue
                            if ((adj[c][x1] >> y1) & 1 and 
                                (adj[c][x1] >> y2) & 1 and
                                (adj[c][x2] >> y1) & 1 and
                                (adj[c][x2] >> y2) & 1):
                                return 1
        return 0
    
    # Local search for vertex 29's edges
    random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Random initial assignment for vertex 29
        new_edges = [random.randint(0, 1) for _ in range(29)]
        
        # Build full color list
        full_colors = base_colors + new_edges
        
        # Check validity
        if _check_k222_colors(full_colors, 30):
            return {'n': 30, 'colors': full_colors}
        
        # Local search: flip edges to try to eliminate violations
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Check current state
            if _check_k222_colors(full_colors, 30):
                return {'n': 30, 'colors': full_colors}
            
            # Try flipping a random edge
            flip_idx = random.randint(0, 28)
            old_val = new_edges[flip_idx]
            new_edges[flip_idx] = 1 - old_val
            full_colors = base_colors + new_edges
            
            if _check_k222_colors(full_colors, 30):
                return {'n': 30, 'colors': full_colors}
    
    # If we couldn't find a 30-vertex coloring, return the best we have
    return best_result
# EVOLVE-BLOCK-END
