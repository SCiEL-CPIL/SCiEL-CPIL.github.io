import time
import random

N = 16
V = list(range(N))

# Pre-generate all pairs of remaining vertices for each edge
# edge_index -> list of (x, y, edge_vx, edge_xy, edge_yu) for cycle 1
#                and (x, y, edge_vy, edge_yx, edge_xu) for cycle 2
# Actually, we just need to check:
# Cycle 1: (u,v), (v,x), (x,y), (y,u)
# Cycle 2: (u,v), (v,y), (y,x), (x,u)

# To optimize, we precompute for each edge (u,v) the list of tuples needed for check.
# But (u,v) is dynamic in the sense that u,v are indices.
# We can just compute on the fly, or precompute a lookup.
# There are 120 edges.
# For each edge, 91 pairs.
# Total 120 * 91 * 2 cycles = 21840 checks.

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # 1. Initialize
    # Use the greedy split as a starting point to ensure a valid base
    # We'll re-implement the greedy logic quickly to get a valid start
    colors = [[0] * N for _ in range(N)]
    
    # Simple recursive Gallai split
    G = [0, 0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20]
    SPLIT = {4: 3, 5: 3, 6: 3, 7: 3, 8: 4, 9: 3, 10: 4, 11: 5, 12: 6, 13: 6, 14: 7, 15: 6, 16: 7}
    
    def solve(vertices, current_color):
        n = len(vertices)
        if n <= 1: return current_color
        if n == 2:
            u, v = vertices
            colors[u][v] = current_color
            colors[v][u] = current_color
            return current_color + 1
        if n == 3:
            u, v, w = vertices
            colors[u][v] = current_color
            colors[v][u] = current_color
            colors[v][w] = current_color + 1
            colors[w][v] = current_color + 1
            colors[u][w] = current_color + 2
            colors[w][u] = current_color + 2
            return current_color + 3
        
        k = SPLIT[n]
        v1 = vertices[:k]
        v2 = vertices[k:]
        cross_color = current_color
        for u in v1:
            for v in v2:
                colors[u][v] = cross_color
                colors[v][u] = cross_color
        next_color = current_color + 1
        next_color = solve(v1, next_color)
        next_color = solve(v2, next_color)
        return next_color
    
    solve(V, 0)
    
    # Count colors
    color_count = [0] * 120
    for i in range(N):
        for j in range(i+1, N):
            c = colors[i][j]
            color_count[c] += 1
    current_distinct = sum(1 for c in color_count if c > 0)
    
    # Pre-compute the check dependencies
    # For each edge (u,v), store a list of "checks"
    # A check is a tuple of 3 OTHER edges involved in the 4-cycle with (u,v)
    # Cycle: u-v-x-y-u. Others: (v,x), (x,y), (y,u)
    # We need to check if colors[(u,v)], colors[(v,x)], colors[(x,y)], colors[(y,u)] are all distinct.
    
    edge_checks = {}
    for i in range(N):
        for j in range(i+1, N):
            u, v = i, j
            others = [w for w in V if w != u and w != v]
            checks_list = []
            # Iterate all pairs x,y from others
            for a in range(len(others)):
                for b in range(a+1, len(others)):
                    x, y = others[a], others[b]
                    # Cycle 1: u-v-x-y-u
                    # Edges: (u,v) [current], (v,x), (x,y), (y,u)
                    # We store the other 3 edges
                    checks_list.append(((v, x), (min(x, y), max(x, y)), (min(y, u), max(y, u))))
                    # Cycle 2: u-v-y-x-u
                    # Edges: (u,v) [current], (v,y), (y,x), (x,u)
                    checks_list.append(((v, y), (min(y, x), max(y, x)), (min(x, u), max(x, u))))
            edge_checks[(i, j)] = checks_list
            edge_checks[(j, i)] = checks_list # Same structure, just for lookup if needed, but we always use (min, max) for edge id? 
            # Wait, colors[u][v] is symmetric.
            # The key for edge_checks should be canonical (min, max).
            # But the check logic uses specific orientations.
            # It's safer to keep (i,j) and (j,i) pointing to the same list or generate correctly.
            # Since colors[u][v] == colors[v][u], the color lookup is fine.
            # The cycle definition is geometric.
            
    # Helper to check validity of a specific edge change
    # We assume colors are updated except for the edge being changed?
    # No, we update colors, check, then revert if needed.
    
    def is_valid_after_change(u, v, new_color):
        # Get the checks for this edge
        checks = edge_checks[(u, v)] if u < v else edge_checks[(v, u)]
        # Actually, we stored for (i,j) with i<j.
        key = (min(u,v), max(u,v))
        checks = edge_checks[key]
        
        for e1, e2, e3 in checks:
            c1 = colors[u][v] # This is still OLD value! 
            # We want to check if changing u,v to new_color creates a rainbow cycle.
            # So we use new_color for c_uv.
            c_uv = new_color
            c_e1 = colors[e1[0]][e1[1]]
            c_e2 = colors[e2[0]][e2[1]]
            c_e3 = colors[e3[0]][e3[1]]
            
            # Check if 4 colors are distinct
            if len({c_uv, c_e1, c_e2, c_e3}) == 4:
                return False
        return True

    # Local Search
    best_distinct = current_distinct
    best_colors = [row[:] for row in colors]
    
    t = 1.0
    min_t = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick random edge
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i == j: continue
        if i > j: i, j = j, i
        
        old_color = colors[i][j]
        
        # Pick new color
        # Bias towards used colors + some new
        if random.random() < 0.7:
            # Pick from used colors
            new_color = random.randint(0, 119)
            # Optimization: pick from colors present in neighbors?
            # Just random is fine.
        else:
            new_color = random.randint(0, 119)
            
        if new_color == old_color:
            continue
            
        # Update counts
        delta_distinct = 0
        if color_count[old_color] == 1:
            delta_distinct -= 1
        if color_count[new_color] == 0:
            delta_distinct += 1
            
        color_count[old_color] -= 1
        color_count[new_color] += 1
        colors[i][j] = new_color
        colors[j][i] = new_color
        
        # Check validity
        valid = is_valid_after_change(i, j, new_color)
        
        if valid:
            new_distinct = current_distinct + delta_distinct
            if new_distinct > current_distinct:
                # Always accept improvement
                current_distinct = new_distinct
                if new_distinct > best_distinct:
                    best_distinct = new_distinct
                    best_colors = [row[:] for row in colors]
            else:
                # Accept with probability depending on temperature/delta
                # Since delta_distinct <= 0, we accept to allow exploration
                # But we shouldn't accept if it degrades too much?
                # If delta_distinct == 0, accept with prob 0.5
                # If delta_distinct < 0, reject (or accept with very low prob)
                if delta_distinct == 0:
                    if random.random() < 0.5:
                        current_distinct = new_distinct
                    else:
                        # Revert
                        colors[i][j] = old_color
                        colors[j][i] = old_color
                        color_count[old_color] += 1
                        color_count[new_color] -= 1
                else:
                    # Revert degradation
                    colors[i][j] = old_color
                    colors[j][i] = old_color
                    color_count[old_color] += 1
                    color_count[new_color] -= 1
        else:
            # Invalid, revert
            colors[i][j] = old_color
            colors[j][i] = old_color
            color_count[old_color] += 1
            color_count[new_color] -= 1
            
        t *= cooling

    # Ensure best_colors is used if current is not valid? 
    # We only accept valid moves, so current is always valid.
    # But maybe we want to return the one with max distinct?
    # current_distinct is tracked. best_distinct is tracked.
    # They should be similar. Return best.
    
    # Flatten best_colors
    result = []
    for i in range(N):
        for j in range(i+1, N):
            result.append(best_colors[i][j])
            
    return result
