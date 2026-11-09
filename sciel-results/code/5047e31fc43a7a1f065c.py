import random
import time

def construct():
    """
    Greedy construction with forbidden-pair counting for 5-colourings of K_n
    without monochromatic C4. Maximizes n up to 29.
    """
    max_n = 29
    num_colors = 5
    
    # Try to build from largest n down
    for n in range(max_n, 10, -1):
        result = _build_coloring(n, num_colors)
        if result is not None:
            return result
    
    # Fallback to known construction for n=11
    n = 11
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}

def _build_coloring(n, num_colors, max_attempts=50):
    """
    Attempt to build a valid 5-coloring of K_n without monochromatic C4.
    Uses forbidden-pair counting to guide color assignment.
    Returns the result dict if successful, else None.
    """
    start_time = time.time()
    timeout = 115  # Leave buffer for 120s limit
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
        
        # colors[i][j] = color of edge (i,j), -1 if unassigned
        colors = [[-1] * n for _ in range(n)]
        
        # For each color c, track for each pair (a,b) the count of common neighbors in color c.
        # common_count[c][a][b] = number of vertices w with colors[a][w]==c and colors[b][w]==c
        # We need this to always be < 2.
        common_count = [[[0]*n for _ in range(n)] for _ in range(num_colors)]
        
        # For each color c and each vertex v, track the set of neighbors in color c
        adj = [set() for _ in range(num_colors)]  # adj[c] is a list of sets, one per vertex
        for c in range(num_colors):
            adj[c] = [set() for _ in range(n)]
        
        # Precompute edge list
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        rng = random.Random(attempt * 7919 + 104729)
        shuffled_edges = edges[:]
        rng.shuffle(shuffled_edges)
        
        valid = True
        for (u, v) in shuffled_edges:
            if time.time() - start_time > timeout:
                valid = False
                break
            
            # For each color, compute the "cost" of assigning that color:
            # cost = number of new forbidden pairs created (pairs that would go from 1 to 2 common neighbors)
            # If any pair would go to 2+, that color is invalid.
            best_color = -1
            best_cost = float('inf')
            
            for c in range(num_colors):
                cost, feasible = _evaluate_color(colors, common_count, adj, n, u, v, c)
                if feasible and cost < best_cost:
                    best_cost = cost
                    best_color = c
            
            if best_color == -1:
                valid = False
                break
            
            c = best_color
            # Assign color c to edge (u, v)
            colors[u][v] = c
            colors[v][u] = c
            adj[c][u].add(v)
            adj[c][v].add(u)
            
            # Update common_count:
            # For pair (u,v): common neighbors in c are w in adj[c][u] ∩ adj[c][v]
            #   (excluding u and v themselves, but they're not in their own adj)
            #   Actually, after adding, common_count[c][u][v] = |adj[c][u] ∩ adj[c][v]|
            #   But we need to update it properly.
            
            # The new common neighbors for pair (u,v) are vertices w such that
            # w is in adj[c][u] and w is in adj[c][v].
            # Before adding (u,v) to color c, adj[c][u] and adj[c][v] didn't include each other.
            # After adding, we need to update common_count for affected pairs.
            
            # Pairs affected:
            # 1. Pair (u,v): common neighbors = adj[c][u] ∩ adj[c][v] (now includes each other? No, 
            #    adj[c][u] contains v, adj[c][v] contains u, but u is not in adj[c][v]'s neighbor list
            #    for common neighbor purposes... let me think more carefully.
            #    
            #    common_count[c][a][b] = |{w : w != a, w != b, w in adj[c][a], w in adj[c][b]}|
            #    
            #    When we add edge (u,v) to color c:
            #    - adj[c][u] gains v, adj[c][v] gains u
            #    - For pair (u,v): common neighbors are w in adj[c][u] ∩ adj[c][v] where w != u,v
            #      = (adj[c][u] ∩ adj[c][v]) - {u,v}
            #      Before the update, adj[c][u] and adj[c][v] didn't contain each other.
            #      After, adj[c][u] contains v and adj[c][v] contains u, but u ∉ adj[c][v] as a 
            #      "neighbor for common neighbor count" because we exclude the pair members.
            #      So common_count[c][u][v] = |(old adj[c][u] ∩ old adj[c][v])|
            #      Wait, that's not right either. Let me just recompute.
            #
            #    Actually, the cleanest way: after updating adj, recompute common_count for affected pairs.
            #    Affected pairs are those involving u or v where the other vertex is in adj[c][u] or adj[c][v].
            
            # Let's update common_count for all affected pairs:
            # For each w in adj[c][u] (w != v): pair (v, w) is affected (u becomes a common neighbor)
            # For each w in adj[c][v] (w != u): pair (u, w) is affected (v becomes a common neighbor)
            # For pair (u, v): affected
            
            # Update pair (u, v):
            common_uv = len(adj[c][u] & adj[c][v])
            # Remove u and v from consideration (they shouldn't count as their own common neighbors)
            # adj[c][u] contains v, adj[c][v] contains u, but u is not in adj[c][u] and v is not in adj[c][v]
            # So the intersection won't include u or v unless there's a self-loop (there isn't).
            # Actually, v is in adj[c][u] and u is in adj[c][v], but u is not in adj[c][u] and v is not in adj[c][v].
            # So u is not in adj[c][u] ∩ adj[c][v] and v is not in adj[c][u] ∩ adj[c][v].
            # So common_uv is correct.
            common_count[c][u][v] = common_uv
            common_count[c][v][u] = common_uv
            
            # Update pairs (v, w) for w in adj[c][u] where w != v:
            for w in adj[c][u]:
                if w == v:
                    continue
                # u is now a common neighbor of v and w in color c
                # Recompute common_count[c][v][w]
                common_vw = len(adj[c][v] & adj[c][w])
                common_count[c][v][w] = common_vw
                common_count[c][w][v] = common_vw
            
            # Update pairs (u, w) for w in adj[c][v] where w != u:
            for w in adj[c][v]:
                if w == u:
                    continue
                # v is now a common neighbor of u and w in color c
                # Recompute common_count[c][u][w]
                common_uw = len(adj[c][u] & adj[c][w])
                common_count[c][u][w] = common_uw
                common_count[c][w][u] = common_uw
        
        if valid:
            # Verify no C4 exists
            for c in range(num_colors):
                for a in range(n):
                    for b in range(a+1, n):
                        if common_count[c][a][b] >= 2:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                flat_colors = [colors[i][j] for i in range(n) for j in range(i+1, n)]
                return {'n': n, 'colors': flat_colors}
    
    return None

def _evaluate_color(colors, common_count, adj, n, u, v, c):
    """
    Evaluate the cost of assigning color c to edge (u, v).
    Returns (cost, feasible) where:
      - feasible is False if this would create a C4
      - cost is the number of new "forbidden" pairs (pairs going from 1 to 2 common neighbors)
    """
    # First check: would pair (u,v) have 2+ common neighbors?
    # Current common neighbors of u and v in color c:
    common_uv = 0
    for w in adj[c][u]:
        if w in adj[c][v]:
            common_uv += 1
    # After adding (u,v) to color c, adj[c][u] gains v and adj[c][v] gains u.
    # But v is not in adj[c][v] and u is not in adj[c][u], so the intersection doesn't change
    # due to the new edges themselves. The common neighbors remain the same.
    # Wait, that's correct: adding (u,v) to color c means v becomes a neighbor of u in color c,
    # and u becomes a neighbor of v in color c. But for common neighbors of u and v, we look for
    # w such that w is neighbor of both u and v. v is a neighbor of u but not of v (no self-loops).
    # u is a neighbor of v but not of u. So the set of common neighbors doesn't change.
    # So common_count for (u,v) stays the same.
    
    if common_uv >= 2:
        return (float('inf'), False)
    
    # Now check pairs (v, w) for w in adj[c][u] (w != v):
    # After adding (u,v) to color c, u becomes a common neighbor of v and w.
    # So common_count[c][v][w] increases by 1.
    new_forbidden = 0
    
    for w in adj[c][u]:
        if w == v:
            continue
        # Current common neighbors of v and w in color c:
        # This is common_count[c][v][w] before the update.
        # After adding (u,v) to color c, u becomes a common neighbor of v and w
        # (since u is neighbor of v in color c [new] and u is neighbor of w in color c [existing]).
        # So the new count = common_count[c][v][w] + 1.
        # If common_count[c][v][w] + 1 >= 2, i.e., common_count[c][v][w] >= 1, then
        # this pair becomes forbidden (has 2 common neighbors) -> C4!
        if common_count[c][v][w] >= 1:
            return (float('inf'), False)
        else:
            # common_count[c][v][w] is 0, becomes 1. This creates a new "at-risk" pair.
            new_forbidden += 1
    
    # Check pairs (u, w) for w in adj[c][v] (w != u):
    for w in adj[c][v]:
        if w == u:
            continue
        # After adding (u,v) to color c, v becomes a common neighbor of u and w.
        if common_count[c][u][w] >= 1:
            return (float('inf'), False)
        else:
            new_forbidden += 1
    
    return (new_forbidden, True)
