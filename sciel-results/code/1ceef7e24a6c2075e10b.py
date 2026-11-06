import time
import random
import math
import sys

def construct():
    deadline = time.time() + 115
    best_n = 0
    best_colors = []
    
    # Try 26 first (known achievable), then 25, etc.
    # We don't try 27 as it's likely infeasible within time limits for a random search
    # without very specific structural initialization.
    
    for n in [26, 25, 24, 23, 22, 21, 20]:
        if time.time() > deadline - 5:
            break
            
        # Time budget allocation
        if n == 26:
            t_limit = 50
        elif n == 25:
            t_limit = 30
        elif n == 24:
            t_limit = 20
        else:
            t_limit = 10
            
        actual_limit = min(deadline - time.time() - 2, t_limit)
        if actual_limit < 1:
            continue
            
        success, colors = _solve_n(n, time.time() + actual_limit)
        if success:
            best_n = n
            best_colors = colors
            break
            
    if best_n == 0:
        # Fallback to a small valid coloring if all else fails
        # n=11 is usually easy
        success, colors = _solve_n(11, time.time() + 5)
        if success:
            best_n = 11
            best_colors = colors
        else:
            # Hardcode a valid coloring for n=11? 
            # For n=1, no edges, valid.
            best_n = 1
            best_colors = []

    return {'n': best_n, 'colors': best_colors}

def _solve_n(n, deadline):
    m = n * (n - 1) // 2
    if m == 0:
        return True, []
        
    # Build edge list
    edge_u = [0] * m
    edge_v = [0] * m
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_u[idx] = i
            edge_v[idx] = j
            pair_to_idx[(i, j)] = idx
            pair_to_idx[(j, i)] = idx
            idx += 1
            
    # Try multiple random restarts
    # Each restart: initialize randomly, then run local search / SA
    
    # Precompute adjacency lists for speed? 
    # For n=26, n is small. 
    # Key optimization: fast violation counting and update.
    
    # Let's use a different approach for initialization:
    # Instead of pure random, maybe use a greedy approach or a known structure?
    # For now, random with good local search.
    
    # Local Search Strategy:
    # 1. Count violations (number of C4s in each color).
    # 2. Pick a random edge.
    # 3. Try changing its color.
    # 4. Calculate delta in violations.
    # 5. Accept if delta < 0 or with prob exp(-delta/T).
    
    # To speed up delta calculation, we maintain:
    # common_count[c][u][v] = number of common neighbors of u and v in color c.
    # A C4 exists if common_count[c][u][v] >= 2.
    # Violations = sum over c, u<v of (1 if common_count[c][u][v] >= 2 else 0)
    # Actually, we want to minimize the number of C4s. 
    # Let's define cost = sum over c, u<v of max(0, common_count[c][u][v] - 1).
    # This penalizes extra common neighbors.
    
    # Let's stick to: cost = number of pairs (u,v) such that common_count[c][u][v] >= 2.
    # This is equivalent to the number of C4s if we count each C4 once?
    # No, a C4 is defined by 4 vertices. A pair of common neighbors defines a C4.
    # If common_count is 2, there is 1 C4. If 3, there are 3 C4s (choose 2 from 3).
    # So cost = sum over c, u<v of C(common_count[c][u][v], 2).
    
    # Let's use this cost function.
    
    def init_state():
        colors = [random.randint(0, 4) for _ in range(m)]
        common_count = [[[0]*n for _ in range(n)] for _ in range(5)]
        
        # Initialize common_count
        # For each color c, for each edge (u,v) of color c, 
        # for each neighbor w of u (in any color), if (v,w) is also color c, increment.
        # This is O(m * n) which is ~ 325 * 26 = 8450. Fast.
        
        # Better:
        # For each color c:
        #   Build adjacency list adj_c[u] = list of v such that (u,v) is color c.
        #   For each u, for each v in adj_c[u]:
        #       For each w in adj_c[u] where w > v:
        #           common_count[c][v][w] += 1
        #           common_count[c][w][v] += 1
        # Wait, this counts pairs of neighbors.
        # If u is connected to v and w in color c, then v and w have a common neighbor u in color c.
        # So for each u, and each pair (v,w) in adj_c[u], we increment common_count[c][v][w].
        
        adj = [[] for _ in range(5)]
        for c in range(5):
            adj[c] = [[] for _ in range(n)]
            
        for k in range(m):
            c = colors[k]
            u = edge_u[k]
            v = edge_v[k]
            adj[c][u].append(v)
            adj[c][v].append(u)
            
        cost = 0
        for c in range(5):
            for u in range(n):
                neighbors = adj[c][u]
                ln = len(neighbors)
                for i in range(ln):
                    v = neighbors[i]
                    for j in range(i+1, ln):
                        w = neighbors[j]
                        common_count[c][v][w] += 1
                        common_count[c][w][v] += 1
                        # Update cost
                        # If count becomes 2, cost increases by 1.
                        # If count becomes 3, cost increases by 2 (C(3,2)-C(2,2) = 3-1=2).
                        # General: delta_cost = new_count - 1 (if new_count >= 2)
                        # Actually, C(k,2) - C(k-1,2) = k-1.
                        # So if common_count[c][v][w] becomes k, cost increases by k-1.
                        cnt = common_count[c][v][w]
                        cost += (cnt - 1)
                        
        return colors, common_count, cost, adj

    def local_search(start_colors, start_common_count, start_cost, start_adj, deadline):
        colors = start_colors[:]
        common_count = [[[row[:] for row in mat] for mat in cc] for cc in start_common_count]
        cost = start_cost
        adj = [ [lst[:] for lst in adj_c] for adj_c in start_adj ]
        
        best_cost = cost
        best_colors = colors[:]
        
        T = 1.0
        alpha = 0.9995
        min_T = 0.001
        start_time = time.time()
        
        # Precompute edge indices for fast access
        # We already have pair_to_idx.
        
        while time.time() < deadline:
            k = random.randint(0, m - 1)
            u = edge_u[k]
            v = edge_v[k]
            old_c = colors[k]
            new_c = random.randint(0, 4)
            
            if new_c == old_c:
                continue
            
            # Calculate delta cost
            delta = 0
            revert_info = []
            
            # The edge (u,v) changes from old_c to new_c.
            # This affects:
            # 1. The adjacency lists of u and v in old_c and new_c.
            # 2. The common counts of pairs involving neighbors of u and v.
            
            # Let's trace the impact on common_count.
            # common_count[c][x][y] is the number of z such that (z,x) and (z,y) are color c.
            
            # When (u,v) changes:
            # For any w:
            # - If (u,w) is old_c, then u was a common neighbor of v and w in old_c.
            #   Now (u,v) is new_c, so u is NO LONGER a common neighbor of v and w in old_c.
            #   So common_count[old_c][v][w] decreases by 1.
            #   (And symmetrically common_count[old_c][w][v]).
            # - If (u,w) is new_c, then u is NOW a common neighbor of v and w in new_c.
            #   So common_count[new_c][v][w] increases by 1.
            
            # - If (v,w) is old_c, then v was a common neighbor of u and w in old_c.
            #   Now (v,u) is new_c, so v is NO LONGER a common neighbor of u and w in old_c.
            #   So common_count[old_c][u][w] decreases by 1.
            # - If (v,w) is new_c, then v is NOW a common neighbor of u and w in new_c.
            #   So common_count[new_c][u][w] increases by 1.
            
            # Note: w cannot be u or v.
            
            for w in range(n):
                if w == u or w == v:
                    continue
                
                # Check color of (u, w)
                c_uw = colors[pair_to_idx[(u, w)]]
                # Check color of (v, w)
                c_vw = colors[pair_to_idx[(v, w)]]
                
                # Impact on pair (v, w)
                if c_uw == old_c:
                    # common_count[old_c][v][w] decreases
                    old_val = common_count[old_c][v][w]
                    new_val = old_val - 1
                    # Cost change: C(new_val, 2) - C(old_val, 2) = (new_val - 1) - (old_val - 1) = -1 if old_val >= 2
                    # If old_val == 2, new_val == 1. C(1,2)=0, C(2,2)=1. Delta = -1.
                    # If old_val == 3, new_val == 2. C(2,2)=1, C(3,2)=3. Delta = -2.
                    # So delta = -(old_val - 1) if old_val >= 2 else 0.
                    if old_val >= 2:
                        delta -= (old_val - 1)
                    common_count[old_c][v][w] = new_val
                    common_count[old_c][w][v] = new_val
                    revert_info.append(('dec', old_c, v, w, old_val))
                    
                if c_uw == new_c:
                    # common_count[new_c][v][w] increases
                    old_val = common_count[new_c][v][w]
                    new_val = old_val + 1
                    # Cost change: C(new_val, 2) - C(old_val, 2) = new_val - 1.
                    delta += (new_val - 1)
                    common_count[new_c][v][w] = new_val
                    common_count[new_c][w][v] = new_val
                    revert_info.append(('inc', new_c, v, w, old_val))
                    
                # Impact on pair (u, w)
                if c_vw == old_c:
                    # common_count[old_c][u][w] decreases
                    old_val = common_count[old_c][u][w]
                    new_val = old_val - 1
                    if old_val >= 2:
                        delta -= (old_val - 1)
                    common_count[old_c][u][w] = new_val
                    common_count[old_c][w][u] = new_val
                    revert_info.append(('dec', old_c, u, w, old_val))
                    
                if c_vw == new_c:
                    # common_count[new_c][u][w] increases
                    old_val = common_count[new_c][u][w]
                    new_val = old_val + 1
                    delta += (new_val - 1)
                    common_count[new_c][u][w] = new_val
                    common_count[new_c][w][u] = new_val
                    revert_info.append(('inc', new_c, u, w, old_val))
                
            cost += delta
            
            if delta <= 0 or random.random() < math.exp(-delta / T):
                # Accept
                # Update colors
                colors[k] = new_c
                
                # Update adj lists
                adj[old_c][u].remove(v)
                adj[old_c][v].remove(u)
                adj[new_c][u].append(v)
                adj[new_c][v].append(u)
                
                if cost < best_cost:
                    best_cost = cost
                    best_colors = colors[:]
                    if best_cost == 0:
                        return True, best_colors
            else:
                # Revert
                colors[k] = old_c
                # Revert adj lists (not strictly necessary if we revert common_count correctly,
                # but adj lists are used for initialization only? No, we don't use them in the loop.
                # We only use pair_to_idx. So we can skip adj updates in the loop for speed.
                # Wait, I did update adj above. Let's revert it or not use it.
                # Actually, I don't use adj in the loop. I use pair_to_idx.
                # So I can remove adj updates from the loop to save time.
                # Let's remove adj updates.
                
                for op, c, x, y, old_val in revert_info:
                    if op == 'dec':
                        common_count[c][x][y] = old_val
                        common_count[c][y][x] = old_val
                    else:
                        common_count[c][x][y] = old_val
                        common_count[c][y][x] = old_val
                cost -= delta
                
            T *= alpha
            if T < min_T:
                T = min_T
                
        if best_cost == 0:
            return True, best_colors
        return False, best_colors

    # Main loop: try multiple restarts
    start_time = time.time()
    restarts = 0
    while time.time() < deadline:
        colors, common_count, cost, adj = init_state()
        if cost == 0:
            return True, colors
            
        success, best_colors = local_search(colors, common_count, cost, adj, deadline)
        if success:
            return True, best_colors
            
        restarts += 1
        # If we have time, try again
        if time.time() > deadline - 2:
            break
            
    return False, []

# Test
if __name__ == "__main__":
    res = construct()
    print(res['n'])
