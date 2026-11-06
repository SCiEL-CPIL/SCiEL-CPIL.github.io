import time
import random

_START_TIME = time.time()
_DEADLINE = 115.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _seed_32():
    """Greenwood-Gleason colouring of K32 (two Clebsch graphs joined by colour 3)."""
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

def _compute_index(n, i, j):
    """Compute the index of pair (i,j) in lexicographic edge ordering."""
    return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)

def construct():
    try:
        # Fallback
        best_n = 32
        best_colors = _seed_32()['colors']
        
        # Target n=50
        n_target = 50
        m_target = n_target * (n_target - 1) // 2
        
        # We will use a fast local search with incremental triangle counting.
        # State: 
        # colors: list of int (0-3) for each edge index
        # adj: list of 4 lists of n integers (bitmasks)
        
        def init_state():
            colors = [random.randint(0, 3) for _ in range(m_target)]
            adj = [[0] * n_target for _ in range(4)]
            idx = 0
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    c = colors[idx]
                    adj[c][i] |= (1 << j)
                    adj[c][j] |= (1 << i)
                    idx += 1
            # Count initial triangles
            tri_count = 0
            for c in range(4):
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        if adj[c][i] >> j & 1:
                            tri_count += (adj[c][i] & adj[c][j]).bit_count()
            # Each triangle is counted 3 times (once for each edge)? 
            # No, if we iterate i<j and check edge (i,j), and count common neighbors k>j, we count each triangle exactly once.
            # My loop above: 
            # for i in range(n):
            #   for j in range(i+1, n):
            #     if edge(i,j) exists:
            #        count common neighbors k > j? 
            # My code above: (adj[c][i] & adj[c][j]).bit_count() counts ALL common neighbors, including those < i.
            # This overcounts. 
            # Let's just use a slower init count or fix the logic.
            # Actually, for local search, we only need DELTA. 
            # We can compute the initial count correctly:
            tri_count = 0
            for c in range(4):
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        if adj[c][i] >> j & 1:
                            # Count k > j
                            common = adj[c][i] & adj[c][j] & ~((1 << (j + 1)) - 1)
                            tri_count += common.bit_count()
            return colors, adj, tri_count

        def get_delta(colors, adj, u, v, c_old, c_new):
            """Calculate change in triangle count if we switch edge (u,v) from c_old to c_new."""
            # Remove edge (u,v) from c_old
            # Triangles removed: common neighbors of u and v in c_old, excluding each other? 
            # The triangles containing edge (u,v) are (u,v,w) where w is a common neighbor in c_old.
            # Since (u,v) is an edge in c_old, u is in adj[c_old][v] and v is in adj[c_old][u].
            # The common neighbors are adj[c_old][u] & adj[c_old][v].
            # This set includes u and v? 
            # adj[c_old][u] has bit v set. adj[c_old][v] has bit u set.
            # So u and v are in the intersection? 
            # adj[c_old][u] & adj[c_old][v] will have bit u set if v's neighbor list contains u (yes) and u's neighbor list contains u (no, self-loop not set).
            # Wait, adj[c][i] does NOT contain bit i.
            # So adj[c_old][u] contains v. adj[c_old][v] contains u.
            # Intersection contains u? No, adj[c_old][u] doesn't have u.
            # Intersection contains v? No, adj[c_old][v] doesn't have v.
            # So the intersection is exactly the set of w such that (u,w) and (v,w) are edges of color c_old.
            # These form triangles (u,v,w).
            # So num_removed = (adj[c_old][u] & adj[c_old][v]).bit_count()
            
            # Add edge (u,v) to c_new
            # Before adding, u and v are NOT connected in c_new.
            # num_added = (adj[c_new][u] & adj[c_new][v]).bit_count()
            
            return num_added - num_removed

        # Main Search Loop
        max_restarts = 10
        steps_per_restart = 5000000 # High, limited by time
        
        for restart in range(max_restarts):
            if _time_left() < 5.0:
                break
            
            colors, adj, tri_count = init_state()
            
            # Simulated Annealing / Greedy Descent
            # We will use a simple greedy descent with random restarts if stuck.
            # Or just greedy descent.
            
            stuck_counter = 0
            for step in range(steps_per_restart):
                if _time_left() < 2.0:
                    break
                
                # Pick random edge
                idx = random.randint(0, m_target - 1)
                
                # Decode idx to (u, v)
                # Inverse of _compute_index
                # i * (n - 1) - i * (i + 1) // 2 + (j - i - 1) = idx
                # Let k = n - 1 - i. Then i = n - 1 - k.
                # This is quadratic. 
                # Approximate i: i^2 / 2 ~ idx => i ~ sqrt(2*idx)
                # Better: binary search or precomputed table.
                # For n=50, we can precompute a mapping from idx to (u,v).
                # But creating it every restart is wasteful. Create once outside.
                
                # Precompute idx_to_pair outside the restart loop? 
                # It depends on n_target.
                
                # Let's precompute idx_to_pair for n=50 once.
                # This takes O(n^2) space, 1225 entries. Trivial.
                
                # We'll define a helper or just do it inline if not cached.
                # To keep code simple and fast, let's precompute it in the function scope.
                
                if not hasattr(construct, '_idx_map'):
                    construct._idx_map = {}
                    for i in range(n_target):
                        for j in range(i + 1, n_target):
                            construct._idx_map[_compute_index(n_target, i, j)] = (i, j)
                
                u, v = construct._idx_map[idx]
                c_old = colors[idx]
                
                best_delta = 0
                best_c_new = -1
                
                for c_new in range(4):
                    if c_new == c_old:
                        continue
                    delta = get_delta(colors, adj, u, v, c_old, c_new)
                    if delta < best_delta:
                        best_delta = delta
                        best_c_new = c_new
                
                if best_delta < 0:
                    # Accept move
                    # Update adj
                    adj[c_old][u] &= ~(1 << v)
                    adj[c_old][v] &= ~(1 << u)
                    adj[best_c_new][u] |= (1 << v)
                    adj[best_c_new][v] |= (1 << u)
                    
                    colors[idx] = best_c_new
                    tri_count += best_delta
                    stuck_counter = 0
                else:
                    # No improving move found for this random edge.
                    # Try a random move with probability? Or just skip.
                    # To escape local minima, we can accept neutral or slightly worse moves occasionally.
                    # But for now, let's just skip to next random edge.
                    stuck_counter += 1
                    
                if tri_count == 0:
                    return {'n': n_target, 'colors': colors}
                    
                if stuck_counter > 10000:
                    # Stuck, break and restart
                    break
                    
        return {'n': best_n, 'colors': best_colors}
        
    except Exception as e:
        try:
            return _seed_32()
        except:
            return {'n': 1, 'colors': []}
