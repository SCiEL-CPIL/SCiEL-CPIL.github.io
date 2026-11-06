import time
import random
import math
import numpy as np

def construct():
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Known valid coloring for n=11 (fallback)
    # Using the formula from the seed: min(j-i, n-j+i) - 1
    # For n=11:
    # i=0: j=1..10 -> min(j, 11-j)-1
    # j=1: min(1,10)-1=0
    # j=2: min(2,9)-1=1
    # j=3: min(3,8)-1=2
    # j=4: min(4,7)-1=3
    # j=5: min(5,6)-1=4
    # j=6: min(6,5)-1=4
    # j=7: min(7,4)-1=3
    # j=8: min(8,3)-1=2
    # j=9: min(9,2)-1=1
    # j=10: min(10,1)-1=0
    # This pattern repeats for other i's.
    # Let's just hardcode a small valid one or generate one quickly.
    # Actually, the seed program returns n=11 with specific colors.
    # I will use a simple deterministic construction for n=11 as fallback.
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Targets to try, in descending order of priority
    # We want to maximize n.
    # Try 27, then 26, then 25...
    # If we find a valid coloring for n, we return it immediately (since we go down).
    # Wait, if we find 25, we should stop? Yes, because we tried 27 and 26 and failed.
    # So the first success in the loop [27, 26, 25, ...] is the maximum n found.
    
    targets = [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12]
    
    for n in targets:
        if time.time() > deadline - 2:
            break
            
        # Allocate time based on n
        # Give more time to larger n as they are harder but more valuable.
        # But if 27 is impossible, we shouldn't waste too much time.
        # Let's give 20s for 27, 30s for 26, 10s for 25, 5s for others.
        if n == 27:
            t_limit = 20
        elif n == 26:
            t_limit = 30
        elif n == 25:
            t_limit = 10
        else:
            t_limit = 5
            
        actual_limit = min(deadline - time.time() - 2, t_limit)
        if actual_limit < 1:
            continue
            
        success, colors = _find_coloring(n, time.time() + actual_limit)
        
        if success:
            best_n = n
            best_colors = colors
            break  # We found the largest n that works (so far)
            
    return {'n': best_n, 'colors': best_colors}

def _find_coloring(n, deadline):
    """
    Attempts to find a valid 5-coloring of K_n with no monochromatic C4.
    Uses Simulated Annealing with efficient violation counting.
    """
    if n < 3:
        return True, []
        
    m = n * (n - 1) // 2
    
    # Edge index mapping
    # We need to map (u, v) to an index in the colors array.
    # And we need to know u and v for each index.
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
            
    # Initial random coloring
    colors = [random.randint(0, 4) for _ in range(m)]
    
    # Adjacency lists per color: adj[c][v] = list of neighbors of v in color c
    # Or better, use numpy arrays for fast common neighbor counting?
    # For n=26, m=325.
    # Let's use bitmasks for adjacency.
    # adj_mask[c][v] is an integer where bit u is 1 if (u,v) is color c.
    
    adj_mask = [[0] * n for _ in range(5)]
    
    def update_masks():
        for c in range(5):
            for v in range(n):
                adj_mask[c][v] = 0
        for k in range(m):
            u = edge_u[k]
            v = edge_v[k]
            c = colors[k]
            adj_mask[c][u] |= (1 << v)
            adj_mask[c][v] |= (1 << u)
            
    update_masks()
    
    def count_violations():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    # Check if u and v have >= 2 common neighbors in color c
                    # common = adj_mask[c][u] & adj_mask[c][v]
                    # But we must exclude u and v themselves?
                    # No, adj_mask[c][u] contains v, and adj_mask[c][v] contains u.
                    # The intersection will contain bits for common neighbors.
                    # It will also contain... wait.
                    # If u and v are connected by color c, then v is in adj_mask[c][u] and u is in adj_mask[c][v].
                    # The intersection adj_mask[c][u] & adj_mask[c][v] will NOT contain u or v,
                    # because u is not in adj_mask[c][u] (no self-loops) and v is not in adj_mask[c][v].
                    # So the intersection is exactly the set of common neighbors.
                    common = adj_mask[c][u] & adj_mask[c][v]
                    # Count bits
                    cnt = bin(common).count('1')
                    if cnt >= 2:
                        total += cnt - 1
        return total
        
    current_violations = count_violations()
    best_violations = current_violations
    best_colors = colors[:]
    
    if current_violations == 0:
        return True, colors
        
    T = 1.0
    alpha = 0.9998
    min_T = 0.01
    start_time = time.time()
    
    # Precompute list of edges for random selection
    # We can just pick random k
    
    while time.time() < deadline:
        # Pick random edge
        k = random.randint(0, m - 1)
        u = edge_u[k]
        v = edge_v[k]
        old_c = colors[k]
        new_c = random.randint(0, 4)
        
        if new_c == old_c:
            continue
            
        # Calculate Delta
        # Affected pairs:
        # 1. (u, v) itself
        # 2. (u, w) for all w != u, v
        # 3. (v, w) for all w != u, v
        
        # We need to calculate the change in violations for these pairs.
        # A pair (a, b) contributes to violations if they have >= 2 common neighbors in some color c.
        # Specifically, for a fixed color c, the contribution is max(0, common_neighbors(a, b, c) - 1).
        
        # When we change color of (u, v) from old_c to new_c:
        # - For color old_c:
        #   - (u, v) loses a common neighbor? No, (u, v) are the pair.
        #   - The set of common neighbors for any pair involving u or v might change.
        #   - Specifically, for pair (u, w), w is a common neighbor in old_c if w is connected to u in old_c AND w is connected to v in old_c? No.
        #   - Common neighbors of (u, w) in color c are vertices x such that (u,x) is c and (w,x) is c.
        #   - Changing (u,v) affects:
        #     - Common neighbors of (u, w) for all w: v is a common neighbor of (u, w) in old_c if (w,v) is old_c.
        #       So if we remove (u,v) from old_c, v is no longer a common neighbor of (u, w) in old_c.
        #     - Common neighbors of (v, w) for all w: u is a common neighbor of (v, w) in old_c if (w,u) is old_c.
        #       So if we remove (u,v) from old_c, u is no longer a common neighbor of (v, w) in old_c.
        #     - Common neighbors of (u, v) in old_c: The set of x such that (u,x) is old_c and (v,x) is old_c.
        #       This set doesn't change when we change (u,v) itself, because x is not u or v.
        #       Wait, the definition of C4 is 4 distinct vertices a,b,c,d.
        #       So common neighbors of (u,v) are x != u,v.
        #       Changing color of (u,v) does not change the colors of (u,x) or (v,x).
        #       So the number of common neighbors of (u,v) in old_c remains the same?
        #       Yes. But the pair (u,v) is no longer "connected" in old_c.
        #       Does a pair need to be connected in color c to form a C4 in color c?
        #       Yes, a C4 in color c is a cycle u-w-v-x-u where all edges are color c.
        #       This requires (u,w), (w,v), (v,x), (x,u) to be color c.
        #       This is equivalent to w and x being common neighbors of u and v in color c.
        #       So the condition is: u and v have at least 2 common neighbors in color c.
        #       This condition does NOT depend on the color of the edge (u,v) itself.
        #       It only depends on the colors of edges incident to u and v (excluding (u,v)).
        #       
        #       WAIT.
        #       If u and v have 2 common neighbors in color c, say w and x.
        #       Then we have edges (u,w), (v,w), (u,x), (v,x) all in color c.
        #       This forms a K_{2,2} subgraph (a C4) with vertices {u,v,w,x}.
        #       The edges of the C4 are (u,w), (w,v), (v,x), (x,u).
        #       Notice that the edge (u,v) is NOT part of the C4.
        #       So the color of (u,v) does NOT affect whether (u,v) forms a C4 with other vertices.
        #       
        #       Therefore, changing the color of (u,v) does NOT change the violation status of the pair (u,v).
        #       It DOES change the violation status of pairs (u, w) and (v, w).
        #       Why?
        #       Consider pair (u, w). Its common neighbors in color c are vertices x such that (u,x) is c and (w,x) is c.
        #       If we change (u,v) from old_c to new_c:
        #       - In old_c: v is a common neighbor of (u, w) if (w,v) is old_c.
        #         Before change: (u,v) is old_c. So v is a neighbor of u in old_c.
        #         If (w,v) is old_c, then v is a common neighbor of (u,w) in old_c.
        #         After change: (u,v) is new_c. So v is NOT a neighbor of u in old_c.
        #         So v is no longer a common neighbor of (u,w) in old_c.
        #         So the count of common neighbors of (u,w) in old_c decreases by 1 if (w,v) is old_c.
        #       - In new_c:
        #         Before change: (u,v) is old_c. So v is NOT a neighbor of u in new_c (assuming old_c != new_c).
        #         After change: (u,v) is new_c. So v IS a neighbor of u in new_c.
        #         If (w,v) is new_c, then v becomes a common neighbor of (u,w) in new_c.
        #         So the count of common neighbors of (u,w) in new_c increases by 1 if (w,v) is new_c.
        #       
        #       Similarly for pair (v, w).
        #       
        #       So the delta in violations comes from changes in common neighbor counts for pairs (u, w) and (v, w).
        
        delta = 0
        
        # For each w != u, v:
        # Pair (u, w):
        #   old_c count changes by -1 if (w, v) is old_c.
        #   new_c count changes by +1 if (w, v) is new_c.
        #   Violation contribution is max(0, count - 1).
        #   Let C_old = count_old(u, w, old_c)
        #   Let C_new = count_old(u, w, new_c)
        #   After change:
        #   C_old' = C_old - (1 if (w, v) is old_c else 0)
        #   C_new' = C_new + (1 if (w, v) is new_c else 0)
        #   Delta_violation = max(0, C_old' - 1) + max(0, C_new' - 1) - (max(0, C_old - 1) + max(0, C_new - 1))
        
        # We can precompute counts? Or just use masks.
        # Using masks is O(1) per pair if we use bit operations.
        
        # Let's compute the delta for all w.
        # To speed this up, we can iterate w.
        
        # Optimization:
        # We only need to check w where the count changes.
        # (w, v) is old_c: w is in adj_mask[old_c][v].
        # (w, v) is new_c: w is in adj_mask[new_c][v].
        
        # So for w in adj_mask[old_c][v]:
        #   C_old(u, w, old_c) decreases by 1.
        #   This might reduce violations for (u, w) in old_c.
        #   Specifically, if C_old >= 2, it was contributing C_old - 1.
        #   Now it contributes C_old - 2.
        #   Delta = -1.
        #   If C_old == 1, it was contributing 0.
        #   Now it contributes 0.
        #   Delta = 0.
        #   So for each w in adj_mask[old_c][v], if count_common(u, w, old_c) >= 2, delta -= 1.
        #   Note: count_common(u, w, old_c) includes v?
        #   Yes, v is a common neighbor of (u, w) in old_c if (w, v) is old_c and (u, v) is old_c.
        #   Since we are removing (u, v) from old_c, v is no longer a common neighbor.
        #   So the count decreases by 1.
        #   The "old" count included v.
        #   So if the old count was K, the new count is K-1.
        #   If K >= 2, the contribution goes from K-1 to K-2. Delta = -1.
        #   If K == 1, the contribution goes from 0 to 0. Delta = 0.
        #   Wait, if K == 1, the only common neighbor was v.
        #   After removal, count is 0. Contribution 0.
        #   So yes, delta is -1 if old_count >= 2, else 0.
        
        # Similarly for new_c:
        # For w in adj_mask[new_c][v]:
        #   (w, v) is new_c.
        #   We are adding (u, v) to new_c.
        #   So v becomes a common neighbor of (u, w) in new_c.
        #   Count increases by 1.
        #   If new_count (before) was K, new count is K+1.
        #   If K+1 >= 2 (i.e., K >= 1), the contribution goes from max(0, K-1) to K.
        #   If K == 0: 0 -> 0. Delta 0.
        #   If K == 1: 0 -> 1. Delta +1.
        #   If K >= 2: K-1 -> K. Delta +1.
        #   So delta is +1 if old_count >= 1, else 0.
        
        # Wait, this logic applies to pair (u, w).
        # We also have pair (v, w).
        # By symmetry, the same logic applies with u and v swapped.
        
        # So total delta =
        #   Sum over w in adj_mask[old_c][v] of ( -1 if count_common(u, w, old_c) >= 2 else 0 )
        #   + Sum over w in adj_mask[new_c][v] of ( +1 if count_common(u, w, new_c) >= 1 else 0 )
        #   + Sum over w in adj_mask[old_c][u] of ( -1 if count_common(v, w, old_c) >= 2 else 0 )
        #   + Sum over w in adj_mask[new_c][u] of ( +1 if count_common(v, w, new_c) >= 1 else 0 )
        
        # Note: w must be != u, v.
        # adj_mask[old_c][v] contains u? Yes, because (u,v) is old_c.
        # So when we iterate w in adj_mask[old_c][v], we must skip w=u.
        # Similarly for others.
        
        # Let's implement this efficiently.
        
        # Helper to get count of common neighbors
        def count_common(a, b, c):
            return bin(adj_mask[c][a] & adj_mask[c][b]).count('1')
            
        # Iterate w for pair (u, w)
        # w in adj_mask[old_c][v]
        mask_w_old_v = adj_mask[old_c][v]
        # Skip u
        # Iterate set bits
        w = 0
        mask = mask_w_old_v
        while mask:
            # Get lowest set bit
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            mask -= lsb
            if w == u:
                continue
            # Check count_common(u, w, old_c)
            # This count includes v.
            # We want to know if the count BEFORE removal is >= 2.
            # If count >= 2, then after removal it is >= 1, so violation decreases by 1?
            # Wait.
            # If count >= 2, contribution was count - 1.
            # After removal, count is count - 1. Contribution is count - 2.
            # Delta = -1.
            # If count == 1, contribution was 0.
            # After removal, count is 0. Contribution is 0.
            # Delta = 0.
            # So we just check if current count >= 2.
            cnt = count_common(u, w, old_c)
            if cnt >= 2:
                delta -= 1
                
        # w in adj_mask[new_c][v]
        mask_w_new_v = adj_mask[new_c][v]
        mask = mask_w_new_v
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            mask -= lsb
            if w == u:
                continue
            # Check count_common(u, w, new_c)
            # This count does NOT include v (since (u,v) is not new_c yet).
            # After addition, count increases by 1.
            # If old count >= 1, then new count >= 2, so violation increases by 1.
            # If old count == 0, new count == 1, violation stays 0.
            cnt = count_common(u, w, new_c)
            if cnt >= 1:
                delta += 1
                
        # Iterate w for pair (v, w)
        # w in adj_mask[old_c][u]
        mask_w_old_u = adj_mask[old_c][u]
        mask = mask_w_old_u
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            mask -= lsb
            if w == v:
                continue
            cnt = count_common(v, w, old_c)
            if cnt >= 2:
                delta -= 1
                
        # w in adj_mask[new_c][u]
        mask_w_new_u = adj_mask[new_c][u]
        mask = mask_w_new_u
        while mask:
            lsb = mask & -mask
            w = lsb.bit_length() - 1
            mask -= lsb
            if w == v:
                continue
            cnt = count_common(v, w, new_c)
            if cnt >= 1:
                delta += 1
                
        # Apply change
        adj_mask[old_c][u] &= ~(1 << v)
        adj_mask[old_c][v] &= ~(1 << u)
        adj_mask[new_c][u] |= (1 << v)
        adj_mask[new_c][v] |= (1 << u)
        colors[k] = new_c
        
        current_violations += delta
        
        if delta <= 0 or random.random() < math.exp(-delta / T):
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return True, best_colors
        else:
            # Revert
            adj_mask[new_c][u] &= ~(1 << v)
            adj_mask[new_c][v] &= ~(1 << u)
            adj_mask[old_c][u] |= (1 << v)
            adj_mask[old_c][v] |= (1 << u)
            colors[k] = old_c
            current_violations -= delta
            
        T *= alpha
        if T < min_T:
            T = min_T
            
    if best_violations == 0:
        return True, best_colors
    return False, best_colors
