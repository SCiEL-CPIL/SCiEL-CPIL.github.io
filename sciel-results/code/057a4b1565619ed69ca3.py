import time
import random
import math
import numpy as np

def _popcount(x):
    # Python 3.10+ has int.bit_count()
    if hasattr(int, 'bit_count'):
        return x.bit_count()
    return bin(x).count('1')

def _fast_local_search(n, deadline):
    """
    Optimized local search for C4-free coloring.
    """
    if n < 3:
        return True, []
        
    m = n * (n - 1) // 2
    colors = [random.randint(0, 4) for _ in range(m)]
    
    # Precompute edge endpoints
    edge_u = [0] * m
    edge_v = [0] * m
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_u[idx] = i
            edge_v[idx] = j
            idx += 1
            
    # masks[v][c] is a bitmask of neighbors of v in color c
    masks = [[0]*5 for _ in range(n)]
    
    def build_masks():
        for v in range(n):
            for c in range(5):
                masks[v][c] = 0
        for k in range(m):
            u = edge_u[k]
            v = edge_v[k]
            c = colors[k]
            masks[u][c] |= (1 << v)
            masks[v][c] |= (1 << u)
            
    build_masks()
    
    def count_violations():
        total = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = _popcount(masks[u][c] & masks[v][c])
                    if common >= 2:
                        total += common - 1
        return total
        
    current_violations = count_violations()
    best_violations = current_violations
    best_colors = colors[:]
    
    if current_violations == 0:
        return True, colors
        
    T = 1.0
    alpha = 0.9995
    min_T = 0.01
    
    # Precompute list of edges for random selection
    # Using random.randint is fine.
    
    while time.time() < deadline:
        k = random.randint(0, m - 1)
        u = edge_u[k]
        v = edge_v[k]
        old_c = colors[k]
        new_c = random.randint(0, 4)
        
        if new_c == old_c:
            continue
            
        # Calculate delta efficiently
        # Violations are defined by pairs (x,y) having >= 2 common neighbors in color c.
        # Changing (u,v) from old_c to new_c affects:
        # 1. Pairs (u, w) and (v, w) in old_c and new_c.
        # 2. Pair (u, v) in old_c and new_c.
        
        # We can compute the change in "common neighbor count" for affected pairs.
        # Let CN(x, y, c) = popcount(mask[x][c] & mask[y][c]).
        # Violation contribution of pair (x,y) in color c is max(0, CN(x,y,c) - 1).
        
        # Pairs affected in old_c: (u,w), (v,w), (u,v) for w != u,v.
        # Pairs affected in new_c: (u,w), (v,w), (u,v) for w != u,v.
        
        delta = 0
        
        # Helper to get contribution
        def contrib(x, y, c):
            if x == y: return 0
            cn = _popcount(masks[x][c] & masks[y][c])
            return max(0, cn - 1)
            
        # Sum old contributions
        old_sum = 0
        new_sum_old = 0 # Contributions in new_c BEFORE change
        new_sum_new = 0 # Contributions in old_c AFTER change? No.
        
        # Before change:
        # old_c has edge (u,v). new_c does not.
        # new_c has no edge (u,v). old_c does not.
        
        # Affected pairs:
        # For each w in 0..n-1, w != u, w != v:
        #   Pair (u,w): 
        #     In old_c: u is neighbor of w? No, edge (u,w) has some color. 
        #     Wait, masks[u][c] contains v if edge(u,v) is color c.
        #     So changing edge(u,v) changes masks[u][old_c] and masks[v][old_c] (removes each other).
        #     And masks[u][new_c] and masks[v][new_c] (adds each other).
        
        # This changes CN(x, y, c) for:
        # 1. Pairs (x, u) or (x, v) in old_c or new_c?
        #    CN(x, u, c) depends on mask[x][c] & mask[u][c].
        #    mask[u][old_c] loses v. So CN(x, u, old_c) changes if x is connected to v in old_c?
        #    No, CN(x, u, c) counts common neighbors.
        #    If we remove v from mask[u][old_c], then for any x, 
        #    CN(x, u, old_c) decreases by 1 if v is in mask[x][old_c] AND v is in mask[u][old_c] (which it was).
        #    So CN(x, u, old_c) decreases by 1 if x ~_old_c v.
        
        # This is getting complicated to update incrementally without rebuilding.
        # Given n is small (<= 29), rebuilding masks is O(m) = O(n^2).
        # Rebuilding violations is O(n^2 * 5).
        # For n=26, m=325. n^2=676.
        # Rebuild is ~1000 ops.
        # Python overhead is high.
        # The current incremental method is O(n) per step.
        # O(n) is ~26 ops.
        # So incremental is ~40x faster than rebuild.
        # The current code is already incremental.
        # The bottleneck is Python function calls and loop overhead.
        
        # Let's stick to the incremental method but optimize the inner loop.
        
        # We will compute delta by summing changes in contribution for all affected pairs.
        
        # Affected pairs in old_c:
        # (u, w) for w != u, v.
        # (v, w) for w != u, v.
        # (u, v).
        
        # Affected pairs in new_c:
        # (u, w) for w != u, v.
        # (v, w) for w != u, v.
        # (u, v).
        
        # Note: The masks for u and v change.
        # mask[u][old_c] loses v.
        # mask[v][old_c] loses u.
        # mask[u][new_c] gains v.
        # mask[v][new_c] gains u.
        
        # Change in CN(x, y, c) for a pair (x,y):
        # CN(x, y, c) = popcount(mask[x][c] & mask[y][c]).
        
        # For pair (u, w) in old_c:
        # mask[u][old_c] loses v.
        # So CN(u, w, old_c) changes by -1 if v is in mask[w][old_c].
        # i.e., if w ~_old_c v.
        
        # For pair (v, w) in old_c:
        # mask[v][old_c] loses u.
        # So CN(v, w, old_c) changes by -1 if u is in mask[w][old_c].
        # i.e., if w ~_old_c u.
        
        # For pair (u, v) in old_c:
        # mask[u][old_c] loses v, mask[v][old_c] loses u.
        # CN(u, v, old_c) = popcount(mask[u][old_c] & mask[v][old_c]).
        # Before: includes u in mask[v]? No, mask[v][old_c] has u. mask[u][old_c] has v.
        # Wait, mask[u][c] is neighbors of u.
        # So v is in mask[u][old_c]. u is in mask[v][old_c].
        # Intersection mask[u][old_c] & mask[v][old_c] does NOT contain u or v (usually, unless self-loops, which we don't have).
        # So removing v from mask[u] and u from mask[v] does NOT change the intersection mask[u] & mask[v] directly?
        # Wait. mask[u] & mask[v] contains x if x is neighbor of u AND x is neighbor of v.
        # Removing v from mask[u] means v is no longer a neighbor of u.
        # Does v appear in mask[u] & mask[v]?
        # v is in mask[u]. Is v in mask[v]? No (no self loop).
        # So v was NOT in the intersection.
        # So removing v from mask[u] does NOT change mask[u] & mask[v].
        # Similarly removing u from mask[v] does NOT change mask[u] & mask[v].
        # So CN(u, v, old_c) is UNCHANGED?
        # Let's verify.
        # Common neighbors of u and v are w such that w~u and w~v.
        # Changing color of (u,v) does not change neighbors of u or v OTHER than u and v themselves.
        # But u is not a neighbor of u. v is not a neighbor of v.
        # So the set of common neighbors of u and v is unchanged.
        # So contribution of (u,v) is unchanged.
        
        # So we only need to update pairs (u, w) and (v, w) for w != u, v.
        
        # For (u, w) in old_c:
        # CN(u, w, old_c) changes by -1 if v ~_old_c w.
        # Let delta_cn = -1 if (1 << v) & masks[w][old_c] else 0.
        # Contribution change: f(CN+delta) - f(CN).
        
        # For (v, w) in old_c:
        # CN(v, w, old_c) changes by -1 if u ~_old_c w.
        
        # For (u, w) in new_c:
        # mask[u][new_c] gains v.
        # CN(u, w, new_c) changes by +1 if v ~_new_c w.
        
        # For (v, w) in new_c:
        # mask[v][new_c] gains u.
        # CN(v, w, new_c) changes by +1 if u ~_new_c w.
        
        # Total delta:
        delta = 0
        for w in range(n):
            if w == u or w == v: continue
            
            # Pair (u, w) in old_c
            if (1 << v) & masks[w][old_c]:
                cn = _popcount(masks[u][old_c] & masks[w][old_c])
                # cn will decrease by 1
                delta += max(0, cn - 2) - max(0, cn - 1)
                
            # Pair (v, w) in old_c
            if (1 << u) & masks[w][old_c]:
                cn = _popcount(masks[v][old_c] & masks[w][old_c])
                # cn will decrease by 1
                delta += max(0, cn - 2) - max(0, cn - 1)
                
            # Pair (u, w) in new_c
            if (1 << v) & masks[w][new_c]:
                cn = _popcount(masks[u][new_c] & masks[w][new_c])
                # cn will increase by 1
                delta += max(0, cn) - max(0, cn - 1)
                
            # Pair (v, w) in new_c
            if (1 << u) & masks[w][new_c]:
                cn = _popcount(masks[v][new_c] & masks[w][new_c])
                # cn will increase by 1
                delta += max(0, cn) - max(0, cn - 1)
                
        # Apply change
        masks[u][old_c] &= ~(1 << v)
        masks[v][old_c] &= ~(1 << u)
        masks[u][new_c] |= (1 << v)
        masks[v][new_c] |= (1 << u)
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
            colors[k] = old_c
            masks[u][new_c] &= ~(1 << v)
            masks[v][new_c] &= ~(1 << u)
            masks[u][old_c] |= (1 << v)
            masks[v][old_c] |= (1 << u)
            current_violations -= delta
            
        T *= alpha
        if T < min_T:
            T = min_T
            
    if best_violations == 0:
        return True, best_colors
    return False, best_colors

def construct():
    deadline = time.time() + 115
    
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    targets = [27, 26, 25, 24]
    
    for n in targets:
        if time.time() > deadline - 5:
            break
            
        if n == 27:
            t_limit = 20
        elif n == 26:
            t_limit = 40
        else:
            t_limit = 10
            
        actual_limit = min(deadline - time.time() - 2, t_limit)
        if actual_limit < 1:
            continue
            
        success, colors = _fast_local_search(n, time.time() + actual_limit)
        
        if success:
            best_n = n
            best_colors = colors
            break
            
    return {'n': best_n, 'colors': best_colors}
