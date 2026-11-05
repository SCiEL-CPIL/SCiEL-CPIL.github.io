import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # We aim for n=26
    n = 26
    num_colors = 5
    
    # State:
    # color[i][j] for i < j
    # masks[c][i] : bitmask of neighbors of i in color c
    
    def init_state():
        col = [[0]*n for _ in range(n)]
        masks = [[0]*n for _ in range(num_colors)]
        for i in range(n):
            for j in range(i+1, n):
                c = random.randint(0, num_colors-1)
                col[i][j] = c
                col[j][i] = c
                masks[c][i] |= (1 << j)
                masks[c][j] |= (1 << i)
        return col, masks

    def calc_total_penalty(masks):
        total = 0
        for c in range(num_colors):
            m = masks[c]
            for i in range(n):
                for j in range(i+1, n):
                    common = m[i] & m[j]
                    k = bin(common).count('1')
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total

    def get_penalty_delta(u, v, c_old, c_new, col, masks):
        # This function calculates the change in penalty if we change edge (u,v) from c_old to c_new.
        # It assumes the masks are currently consistent with col.
        # We simulate the change, calculate delta, and return it.
        # We do NOT update the state here, just calculate the delta.
        
        delta = 0
        
        # Affected pairs: (u, x) and (v, x) for all x != u, v.
        # And the pair (u, v) itself?
        # The term for (u,v) is $\binom{|M_{c}[u] \cap M_{c}[v]|}{2}$.
        # Note: u is not in M_c[u]. v is not in M_c[v].
        # But u is in M_c[v] and v is in M_c[u] if edge (u,v) is color c.
        
        # Let's calculate the contribution of a specific pair (a,b) in color c.
        # Contrib(a,b,c) = C(|M_c[a] & M_c[b]|, 2)
        
        # Pairs to check:
        # 1. Pair (u, v) in c_old and c_new.
        # 2. Pairs (u, x) for x != u,v in c_old and c_new.
        # 3. Pairs (v, x) for x != u,v in c_old and c_new.
        
        # Helper to get contribution
        def contrib(a, b, c, masks):
            common = masks[c][a] & masks[c][b]
            k = bin(common).count('1')
            return k * (k - 1) // 2

        # Remove from c_old
        # We need to know the state BEFORE change.
        # Current state has edge (u,v) in c_old.
        
        # Delta from removing (u,v) from c_old:
        # Pairs involving u or v change.
        # Pair (u,v):
        # Before: M_c_old[u] contains v. M_c_old[v] contains u.
        # After: M_c_old'[u] does not contain v. M_c_old'[v] does not contain u.
        # Common neighbors of u and v in c_old:
        # Before: (M_c_old[u] & M_c_old[v]). Note v is in M[u], u is in M[v].
        # The intersection includes neighbors of u and v.
        # Let S = M_c_old[u] & M_c_old[v].
        # Note: u is not in M[u], v is not in M[v].
        # But v is in M[u] and u is in M[v].
        # So v is in S? M[u] has v. M[v] has u. v is not in M[v]. So v is NOT in M[u] & M[v]?
        # Wait. M[c][u] is the set of neighbors of u.
        # So v is in M[c][u].
        # Is v in M[c][v]? No, no self loops.
        # So v is NOT in the intersection M[c][u] & M[c][v].
        # Similarly, u is NOT in the intersection.
        # So the intersection is the set of common neighbors w such that (u,w) is c and (v,w) is c.
        
        # Let's verify:
        # M[u] & M[v] = { w | w!=u, w!=v, (u,w) is c, (v,w) is c }
        
        # So removing (u,v) from c_old:
        # M_c_old[u] loses v. M_c_old[v] loses u.
        # For pair (u,v):
        # Common neighbors before: S_old = M_c_old[u] & M_c_old[v].
        # Common neighbors after: S_new = (M_c_old[u] \ {v}) & (M_c_old[v] \ {u}).
        # Since v not in M_c_old[v] and u not in M_c_old[u], S_old = S_new.
        # So the contribution of pair (u,v) does NOT change when removing (u,v) from c_old?
        # Wait. The edge (u,v) itself is not a common neighbor.
        # The common neighbors are third vertices.
        # So yes, the count of common neighbors for pair (u,v) is unaffected by the existence of the edge (u,v) itself.
        # HOWEVER, the existence of edge (u,v) affects the common neighbors of other pairs?
        # No, M[u] and M[v] are the sets.
        # Changing M[u] and M[v] affects:
        # 1. Pairs (u, x) for x != v.
        # 2. Pairs (v, x) for x != u.
        # 3. Pair (u, v).
        
        # Let's re-evaluate Pair (u,v) contribution change.
        # Before: C(|M[u] & M[v]|, 2).
        # After: C(|(M[u]\{v}) & (M[v]\{u})|, 2).
        # As established, v not in M[v], u not in M[u].
        # So M[u] & M[v] = (M[u]\{v}) & (M[v]\{u}).
        # So contribution of (u,v) is UNCHANGED.
        
        # Now consider Pair (u, x) where x != u, v.
        # Before: C(|M[u] & M[x]|, 2).
        # After: C(|(M[u]\{v}) & M[x]|, 2).
        # The set M[u] & M[x] might contain v.
        # If v is in M[u] & M[x], it means (u,v) is c and (v,x) is c.
        # Since we are removing (u,v) from c, v is removed from M[u].
        # So if v was in the intersection, the count decreases by 1.
        # Let k_old = |M[u] & M[x]|.
        # Let k_new = |M[u]' & M[x]|.
        # k_new = k_old - 1 if v in M[u] & M[x], else k_old.
        # v in M[u] is True (before change).
        # v in M[x] is True if (v,x) is c.
        # So if (v,x) is c, k decreases by 1.
        
        # We can calculate this for all x.
        
        # This manual delta calculation is error-prone.
        # Given n=26, simply recalculating the full penalty for the affected colors is fast.
        # Or, even simpler:
        # Since we are doing many iterations, let's just use the "rebuild affected parts" logic.
        # But actually, for n=26, a full penalty calculation is ~3000 ops.
        # If we do 100k iterations, that's 300M ops. Might be too slow in Python.
        # We need the delta.
        
        # Let's implement the delta carefully.
        
        # Function to get count of common neighbors
        def common_count(a, b, c, masks):
            return bin(masks[c][a] & masks[c][b]).count('1')
        
        def penalty_term(k):
            return k * (k - 1) // 2
        
        # Delta for c_old (removal)
        # Pairs involving u: (u, x) for x in 0..n-1, x != u, x != v
        # Pairs involving v: (v, x) for x in 0..n-1, x != u, x != v
        # Pair (u, v) -> No change as proven.
        
        delta_c_old = 0
        for x in range(n):
            if x == u or x == v: continue
            
            # Pair (u, x)
            # Before: k_old = |M[u] & M[x]|
            # After: M[u] loses v.
            # v is in M[u] (True).
            # v is in M[x] if (v,x) is c_old.
            if masks[c_old][x] & (1 << v):
                k_old = common_count(u, x, c_old, masks)
                k_new = k_old - 1
                delta_c_old += penalty_term(k_new) - penalty_term(k_old)
            
            # Pair (v, x)
            # Before: k_old = |M[v] & M[x]|
            # After: M[v] loses u.
            # u is in M[v] (True).
            # u is in M[x] if (u,x) is c_old.
            if masks[c_old][x] & (1 << u):
                k_old = common_count(v, x, c_old, masks)
                k_new = k_old - 1
                delta_c_old += penalty_term(k_new) - penalty_term(k_old)
                
        # Delta for c_new (addition)
        # M[u] gains v. M[v] gains u.
        delta_c_new = 0
        for x in range(n):
            if x == u or x == v: continue
            
            # Pair (u, x)
            # Before: k_old = |M[u] & M[x]|
            # After: M[u] gains v.
            # v is NOT in M[u] (before).
            # v is in M[x] if (v,x) is c_new.
            if masks[c_new][x] & (1 << v):
                k_old = common_count(u, x, c_new, masks)
                k_new = k_old + 1
                delta_c_new += penalty_term(k_new) - penalty_term(k_old)
            
            # Pair (v, x)
            # Before: k_old = |M[v] & M[x]|
            # After: M[v] gains u.
            # u is NOT in M[v] (before).
            # u is in M[x] if (u,x) is c_new.
            if masks[c_new][x] & (1 << u):
                k_old = common_count(v, x, c_new, masks)
                k_new = k_old + 1
                delta_c_new += penalty_term(k_new) - penalty_term(k_old)
                
        return delta_c_old + delta_c_new

    # Main Search Loop
    best_sol = None
    
    # Try multiple restarts
    for restart in range(5):
        if time.time() > deadline: break
        
        col, masks = init_state()
        current_penalty = calc_total_penalty(masks)
        
        if current_penalty == 0:
            best_sol = col
            break
            
        T = 1.0
        no_improve = 0
        best_local_penalty = current_penalty
        best_local_col = [row[:] for row in col]
        
        while time.time() < deadline:
            if current_penalty == 0:
                best_sol = col
                break
                
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v: continue
            if u > v: u, v = v, u
            
            c_old = col[u][v]
            c_new = random.randint(0, num_colors-1)
            if c_new == c_old: continue
            
            delta = get_penalty_delta(u, v, c_old, c_new, col, masks)
            
            if delta < 0 or (delta > 0 and random.random() < np.exp(-delta / T)):
                # Accept
                col[u][v] = c_new
                col[v][u] = c_new
                
                # Update masks
                masks[c_old][u] &= ~(1 << v)
                masks[c_old][v] &= ~(1 << u)
                masks[c_new][u] |= (1 << v)
                masks[c_new][v] |= (1 << u)
                
                current_penalty += delta
                no_improve = 0
                if current_penalty < best_local_penalty:
                    best_local_penalty = current_penalty
                    best_local_col = [row[:] for row in col]
            else:
                no_improve += 1
                
            if no_improve > 1000:
                break
            T *= 0.999
            
        if best_local_penalty == 0:
            best_sol = best_local_col
            break
            
        # If we found a good local minimum, we might restart from it?
        # For now, just restart randomly.

    if best_sol is not None:
        res_colors = []
        for i in range(n):
            for j in range(i+1, n):
                res_colors.append(best_sol[i][j])
        return {'n': n, 'colors': res_colors}
    
    # Fallback: n=25
    # We can try to find a solution for n=25 quickly
    n = 25
    col, masks = init_state()
    current_penalty = calc_total_penalty(masks)
    T = 1.0
    for _ in range(50000):
        if current_penalty == 0: break
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        if u > v: u, v = v, u
        c_old = col[u][v]
        c_new = random.randint(0, num_colors-1)
        if c_new == c_old: continue
        delta = get_penalty_delta(u, v, c_old, c_new, col, masks)
        if delta < 0 or (delta > 0 and random.random() < np.exp(-delta / T)):
            col[u][v] = c_new
            col[v][u] = c_new
            masks[c_old][u] &= ~(1 << v)
            masks[c_old][v] &= ~(1 << u)
            masks[c_new][u] |= (1 << v)
            masks[c_new][v] |= (1 << u)
            current_penalty += delta
        T *= 0.999
        
    if current_penalty == 0:
        res_colors = []
        for i in range(n):
            for j in range(i+1, n):
                res_colors.append(col[i][j])
        return {'n': n, 'colors': res_colors}
        
    # Final fallback: n=11 (from seed)
    n_seed = 11
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    return {'n': n_seed, 'colors': colors_seed}
