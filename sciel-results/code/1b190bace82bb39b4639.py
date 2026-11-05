import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def popcount(x):
        return bin(x).count('1')
        
    def get_violation(masks):
        """Find one K3,3 violation or return None."""
        for i in range(n):
            for j in range(i+1, n):
                common_ij = masks[i] & masks[j]
                if popcount(common_ij) < 3:
                    continue
                for k in range(j+1, n):
                    common_ijk = common_ij & masks[k]
                    if popcount(common_ijk) >= 3:
                        return (i, j, k, common_ijk)
        return None

    def is_valid(masks):
        return get_violation(masks) is None

    def try_add_edge(masks, u, v):
        """
        Try to add edge (u,v).
        Returns True if valid, False otherwise.
        Uses incremental check: only triples involving u or v need checking.
        """
        # Tentatively add
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
        
        # Check triples involving u or v
        # Triples with u: (u, a, b) where a, b in 0..19, a != u, b != u, a < b
        # Triples with v: (v, a, b) where a, b in 0..19, a != v, b != v, a < b
        # Note: (u, v, k) is counted in both?
        # We can just iterate all triples containing u or v.
        
        valid = True
        
        # Check triples containing u
        for a in range(n):
            if a == u: continue
            for b in range(a+1, n):
                if b == u: continue
                # Triple is (min, mid, max) for consistency, but intersection is commutative
                # We just need to check the set {u, a, b}
                # Ensure we don't double check or miss any.
                # Simple approach: iterate all pairs (a,b) from remaining 19 vertices.
                # There are 171 such pairs.
                pass
        
        # More efficient:
        # Iterate a from 0 to n-1
        # Iterate b from a+1 to n-1
        # If u in {a,b} or v in {a,b}, check.
        # Actually, just iterate all pairs (a,b) with a < b.
        # If a == u or b == u or a == v or b == v:
        #    k is the third vertex? No, the triple is (u, a, b) where a,b are the other two.
        #    So for each pair (a,b) from the 19 vertices excluding u:
        #       triple = (u, a, b)
        #    For each pair (a,b) from the 19 vertices excluding v:
        #       triple = (v, a, b)
        #    We might check (u, v, k) twice?
        #    (u, v, k) is in "triples with u" (pair v,k) and "triples with v" (pair u,k).
        #    It's fine to check twice, or we can be careful.
        
        # Let's just do:
        # For all pairs (a, b) with a < b from 0..19:
        #   If u in {a,b} or v in {a,b}:
        #      Let the third vertex be c.
        #      If the triple is (u, a, b) where c=u, then a,b are the others.
        #      If the triple is (v, a, b) where c=v, then a,b are the others.
        #      If the triple is (u, v, c), it is covered by both.
        
        # Simpler:
        # List of vertices excluding u: V_u
        # For each pair (a, b) in V_u:
        #    Check triple (u, a, b)
        # List of vertices excluding v: V_v
        # For each pair (a, b) in V_v:
        #    Check triple (v, a, b)
        #    But skip if a==u and b==... wait.
        #    If we check (v, u, c), that's the same as (u, v, c).
        #    We can just check all pairs from V_u and all pairs from V_v.
        #    Total ~ 342 checks.
        
        V_u = [x for x in range(n) if x != u]
        for idx_a in range(len(V_u)):
            a = V_u[idx_a]
            for idx_b in range(idx_a + 1, len(V_u)):
                b = V_u[idx_b]
                # Triple (u, a, b)
                common = masks[u] & masks[a] & masks[b]
                if popcount(common) >= 3:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            V_v = [x for x in range(n) if x != v]
            for idx_a in range(len(V_v)):
                a = V_v[idx_a]
                for idx_b in range(idx_a + 1, len(V_v)):
                    b = V_v[idx_b]
                    # Triple (v, a, b)
                    # Optimization: if a==u and b==u? No, a!=b.
                    # If {a,b} == {u, c}, then triple is (v, u, c).
                    # This was already checked in the first loop as (u, v, c) if v was in V_u?
                    # V_u contains v. So (u, v, c) was checked in the first loop.
                    # So we can skip if a == u or b == u?
                    # If a == u, triple is (v, u, b) = (u, v, b). Checked in loop 1.
                    # If b == u, triple is (v, a, u) = (u, v, a). Checked in loop 1.
                    if a == u or b == u:
                        continue
                    common = masks[v] & masks[a] & masks[b]
                    if popcount(common) >= 3:
                        valid = False
                        break
                if not valid:
                    break

        if not valid:
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
            return False
        
        return True

    def try_remove_edge(masks, u, v):
        masks[u] &= ~(1 << v)
        masks[v] &= ~(1 << u)

    best_edges = 0
    best_masks = [0] * n
    
    # Multiple restarts
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        # Initialize
        masks = [0] * n
        # Random dense start
        p = 0.8
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
                    
        # Repair
        max_repair = 100
        for _ in range(max_repair):
            if time.time() > deadline - 5:
                break
            viol = get_violation(masks)
            if viol is None:
                break
            i, j, k, common_mask = viol
            neighbors = []
            m = common_mask
            while len(neighbors) < 3 and m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                neighbors.append(idx)
                m ^= lsb
            u = random.choice([i, j, k])
            v = random.choice(neighbors)
            try_remove_edge(masks, u, v)
            
        if not is_valid(masks):
            continue
            
        current_edges = sum(bin(m).count('1') for m in masks) // 2
        if current_edges > best_edges:
            best_edges = current_edges
            best_masks = masks[:]
            
        # SA Phase
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999
        max_sa_steps = 20000
        
        for step in range(max_sa_steps):
            if time.time() > deadline - 5:
                break
            if T < T_min:
                break
                
            move_type = random.random()
            
            if move_type < 0.4:
                # Add edge
                missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                          if not (masks[i] & (1 << j))]
                if not missing:
                    break
                u, v = random.choice(missing)
                if try_add_edge(masks, u, v):
                    new_edges = sum(bin(m).count('1') for m in masks) // 2
                    delta = new_edges - current_edges
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        current_edges = new_edges
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_masks = masks[:]
                    else:
                        # Revert (try_add_edge already reverted if invalid, 
                        # but if valid and we reject due to T, we need to revert manually)
                        # Wait, try_add_edge returns True if valid and leaves edge added.
                        # So if we reject, we must remove it.
                        try_remove_edge(masks, u, v)
                # If try_add_edge returned False, it already reverted.
                
            elif move_type < 0.7:
                # Remove edge
                existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                           if (masks[i] & (1 << j))]
                if not existing:
                    break
                u, v = random.choice(existing)
                try_remove_edge(masks, u, v)
                new_edges = sum(bin(m).count('1') for m in masks) // 2
                delta = new_edges - current_edges
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current_edges = new_edges
                    if current_edges > best_edges:
                        best_edges = current_edges
                        best_masks = masks[:]
                else:
                    # Re-add
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                    
            else:
                # Swap
                existing = [(i, j) for i in range(n) for j in range(i+1, n) 
                           if (masks[i] & (1 << j))]
                missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                          if not (masks[i] & (1 << j))]
                if not existing or not missing:
                    break
                u1, v1 = random.choice(existing)
                u2, v2 = random.choice(missing)
                if (u1, v1) == (u2, v2) or (u1, v1) == (v2, u2):
                    continue
                    
                # Remove u1,v1
                try_remove_edge(masks, u1, v1)
                # Try add u2,v2
                if try_add_edge(masks, u2, v2):
                    new_edges = sum(bin(m).count('1') for m in masks) // 2
                    delta = new_edges - current_edges
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        current_edges = new_edges
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_masks = masks[:]
                    else:
                        # Revert: Remove u2,v2 and Re-add u1,v1
                        try_remove_edge(masks, u2, v2)
                        masks[u1] |= (1 << v1)
                        masks[v1] |= (1 << u1)
                else:
                    # try_add_edge reverted u2,v2. Re-add u1,v1.
                    masks[u1] |= (1 << v1)
                    masks[v1] |= (1 << u1)
                    
            T *= cooling_rate
            
        # Local Search Polish
        for i in range(n):
            for j in range(i+1, n):
                if time.time() > deadline - 3:
                    break
                if not (masks[i] & (1 << j)):
                    if try_add_edge(masks, i, j):
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_masks = masks[:]

    edges = []
    for i in range(n):
        m = best_masks[i]
        while m:
            lsb = m & -m
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            m ^= lsb
            
    return edges
