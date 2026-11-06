import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin 10s
    
    def get_seed_graph():
        q = 37
        residues = {(x * x) % q for x in range(1, q)}
        edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
        return q, edges

    best_n, best_edges = get_seed_graph()
    
    # Targets to try, descending
    targets = [42, 41, 40]
    
    for target_n in targets:
        if time.time() > deadline:
            break
        
        time_budget = deadline - time.time()
        if time_budget < 10:
            break
            
        found = solve_for_n(target_n, time_budget)
        if found:
            n, edges = found
            if n > best_n:
                best_n, best_edges = n, edges
                if n >= 42:
                    break
    return best_n, best_edges

def solve_for_n(n, time_budget):
    # Precompute all 5-combinations for fast checking
    all_combos = list(itertools.combinations(range(n), 5))
    
    # Precompute for each vertex, which combos it appears in
    vertex_to_combos = [[] for _ in range(n)]
    for idx, combo in enumerate(all_combos):
        for v in combo:
            vertex_to_combos[v].append(idx)
            
    # We will try multiple restarts with different seeds and densities
    # Use a range of densities around 0.5
    densities = [0.5, 0.48, 0.52, 0.45, 0.55]
    
    # Estimate number of restarts we can afford
    # Each restart involves local search. Let's say ~50ms per restart step? 
    # Actually, local search can run for seconds.
    # Let's allocate time per restart.
    
    base_seed = int(time.time() * 1000) % 100000
    
    for r in range(20): # Max 20 restarts
        if time.time() > time_budget:
            break
            
        seed = base_seed + r * 12345
        random.seed(seed)
        np.random.seed(seed)
        
        density = densities[r % len(densities)]
        
        # Initialize random graph
        adj = [0] * n
        num_pairs = n * (n - 1) // 2
        random_bits = (np.random.rand(num_pairs) < density).astype(int).flatten()
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if random_bits[idx]:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                idx += 1
        
        # Count initial violations
        violations = count_violations(n, adj, all_combos)
        
        # Local search loop
        # We use a "hill climbing" approach with occasional random kicks
        max_steps = 2000
        stuck_counter = 0
        best_violations = violations
        
        for step in range(max_steps):
            if time.time() > time_budget:
                return None
            
            if violations == 0:
                # Success
                edges_list = []
                for i in range(n):
                    mask = adj[i] & ~((1 << (i + 1)) - 1)
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        edges_list.append((i, j))
                        mask ^= lsb
                return n, edges_list
            
            # Find a violation
            combo, type_v = find_one_violation_fast(n, adj, all_combos)
            if not combo:
                # Should be same as violations == 0, but just in case
                edges_list = []
                for i in range(n):
                    mask = adj[i] & ~((1 << (i + 1)) - 1)
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        edges_list.append((i, j))
                        mask ^= lsb
                return n, edges_list
            
            # Try to fix the violation by flipping an edge within the combo
            pairs = list(itertools.combinations(combo, 2))
            random.shuffle(pairs)
            
            best_flip = None
            best_new_violations = violations
            
            # Try a few flips
            for u, v in pairs[:5]:
                has_edge = (adj[u] >> v) & 1
                
                # Apply flip
                if has_edge:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                # Count new violations
                new_violations = count_violations_partial(n, adj, all_combos, vertex_to_combos, [u, v])
                
                # Undo flip
                if has_edge:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                
                if new_violations < best_new_violations:
                    best_new_violations = new_violations
                    best_flip = (u, v, has_edge)
            
            if best_flip:
                u, v, was_edge = best_flip
                if was_edge:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                violations = best_new_violations
                
                if violations < best_violations:
                    best_violations = violations
                    stuck_counter = 0
                else:
                    stuck_counter += 1
            else:
                # No improving flip found in the violation
                # Try a random flip to escape
                u, v = random.sample(range(n), 2)
                if u > v:
                    u, v = v, u
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                violations = count_violations(n, adj, all_combos)
                stuck_counter += 1
                
                # If stuck too long, restart this local search instance
                if stuck_counter > 50:
                    break
                    
    return None

def count_violations(n, adj, all_combos):
    count = 0
    for combo in all_combos:
        a, b, c, d, e = combo
        mask_b_c_d_e = (1 << b) | (1 << c) | (1 << d) | (1 << e)
        
        # Check K5
        if (adj[a] & mask_b_c_d_e) == mask_b_c_d_e:
            mask_c_d_e = (1 << c) | (1 << d) | (1 << e)
            if (adj[b] & mask_c_d_e) == mask_c_d_e:
                mask_d_e = (1 << d) | (1 << e)
                if (adj[c] & mask_d_e) == mask_d_e:
                    if (adj[d] >> e) & 1:
                        count += 1
                        continue
        
        # Check IK5
        if (adj[a] & mask_b_c_d_e) == 0:
            mask_c_d_e = (1 << c) | (1 << d) | (1 << e)
            if (adj[b] & mask_c_d_e) == 0:
                mask_d_e = (1 << d) | (1 << e)
                if (adj[c] & mask_d_e) == 0:
                    if (adj[d] & (1 << e)) == 0:
                        count += 1
    return count

def count_violations_partial(n, adj, all_combos, vertex_to_combos, changed_vertices):
    """Count violations only for combos involving at least one changed vertex"""
    count = 0
    checked = set()
    for v in changed_vertices:
        for combo_idx in vertex_to_combos[v]:
            if combo_idx in checked:
                continue
            checked.add(combo_idx)
            combo = all_combos[combo_idx]
            a, b, c, d, e = combo
            mask_b_c_d_e = (1 << b) | (1 << c) | (1 << d) | (1 << e)
            
            # Check K5
            if (adj[a] & mask_b_c_d_e) == mask_b_c_d_e:
                mask_c_d_e = (1 << c) | (1 << d) | (1 << e)
                if (adj[b] & mask_c_d_e) == mask_c_d_e:
                    mask_d_e = (1 << d) | (1 << e)
                    if (adj[c] & mask_d_e) == mask_d_e:
                        if (adj[d] >> e) & 1:
                            count += 1
                            continue
            
            # Check IK5
            if (adj[a] & mask_b_c_d_e) == 0:
                mask_c_d_e = (1 << c) | (1 << d) | (1 << e)
                if (adj[b] & mask_c_d_e) == 0:
                    mask_d_e = (1 << d) | (1 << e)
                    if (adj[c] & mask_d_e) == 0:
                        if (adj[d] & (1 << e)) == 0:
                            count += 1
    return count

def find_one_violation_fast(n, adj, all_combos):
    for combo in all_combos:
        a, b, c, d, e = combo
        mask_b_c_d_e = (1 << b) | (1 << c) | (1 << d) | (1 << e)
        
        if (adj[a] & mask_b_c_d_e) == mask_b_c_d_e:
            mask_c_d_e = (1 << c) | (1 << d) | (1 << e)
            if (adj[b] & mask_c_d_e) == mask_c_d_e:
                mask_d_e = (1 << d) | (1 << e)
                if (adj[c] & mask_d_e) == mask_d_e:
                    if (adj[d] >> e) & 1:
                        return combo, 'K5'
        
        if (adj[a] & mask_b_c_d_e) == 0:
            mask_c_d_e = (1 << c) | (1 << d) | (1 << e)
            if (adj[b] & mask_c_d_e) == 0:
                mask_d_e = (1 << d) | (1 << e)
                if (adj[c] & mask_d_e) == 0:
                    if (adj[d] & (1 << e)) == 0:
                        return combo, 'IK5'
    return None, None
