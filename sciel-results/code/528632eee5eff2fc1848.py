import time
import random
from itertools import combinations

def _raw_value(values, edges):
    """Compute max(distinct_sums, distinct_products) for given values and edges."""
    sums = set()
    prods = set()
    for i, j in edges:
        s = values[i] + values[j]
        p = values[i] * values[j]
        sums.add(s)
        prods.add(p)
    return max(len(sums), len(prods))

def _fast_edge_optimize(values, m=35, deadline=None, num_restarts=15):
    """Fast edge optimization using bitmask-based collision tracking."""
    n = len(values)
    all_edges = list(combinations(range(n), 2))
    num_all = len(all_edges)
    
    edge_sums = [0] * num_all
    edge_prods = [0] * num_all
    for idx, (i, j) in enumerate(all_edges):
        edge_sums[idx] = values[i] + values[j]
        edge_prods[idx] = values[i] * values[j]
    
    # Map sums and products to bit indices
    all_sums = sorted(set(edge_sums))
    all_prods = sorted(set(edge_prods))
    sum_to_bit = {s: i for i, s in enumerate(all_sums)}
    prod_to_bit = {p: i for i, p in enumerate(all_prods)}
    
    edge_sum_bits = [1 << sum_to_bit[edge_sums[idx]] for idx in range(num_all)]
    edge_prod_bits = [1 << prod_to_bit[edge_prods[idx]] for idx in range(num_all)]
    
    best_score = float('inf')
    best_edges = None
    
    for restart in range(num_restarts):
        if deadline and time.time() > deadline:
            break
        
        # Greedy init: pick edges that maximize collisions
        current = set()
        cur_sum_mask = 0
        cur_prod_mask = 0
        candidates = list(range(num_all))
        random.shuffle(candidates)
        
        for _ in range(m):
            best_idx = -1
            best_key = None
            for idx in candidates:
                if idx in current:
                    continue
                new_sum_mask = cur_sum_mask | edge_sum_bits[idx]
                new_prod_mask = cur_prod_mask | edge_prod_bits[idx]
                ns = new_sum_mask.bit_count()
                np_ = new_prod_mask.bit_count()
                key = (max(ns, np_), ns + np_)
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = idx
            if best_idx == -1:
                break
            current.add(best_idx)
            cur_sum_mask |= edge_sum_bits[best_idx]
            cur_prod_mask |= edge_prod_bits[best_idx]
            candidates.remove(best_idx)
        
        if len(current) < m:
            continue
        
        cur_score = max(cur_sum_mask.bit_count(), cur_prod_mask.bit_count())
        
        # Local search with swaps using bitmask operations
        improved = True
        max_iter = 80
        it = 0
        while improved and it < max_iter and (not deadline or time.time() < deadline):
            improved = False
            it += 1
            cur_list = list(current)
            outside = [x for x in range(num_all) if x not in current]
            random.shuffle(cur_list)
            random.shuffle(outside)
            
            for e_in in cur_list:
                if deadline and time.time() > deadline:
                    break
                s_bit = edge_sum_bits[e_in]
                p_bit = edge_prod_bits[e_in]
                
                # Count how many edges share this sum/product bit
                sum_count = 0
                prod_count = 0
                for idx in current:
                    if edge_sum_bits[idx] == s_bit:
                        sum_count += 1
                    if edge_prod_bits[idx] == p_bit:
                        prod_count += 1
                
                for e_out in outside:
                    s_out_bit = edge_sum_bits[e_out]
                    p_out_bit = edge_prod_bits[e_out]
                    
                    # Simulate removal of e_in
                    trial_sum_mask = cur_sum_mask
                    trial_prod_mask = cur_prod_mask
                    if sum_count == 1:
                        trial_sum_mask &= ~s_bit
                    if prod_count == 1:
                        trial_prod_mask &= ~p_bit
                    
                    # Simulate addition of e_out
                    trial_sum_mask |= s_out_bit
                    trial_prod_mask |= p_out_bit
                    
                    trial_score = max(trial_sum_mask.bit_count(), trial_prod_mask.bit_count())
                    
                    if trial_score < cur_score:
                        current.discard(e_in)
                        current.add(e_out)
                        cur_sum_mask = trial_sum_mask
                        cur_prod_mask = trial_prod_mask
                        cur_score = trial_score
                        improved = True
                        if cur_score < best_score:
                            best_score = cur_score
                            best_edges = [list(all_edges[idx]) for idx in current]
                        break
                if improved:
                    break
        
        if cur_score < best_score:
            best_score = cur_score
            best_edges = [list(all_edges[idx]) for idx in current]
    
    return best_edges, best_score

def _optimize_edges(values, m=35, deadline=None):
    """Optimize edge selection for given values."""
    return _fast_edge_optimize(values, m, deadline)

def _targeted_collision_move(vals, edges, deadline):
    """Try targeted moves that set a value to create sum/product collisions."""
    n = len(vals)
    best_improvement = None
    best_new_vals = None
    best_new_raw = float('inf')
    
    # For each vertex, try to set its value so that some edge sums/products collide
    for vi in range(n):
        if deadline and time.time() > deadline:
            break
        # Find neighbors of vi
        neighbors = set()
        for i, j in edges:
            if i == vi:
                neighbors.add(j)
            elif j == vi:
                neighbors.add(i)
        
        # Try to make sums collide: for pair (vi, nj), want vals[vi]+vals[nj] = existing sum
        # Or make products collide: vals[vi]*vals[nj] = existing product
        
        # Collect existing sums and products
        existing_sums = set()
        existing_prods = set()
        for i, j in edges:
            existing_sums.add(vals[i] + vals[j])
            existing_prods.add(vals[i] * vals[j])
        
        # For each neighbor, try values that would create a collision
        candidates = set()
        for nj in neighbors:
            vj = vals[nj]
            # Sum collision: vals[vi] = s - vj for some existing sum s
            for s in existing_sums:
                cand = s - vj
                if 1 <= cand <= 10000 and cand != vals[vi] and cand not in vals:
                    candidates.add(cand)
            # Product collision: vals[vi] = p / vj for some existing product p
            for p in existing_prods:
                if p % vj == 0:
                    cand = p // vj
                    if 1 <= cand <= 10000 and cand != vals[vi] and cand not in vals:
                        candidates.add(cand)
        
        # Also try small perturbations
        for delta in range(-5, 6):
            cand = vals[vi] + delta
            if 1 <= cand <= 10000 and cand not in vals:
                candidates.add(cand)
        
        for cand in list(candidates)[:20]:  # Limit candidates for speed
            if deadline and time.time() > deadline:
                break
            trial_vals = list(vals)
            trial_vals[vi] = cand
            if len(set(trial_vals)) != 14:
                continue
            
            # Quick evaluation with current edges
            sums = set()
            prods = set()
            for i, j in edges:
                sums.add(trial_vals[i] + trial_vals[j])
                prods.add(trial_vals[i] * trial_vals[j])
            trial_raw = max(len(sums), len(prods))
            
            if trial_raw < best_new_raw:
                best_new_raw = trial_raw
                best_new_vals = trial_vals
    
    return best_new_vals, best_new_raw

def _polish(values, edges, deadline, max_rounds=5):
    """Dedicated polish phase with alternating edge and value local search."""
    vals = list(values)
    edg = [list(e) for e in edges]
    
    for rnd in range(max_rounds):
        if deadline and time.time() > deadline:
            break
        
        # Edge polish
        new_edges, raw = _fast_edge_optimize(vals, 35, deadline, num_restarts=20)
        if new_edges and raw < _raw_value(vals, edg):
            edg = new_edges
        
        # Value polish: targeted collision moves
        new_vals, raw2 = _targeted_collision_move(vals, edg, deadline)
        if new_vals and raw2 < _raw_value(vals, edg):
            vals = new_vals
            # Re-optimize edges for new values
            new_edges, raw = _fast_edge_optimize(vals, 35, deadline, num_restarts=10)
            if new_edges:
                edg = new_edges
        
        # Random value perturbation
        trial_vals = list(vals)
        idxs = random.sample(range(14), random.randint(1, 3))
        for idx in idxs:
            delta = random.choice([-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            new_v = trial_vals[idx] + delta
            if 1 <= new_v <= 10000 and new_v not in trial_vals:
                trial_vals[idx] = new_v
        if len(set(trial_vals)) == 14:
            t_edges, t_raw = _fast_edge_optimize(trial_vals, 35, deadline, num_restarts=5)
            if t_edges and t_raw < _raw_value(vals, edg):
                vals = trial_vals
                edg = t_edges
    
    return {'values': vals, 'edges': edg}, _raw_value(vals, edg)

def _joint_optimize(start_values, deadline, max_rounds=8):
    """Jointly optimize values and edges through iterative perturbation with polish."""
    best_raw = float('inf')
    best_vals = None
    best_edges = None
    
    vals = list(start_values)
    
    for rnd in range(max_rounds):
        if deadline and time.time() > deadline:
            break
        
        # Optimize edges for current values
        edges, raw = _fast_edge_optimize(vals, 35, deadline, num_restarts=12)
        
        if edges and raw < best_raw:
            best_raw = raw
            best_vals = list(vals)
            best_edges = [list(e) for e in edges]
        
        # Targeted collision-seeking value moves
        new_vals, t_raw = _targeted_collision_move(vals, edges if edges else [], deadline)
        if new_vals and t_raw < best_raw:
            best_raw = t_raw
            best_vals = list(new_vals)
            t_edges, t_raw2 = _fast_edge_optimize(new_vals, 35, deadline, num_restarts=8)
            if t_edges and t_raw2 < best_raw:
                best_raw = t_raw2
                best_edges = [list(e) for e in t_edges]
            else:
                best_edges = [list(e) for e in edges] if edges else best_edges
            vals = new_vals
        elif edges:
            vals = best_vals if best_vals else vals
        
        # Random perturbation
        for _ in range(2):
            if deadline and time.time() > deadline:
                break
            trial_vals = list(vals)
            idxs = random.sample(range(14), random.randint(1, 3))
            for idx in idxs:
                delta = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12])
                new_v = trial_vals[idx] + delta
                if 1 <= new_v <= 10000 and new_v not in trial_vals:
                    trial_vals[idx] = new_v
            
            if len(set(trial_vals)) != 14:
                continue
            
            t_edges, t_raw = _fast_edge_optimize(trial_vals, 35, deadline, num_restarts=6)
            if t_edges and t_raw < best_raw:
                best_raw = t_raw
                best_vals = list(trial_vals)
                best_edges = [list(e) for e in t_edges]
                vals = trial_vals
    
    # Final polish phase
    if best_vals and deadline and time.time() < deadline:
        polished, pol_raw = _polish(best_vals, best_edges, deadline, max_rounds=4)
        if pol_raw < best_raw:
            best_raw = pol_raw
            best_vals = polished['values']
            best_edges = polished['edges']
    
    if best_vals is None:
        return None, float('inf')
    return {'values': best_vals, 'edges': best_edges}, best_raw

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin (90s limit, 8s buffer)
    
    best_result = None
    best_raw = float('inf')
    
    # Structured value sets designed for collision promotion
    value_sets = [
        # Small numbers with many additive/multiplicative relations
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20],
        # Numbers with many divisors (promote product collisions)
        [1, 2, 3, 4, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24],
        # Consecutive
        list(range(1, 15)),
        # Mixed small
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20],
        # Powers and composites
        [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36],
        # Highly composite numbers
        [1, 2, 3, 4, 6, 8, 12, 16, 18, 24, 30, 36, 48, 60],
        # Geometric-ish
        [1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 72, 96, 144],
    ]
    
    for vals in value_sets:
        if deadline and time.time() > deadline:
            break
        if len(set(vals)) != 14 or any(v <= 0 or v > 10000 for v in vals):
            continue
        
        # Edge optimization
        edges, raw = _fast_edge_optimize(vals, 35, deadline, num_restarts=10)
        if edges and raw < best_raw:
            best_raw = raw
            best_result = {'values': list(vals), 'edges': [list(e) for e in edges]}
    
    # Joint optimization from best found so far
    if best_result and deadline and time.time() < deadline:
        jr, jr_raw = _joint_optimize(best_result['values'], deadline, max_rounds=6)
        if jr and jr_raw < best_raw:
            best_raw = jr_raw
            best_result = jr
    
    # Additional joint optimization from other starting points if time permits
    if deadline and time.time() < deadline - 10:
        for vals in value_sets[1:3]:
            if deadline and time.time() > deadline:
                break
            jr, jr_raw = _joint_optimize(vals, deadline, max_rounds=4)
            if jr and jr_raw < best_raw:
                best_raw = jr_raw
                best_result = jr
    
    # Final polish if time permits
    if best_result and deadline and time.time() < deadline - 3:
        polished, pol_raw = _polish(best_result['values'], best_result['edges'], deadline, max_rounds=3)
        if pol_raw < best_raw:
            best_raw = pol_raw
            best_result = polished
    
    if best_result is None:
        vals = list(range(1, 15))
        edges = list(combinations(range(14), 2))[:35]
        best_result = {'values': vals, 'edges': edges}
    
    # Format output
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(i), int(j)] for i, j in best_result['edges']]
    
    return best_result
