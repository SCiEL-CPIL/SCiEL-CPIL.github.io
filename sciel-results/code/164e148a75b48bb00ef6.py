import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n = 20
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Bitmask utilities
    def iter_bits(x):
        while x:
            lsb = x & -x
            yield lsb.bit_length() - 1
            x ^= lsb

    best_edges = []
    best_score = -1
    
    # Run multiple restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        # Using p=0.5 gives a mix of valid/invalid, SA will fix.
        # Higher p might get stuck in high violation states.
        # Lower p might be too sparse to grow.
        p_init = 0.5 + 0.1 * random.random()
        
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p_init:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        
        # Calculate initial cn and violations
        cn = [[0] * n for _ in range(n)]
        violations = 0
        m = 0
        for u in range(n):
            for v in range(u + 1, n):
                common = adj_mask[u] & adj_mask[v]
                cnt = common.bit_count()
                cn[u][v] = cnt
                cn[v][u] = cnt
                if cnt > 2:
                    violations += cnt - 2
                if adj_mask[u] & (1 << v):
                    m += 1

        # SA Parameters
        T = 5.0
        cooling = 0.995
        min_T = 0.001
        steps_per_T = 50
        
        current_best_m = m
        current_best_violations = violations
        best_adj_mask_in_run = adj_mask[:]
        best_cn_in_run = [row[:] for row in cn]
        
        # We want to maximize m, subject to violations=0 ideally.
        # Score = m - 10 * violations.
        # If violations > 0, we heavily penalize.
        
        while T > min_T and time.time() < deadline:
            for _ in range(steps_per_T):
                if time.time() > deadline:
                    break
                    
                u, v = random.choice(all_pairs)
                is_edge = (adj_mask[u] >> v) & 1
                
                # Calculate delta m
                delta_m = 1 if not is_edge else -1
                
                # Calculate delta violations
                delta_v = 0
                # If adding edge (u,v):
                # cn[u][w] increases by 1 if v in N(w)
                # cn[v][w] increases by 1 if u in N(w)
                
                # We need to check all w != u, v
                # w is in N(v) if adj_mask[v] has bit w
                # w is in N(u) if adj_mask[u] has bit w
                
                # Note: When we flip, the masks change.
                # We must use the CURRENT masks to determine which w are affected.
                
                if not is_edge:
                    # Adding
                    # Affected w in N(v)
                    for w in iter_bits(adj_mask[v]):
                        if w == u: continue
                        # cn[u][w] will increase
                        old_cn = cn[u][w]
                        if old_cn >= 2:
                            delta_v += 1
                    # Affected w in N(u)
                    for w in iter_bits(adj_mask[u]):
                        if w == v: continue
                        # cn[v][w] will increase
                        old_cn = cn[v][w]
                        if old_cn >= 2:
                            delta_v += 1
                else:
                    # Removing
                    # Affected w in N(v)
                    for w in iter_bits(adj_mask[v]):
                        if w == u: continue
                        # cn[u][w] will decrease
                        old_cn = cn[u][w]
                        if old_cn > 2:
                            delta_v -= 1
                    # Affected w in N(u)
                    for w in iter_bits(adj_mask[u]):
                        if w == v: continue
                        # cn[v][w] will decrease
                        old_cn = cn[v][w]
                        if old_cn > 2:
                            delta_v -= 1
                
                delta_score = delta_m - 10.0 * delta_v
                
                accept = False
                if delta_score > 0:
                    accept = True
                else:
                    if T > 0:
                        if random.random() < math.exp(delta_score / T):
                            accept = True
                
                if accept:
                    # Update state
                    if not is_edge:
                        adj_mask[u] |= (1 << v)
                        adj_mask[v] |= (1 << u)
                        m += 1
                    else:
                        adj_mask[u] &= ~(1 << v)
                        adj_mask[v] &= ~(1 << u)
                        m -= 1
                    
                    # Update cn and violations
                    violations += delta_v
                    
                    if not is_edge:
                        # Adding
                        for w in iter_bits(adj_mask[v]): # Note: adj_mask[v] now includes u? No, we updated it.
                            # Wait, iter_bits(adj_mask[v]) will include u now.
                            # We skipped u in the delta calculation.
                            # We need to update cn[u][w] for w in N(v) \ {u}
                            # But we also need to update cn[u][u]? No, cn is for pairs.
                            pass
                        # Let's just re-calculate cn for affected pairs to be safe and simple?
                        # No, that's O(n^2).
                        # Let's do the update carefully.
                        
                        # The sets of w were calculated BEFORE the flip.
                        # Let's store the affected w's?
                        # Or just re-iterate the OLD masks?
                        # We already iterated them for delta_v.
                        # We can combine the loops.
                        
                        # Optimization: Do the update in the same loop as delta calculation?
                        # Yes, that's better.
                        pass
                    else:
                        pass
                    
                    # RE-DO: Combined Loop for Delta and Update
                    # This was inefficient above. Let's restart the inner logic structure in the code.
                    # For now, I will just re-compute cn for u and v rows to be safe, O(n).
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        common_uw = (adj_mask[u] & adj_mask[w]).bit_count()
                        common_vw = (adj_mask[v] & adj_mask[w]).bit_count()
                        
                        # Update cn
                        if cn[u][w] != common_uw:
                            # This is a sanity check, but we need to update violations
                            # Actually, we already updated violations with delta_v.
                            # We just need to update the cn matrix.
                            cn[u][w] = common_uw
                            cn[w][u] = common_uw
                            
                        if cn[v][w] != common_vw:
                            cn[v][w] = common_vw
                            cn[w][v] = common_vw
                            
                    # Update cn[u][v] and cn[v][u]
                    # Common neighbors of u and v do not change when flipping (u,v)?
                    # Wait.
                    # N(u) changes, N(v) changes.
                    # Common neighbors of u and v: w such that w~u and w~v.
                    # If we add (u,v), does w~u change? Yes, if w=v? No w!=v.
                    # If we add (u,v), w~u is unchanged for w!=v. w~v is unchanged for w!=u.
                    # So cn[u][v] is UNCHANGED.
                    
                    # Track best
                    current_score = m - 10 * violations
                    best_score_in_run = current_best_m - 10 * current_best_violations
                    if current_score > best_score_in_run:
                        current_best_m = m
                        current_best_violations = violations
                        best_adj_mask_in_run = adj_mask[:]
                        best_cn_in_run = [row[:] for row in cn]
                        
            T *= cooling
        
        # Greedy Polish
        if time.time() < deadline:
            fill_mask = best_adj_mask_in_run[:]
            fill_cn = best_cn_in_run[:]
            fill_v = current_best_violations
            fill_m = current_best_m
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for u, v in all_pairs:
                    if (fill_mask[u] >> v) & 1:
                        continue
                    
                    # Check if can add
                    can_add = True
                    # Check w in N(v)
                    for w in iter_bits(fill_mask[v]):
                        if w == u: continue
                        if fill_cn[u][w] >= 2:
                            can_add = False
                            break
                    if not can_add:
                        continue
                    # Check w in N(u)
                    for w in iter_bits(fill_mask[u]):
                        if w == v: continue
                        if fill_cn[v][w] >= 2:
                            can_add = False
                            break
                    if not can_add:
                        continue
                    
                    # Add
                    fill_mask[u] |= (1 << v)
                    fill_mask[v] |= (1 << u)
                    fill_m += 1
                    
                    # Update cn
                    for w in iter_bits(fill_mask[v]):
                        if w == u: continue
                        fill_cn[u][w] += 1
                        fill_cn[w][u] += 1
                        if fill_cn[u][w] > 2:
                            fill_v += 1
                            
                    for w in iter_bits(fill_mask[u]):
                        if w == v: continue
                        fill_cn[v][w] += 1
                        fill_cn[w][v] += 1
                        if fill_cn[v][w] > 2:
                            fill_v += 1
                            
                    improved = True
            
            # Verify and update global best
            if fill_v == 0:
                # Count edges
                final_m = sum(fill_mask[u].bit_count() for u in range(n)) // 2
                if final_m > best_score:
                    best_score = final_m
                    best_edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (fill_mask[u] >> v) & 1:
                                best_edges.append([u, v])
            else:
                # If not valid, we shouldn't use it unless it's better than current best?
                # The task requires a VALID graph.
                # So we must only update best_edges if valid.
                pass

    # Fallback
    if not best_edges:
        # Return a known valid small graph?
        # Empty graph is valid.
        return []
        
    return best_edges
