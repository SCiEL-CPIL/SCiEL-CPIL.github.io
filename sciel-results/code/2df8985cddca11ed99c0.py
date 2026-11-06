"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 256
    vertices = list(range(2, 257))
    n = len(vertices)
    
    # 1. Precompute conflict graph
    # adj[i] is a bitmask of indices j such that lcm(vertices[i], vertices[j]) <= 256
    adj = [0] * n
    for i in range(n):
        vi = vertices[i]
        for j in range(i + 1, n):
            vj = vertices[j]
            g = gcd(vi, vj)
            if (vi // g) * vj <= 256:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 2. Precompute weights
    weights = [1.0 / v for v in vertices]
    
    # Precompute sorted indices by weight descending
    order_by_weight = sorted(range(n), key=lambda i: -weights[i])
    
    # 3. Greedy Initialization
    def greedy_init(exclude_mask=0, sort_key='weight_desc', rng=None):
        if sort_key == 'weight_desc':
            order = order_by_weight
        elif sort_key == 'degree_asc':
            degrees = [bin(adj[i]).count('1') for i in range(n)]
            order = sorted(range(n), key=lambda i: (degrees[i], -weights[i]))
        elif sort_key == 'random':
            order = list(range(n))
            if rng:
                rng.shuffle(order)
            else:
                random.shuffle(order)
        else:
            order = list(range(n))
            
        selected_mask = 0
        current_sum = 0.0
        available_mask = ((1 << n) - 1) & ~exclude_mask
        
        for i in order:
            if not (available_mask & (1 << i)):
                continue
            selected_mask |= (1 << i)
            current_sum += weights[i]
            available_mask &= ~(adj[i] | (1 << i))
            
        return selected_mask, current_sum

    best_mask = 0
    best_sum = 0.0
    
    # Try several initial configurations
    m, s = greedy_init(exclude_mask=0, sort_key='weight_desc')
    if s > best_sum:
        best_sum = s
        best_mask = m
        
    m, s = greedy_init(exclude_mask=(1 << 0), sort_key='weight_desc')
    if s > best_sum:
        best_sum = s
        best_mask = m
        
    m, s = greedy_init(exclude_mask=((1 << 0) | (1 << 1)), sort_key='weight_desc')
    if s > best_sum:
        best_sum = s
        best_mask = m

    m, s = greedy_init(exclude_mask=0, sort_key='degree_asc')
    if s > best_sum:
        best_sum = s
        best_mask = m

    # 4. Local Search with 1-swap and 2-swap moves
    def get_selected_indices(mask):
        return [i for i in range(n) if (mask >> i) & 1]
    
    def can_add(mask, i):
        if (mask >> i) & 1:
            return False
        return (adj[i] & mask) == 0

    def local_search(start_mask, start_sum, max_iters=20000, rng=None):
        current_mask = start_mask
        current_sum = start_sum
        improved = True
        iters = 0
        
        # Precompute neighbor union for the current mask? 
        # We need union of adj[k] for k in mask. Let's call it 'conflict_mask'.
        # Initially:
        conflict_mask = 0
        for i in get_selected_indices(current_mask):
            conflict_mask |= adj[i] | (1 << i)
        
        while improved and iters < max_iters and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            
            # Phase 1: Try to add any vertex that fits
            # Vertices that can be added are those in ((1<<n)-1) & ~conflict_mask
            # We want the one with max weight.
            # Iterate through sorted by weight descending? Or just check all?
            # Since n=255, iterating all is fine.
            added = -1
            best_w = 0.0
            for i in order_by_weight:
                if not (conflict_mask & (1 << i)):
                    best_w = weights[i]
                    added = i
                    break
            
            if added != -1:
                current_mask |= (1 << added)
                current_sum += weights[added]
                conflict_mask |= (adj[added] | (1 << added))
                improved = True
                continue # Restart to check for more adds
            
            # Phase 2: Try 1-swap moves
            # Remove i, add j.
            # Gain = weight[j] - weight[i] + (sum of weights of new adds unlocked by removing i) - (sum of weights of adds lost by removing i? No, adding i only locks, removing i unlocks)
            # Wait, if we remove i, we might be able to add more than just j.
            # Simple 1-swap: remove i, add j. 
            # But removing i might allow adding multiple new vertices.
            # Let's just try: remove i, then greedily add as many as possible, see if sum increases.
            # This is expensive. 
            # Alternative: Just check if there's a j such that weight[j] > weight[i] and j is compatible with mask without i.
            # This is a heuristic.
            
            selected = get_selected_indices(current_mask)
            best_swap_gain = 0.0
            best_i = -1
            best_j = -1
            
            for i in selected:
                mask_without_i = current_mask & ~(1 << i)
                # Recalculate conflict mask for mask_without_i
                # conflict_mask_without_i = conflict_mask without adj[i]? No, it's the union of adj[k] for k in mask_without_i.
                # This is expensive to compute from scratch.
                # Optimization: 
                # conflict_mask is union of adj[k] for k in selected.
                # When we remove i, we lose adj[i] from the union? No, other nodes might conflict with those nodes.
                # Actually, conflict_mask = union of (adj[k] | (1<<k)) for k in selected.
                # To update, we'd need to know which nodes in adj[i] are NOT covered by other adj[k].
                # This is complex.
                
                # Since n=255, recomputing conflict_mask for each i in swap phase is O(n^2 * bitops).
                # selected size is small (~20-30).
                # So O(30 * 255 * bitops) is fast.
                
                cm_i = 0
                for k in selected:
                    if k != i:
                        cm_i |= (adj[k] | (1 << k))
                        
                # Now cm_i is the conflict mask for mask_without_i.
                # Find best j not in cm_i.
                for j in order_by_weight:
                    if not (cm_i & (1 << j)):
                        # j can be added to mask_without_i
                        # If we just add j, new sum = current_sum - weights[i] + weights[j]
                        # But we might add more than j!
                        # Let's try a mini-greedy from mask_without_i to see the potential.
                        # This is getting expensive.
                        # Let's stick to simple 1-swap first: gain = w[j] - w[i]
                        gain = weights[j] - weights[i]
                        if gain > best_swap_gain + 1e-12:
                            best_swap_gain = gain
                            best_i = i
                            best_j = j
                        break # Only need the best j for this i since weights are sorted desc
            
            if best_i != -1 and best_swap_gain > 1e-12:
                # Perform swap
                current_mask &= ~(1 << best_i)
                current_mask |= (1 << best_j)
                current_sum += best_swap_gain
                # Recalculate conflict_mask
                conflict_mask = 0
                for k in get_selected_indices(current_mask):
                    conflict_mask |= (adj[k] | (1 << k))
                improved = True
                continue

            # Phase 3: Try 2-swap moves (remove i, k; add j, l)
            # This is O(n^4) which is too slow if done naively.
            # Skip for now or do heuristic: 
            # Remove i. Try to add best j. Then remove k. Try to add best l.
            # Or: Remove i, k. Greedily add.
            
            # Let's try a "remove and re-greedy" move for a random selected vertex.
            # Remove i, then greedily add as many as possible from the rest.
            # If sum improves, keep it.
            if selected:
                # Pick a random vertex to remove to escape local optima
                if rng:
                    i_remove = rng.choice(selected)
                else:
                    i_remove = random.choice(selected)
                
                mask_trial = current_mask & ~(1 << i_remove)
                # Greedy add from mask_trial
                trial_mask = mask_trial
                trial_sum = current_sum - weights[i_remove]
                
                # Calculate available
                cm_trial = 0
                for k in get_selected_indices(trial_mask):
                    cm_trial |= (adj[k] | (1 << k))
                
                # Greedy add
                # Use order_by_weight for efficiency
                for i in order_by_weight:
                    if not (cm_trial & (1 << i)):
                        trial_mask |= (1 << i)
                        trial_sum += weights[i]
                        cm_trial |= (adj[i] | (1 << i))

                if trial_sum > current_sum + 1e-12:
                    current_mask = trial_mask
                    current_sum = trial_sum
                    # Recalculate conflict_mask
                    conflict_mask = 0
                    for k in get_selected_indices(current_mask):
                        conflict_mask |= (adj[k] | (1 << k))
                    improved = True
                    continue

        return current_mask, current_sum

    # Run local search from best initial
    best_mask, best_sum = local_search(best_mask, best_sum)
    
    # Try other initial seeds if time permits
    if time.time() < deadline - 10.0:
        rng = random.Random(42)
        # Run a few random restarts
        for _ in range(10):
            m, s = greedy_init(exclude_mask=0, sort_key='random', rng=rng)
            m, s = local_search(m, s, max_iters=5000, rng=rng)
            if s > best_sum:
                best_sum = s
                best_mask = m
            if time.time() > deadline - 5.0:
                break

    # Convert mask back to list of values
    result = [vertices[i] for i in range(n) if (best_mask >> i) & 1]
    
    return result

# EVOLVE-BLOCK-END
