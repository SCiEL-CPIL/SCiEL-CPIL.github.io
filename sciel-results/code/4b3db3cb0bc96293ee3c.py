import time
import random
import math

def construct():
    n = 256
    start_time = time.time()
    time_limit = 55.0  # Safety margin from 60s

    # 1. Precompute APs
    # AP is defined by (start, step). 
    # Indices: start, start+step, ... < n
    aps = []
    index_to_aps = [[] for _ in range(n)]
    
    for d in range(1, n + 1):
        for start in range(d):
            idxs = list(range(start, n, d))
            if idxs:
                ap_id = len(aps)
                aps.append(idxs)
                for idx in idxs:
                    index_to_aps[idx].append(ap_id)
    
    num_aps = len(aps)
    
    # 2. Initialize signs and partial sums
    rng = random.Random(42)
    signs = [rng.choice([-1, 1]) for _ in range(n)]
    
    # partial_sums[ap_id] is a list of partial sums for that AP
    partial_sums = []
    max_sums = []
    
    for idxs in aps:
        ps = [0] * len(idxs)
        curr = 0
        for k, idx in enumerate(idxs):
            curr += signs[idx]
            ps[k] = curr
        partial_sums.append(ps)
        max_sums.append(max(abs(x) for x in ps))
        
    current_cost = max(max_sums)
    best_signs = signs.copy()
    best_cost = current_cost
    
    # 3. Simulated Annealing
    T = 10.0
    cooling_rate = 0.999
    min_T = 0.01
    
    while T > min_T and time.time() - start_time < time_limit:
        # Pick random index
        i = rng.randrange(n)
        
        # Calculate delta
        old_sign = signs[i]
        new_sign = -old_sign
        delta = 2 * new_sign # Change in sign value is 2*new_sign? 
        # Wait. Sum changes by (new_sign - old_sign) = 2*new_sign.
        
        # Update affected APs
        affected_ap_ids = index_to_aps[i]
        
        old_max_costs = []
        new_max_costs = []
        
        for ap_id in affected_ap_ids:
            idxs = aps[ap_id]
            # Find position of i in idxs
            # Since idxs is sorted, we can find it. 
            # Actually, we can precompute the local index for each (i, ap_id) pair? 
            # Or just use index(). n is small.
            pos = idxs.index(i)
            
            ps = partial_sums[ap_id]
            old_max = max_sums[ap_id]
            old_max_costs.append(old_max)
            
            # Update partial sums from pos onwards
            # ps[pos:] += delta
            # This is O(L-pos)
            for k in range(pos, len(ps)):
                ps[k] += delta
            
            new_max = max(abs(x) for x in ps)
            new_max_costs.append(new_max)
            max_sums[ap_id] = new_max
            
        # Calculate new global cost
        # We only need to check if the new max of affected APs exceeds the previous global max
        # Or if the previous global max AP was affected and its max decreased.
        # Simplest: recompute global max? No, that's O(num_aps).
        # Better: keep a frequency array of max_sums? 
        # Or just take max(max_sums). num_aps is ~1400. This is fast.
        new_global_cost = max(max_sums)
        
        # Acceptance criterion
        if new_global_cost < current_cost:
            # Accept
            current_cost = new_global_cost
            signs[i] = new_sign
            if current_cost < best_cost:
                best_cost = current_cost
                best_signs = signs.copy()
        else:
            # Probabilistic accept
            delta_cost = new_global_cost - current_cost
            if delta_cost < 0:
                accept = True
            else:
                # Boltzmann factor
                # Avoid overflow/underflow
                if T < 1e-9:
                    accept = False
                else:
                    prob = math.exp(-delta_cost / T)
                    accept = rng.random() < prob
            
            if accept:
                current_cost = new_global_cost
                signs[i] = new_sign
            else:
                # Revert
                for ap_id, old_max in zip(affected_ap_ids, old_max_costs):
                    idxs = aps[ap_id]
                    pos = idxs.index(i)
                    ps = partial_sums[ap_id]
                    # Revert update: ps[pos:] -= delta
                    for k in range(pos, len(ps)):
                        ps[k] -= delta
                    max_sums[ap_id] = old_max
                # current_cost remains unchanged
        
        T *= cooling_rate
        
    return best_signs
