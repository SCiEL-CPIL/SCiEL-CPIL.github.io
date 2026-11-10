import time
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit

    N = 256
    values = list(range(2, N + 1))
    size = len(values)  # 255

    # Precompute pairwise conflict: i and j are incompatible if lcm(values[i], values[j]) <= 256
    conflict_mask = [0] * size
    for i in range(size):
        vi = values[i]
        for j in range(i + 1, size):
            vj = values[j]
            g = gcd(vi, vj)
            if vi * vj // g <= 256:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    # Precompute reciprocal weights
    weights_frac = [Fraction(1, values[i]) for i in range(size)]
    weight_float = [1.0 / values[i] for i in range(size)]

    # Sort by weight descending (smaller numbers have higher weights)
    weight_order = sorted(range(size), key=lambda x: weight_float[x], reverse=True)

    def score_mask(mask):
        total = Fraction(0, 1)
        i = 0
        m = mask
        while m:
            if m & 1:
                total += weights_frac[i]
            i += 1
            m >>= 1
        return total

    def greedy_construct(order, deadline):
        """Greedy independent set construction."""
        S_mask = 0
        for i in order:
            if time.time() - start_time > deadline:
                break
            if not (conflict_mask[i] & S_mask):
                S_mask |= (1 << i)
        return S_mask

    def greedy_color_upper_bound(available_mask, weight_order_local, conflict_local, deadline):
        """
        Compute upper bound using greedy coloring on the subgraph induced by available vertices.
        Returns sum of color class weights as an upper bound on the MWIS weight in that subgraph.
        """
        if time.time() - start_time > deadline:
            return 0.0
        
        # Get list of available vertices in weight order
        avail_list = []
        for idx in weight_order_local:
            if available_mask & (1 << idx):
                avail_list.append(idx)
        
        if not avail_list:
            return 0.0
        
        # Greedy coloring: assign colors to vertices
        # Colors are assigned in order of weight (highest first)
        color_of = {}
        color_weights = []
        
        for v in avail_list:
            # Find the smallest color that v doesn't conflict with
            used_colors = set()
            for u in avail_list:
                if u == v:
                    continue
                if color_of.get(u, -1) >= 0 and (conflict_local[v] & (1 << u)):
                    used_colors.add(color_of[u])
            
            c = 0
            while c in used_colors:
                c += 1
            
            color_of[v] = c
            if c >= len(color_weights):
                color_weights.append(weight_float[v])
            else:
                color_weights[c] += weight_float[v]
        
        # Upper bound: sum of max weight in each color class
        # Actually, for MWIS upper bound, we use: sum over color classes of max weight in class
        # But a simpler valid upper bound is: sum of all weights (loose)
        # Better: use the fact that in each color class, at most one vertex can be in an independent set
        # So upper bound = sum of max weights per color class
        ub = 0.0
        for cw in color_weights:
            ub += cw  # This is actually sum, not max - let me fix
        
        # Correct approach: for each color class, the max weight is an upper bound on contribution
        # But we need to track max per class
        max_per_class = {}
        for v in avail_list:
            c = color_of[v]
            if c not in max_per_class or weight_float[v] > max_per_class[c]:
                max_per_class[c] = weight_float[v]
        
        ub = sum(max_per_class.values())
        return ub

    def branch_and_bound_full(available_mask, current_weight, current_mask, best_weight_ref, deadline):
        """
        Recursive B&B for MWIS.
        available_mask: vertices that can still be added
        current_weight: float weight of current solution
        current_mask: bitmask of current solution
        best_weight_ref: list containing [best_weight, best_mask]
        """
        if time.time() - start_time > deadline:
            return
        
        if available_mask == 0:
            if current_weight > best_weight_ref[0]:
                best_weight_ref[0] = current_weight
                best_weight_ref[1] = current_mask
            return
        
        # Upper bound: current + sum of weights of available vertices (loose but fast)
        # For a tighter bound, we'd use coloring, but that's expensive
        # Let's use: current + sum of weights in available
        # Quick sum:
        avail_sum = 0.0
        m = available_mask
        i = 0
        while m:
            if m & 1:
                avail_sum += weight_float[i]
            i += 1
            m >>= 1
        
        if current_weight + avail_sum <= best_weight_ref[0] + 1e-15:
            return
        
        # Pick the highest-weight available vertex
        # Find it by scanning in weight_order
        v = -1
        for idx in weight_order:
            if available_mask & (1 << idx):
                v = idx
                break
        
        if v == -1:
            return
        
        # Branch 1: include v
        new_available = available_mask & ~(1 << v) & ~conflict_mask[v]
        branch_and_bound_full(new_available, current_weight + weight_float[v], 
                             current_mask | (1 << v), best_weight_ref, deadline)
        
        # Branch 2: exclude v
        new_available = available_mask & ~(1 << v)
        branch_and_bound_full(new_available, current_weight, current_mask, best_weight_ref, deadline)

    # Multi-start approach with different initial solutions
    best_weight_frac = Fraction(0, 1)
    best_mask = 0
    
    # Start 1: Greedy in weight order
    mask1 = greedy_construct(weight_order, deadline)
    w1 = score_mask(mask1)
    if w1 > best_weight_frac:
        best_weight_frac = w1
        best_mask = mask1
    
    # Start 2: Greedy in reverse weight order (to get different structure)
    reverse_order = list(reversed(weight_order))
    mask2 = greedy_construct(reverse_order, deadline)
    w2 = score_mask(mask2)
    if w2 > best_weight_frac:
        best_weight_frac = w2
        best_mask = mask2
    
    # Local search refinement
    def local_search(S_mask, deadline):
        """Improve via additions and swaps."""
        in_set = [False] * size
        for i in range(size):
            if S_mask & (1 << i):
                in_set[i] = True
        
        current_score = score_mask(S_mask)
        improved = True
        max_iterations = 500
        
        while improved and time.time() - start_time < deadline:
            improved = False
            max_iterations -= 1
            if max_iterations <= 0:
                break
            
            # Try adding a node
            for i in weight_order:
                if in_set[i]:
                    continue
                if not (conflict_mask[i] & S_mask):
                    S_mask |= (1 << i)
                    in_set[i] = True
                    current_score += weights_frac[i]
                    improved = True
                    break
            
            if not improved:
                # Try swaps: remove u, add v where v has higher weight
                for u in range(size):
                    if not in_set[u]:
                        continue
                    S_without_u = S_mask & ~(1 << u)
                    for v in weight_order:
                        if in_set[v]:
                            continue
                        if weight_float[v] <= weight_float[u]:
                            break
                        if not (conflict_mask[v] & S_without_u):
                            S_mask = S_without_u | (1 << v)
                            in_set[u] = False
                            in_set[v] = True
                            current_score += weights_frac[v] - weights_frac[u]
                            improved = True
                            break
                    if improved:
                        break
        
        return S_mask, current_score
    
    # Refine each start
    mask1, w1 = local_search(mask1, deadline)
    if w1 > best_weight_frac:
        best_weight_frac = w1
        best_mask = mask1
    
    mask2, w2 = local_search(mask2, deadline)
    if w2 > best_weight_frac:
        best_weight_frac = w2
        best_mask = mask2
    
    # B&B on top K vertices for more precise optimization
    if time.time() - start_time < deadline - 10:
        K = 40
        top_indices = weight_order[:K]
        
        # Build local conflict and weights
        local_conflict = [0] * K
        for i in range(K):
            gi = top_indices[i]
            for j in range(i + 1, K):
                gj = top_indices[j]
                if conflict_mask[gi] & (1 << gj):
                    local_conflict[i] |= (1 << j)
                    local_conflict[j] |= (1 << i)
        
        local_weights = [weight_float[top_indices[i]] for i in range(K)]
        local_weight_order = sorted(range(K), key=lambda x: local_weights[x], reverse=True)
        
        # B&B on local problem
        best_local = [0.0, 0]
        
        def bb_local(available, current_w, current_mask):
            if time.time() - start_time > deadline - 5:
                return
            
            if available == 0:
                if current_w > best_local[0]:
                    best_local[0] = current_w
                    best_local[1] = current_mask
                return
            
            # Upper bound
            avail_sum = 0.0
            m = available
            i = 0
            while m:
                if m & 1:
                    avail_sum += local_weights[i]
                i += 1
                m >>= 1
            
            if current_w + avail_sum <= best_local[0] + 1e-15:
                return
            
            # Pick highest weight available
            v = -1
            for idx in local_weight_order:
                if available & (1 << idx):
                    v = idx
                    break
            
            if v == -1:
                return
            
            # Include v
            new_avail = available & ~(1 << v) & ~local_conflict[v]
            bb_local(new_avail, current_w + local_weights[v], current_mask | (1 << v))
            
            # Exclude v
            new_avail = available & ~(1 << v)
            bb_local(new_avail, current_w, current_mask)
        
        full_avail = (1 << K) - 1
        bb_local(full_avail, 0.0, 0)
        
        # Convert local solution to global mask
        local_best_mask = best_local[1]
        global_local_mask = 0
        for i in range(K):
            if local_best_mask & (1 << i):
                global_local_mask |= (1 << top_indices[i])
        
        # Combine with best global solution: try adding local vertices that don't conflict
        combined_mask = best_mask
        for i in range(K):
            if local_best_mask & (1 << i):
                gi = top_indices[i]
                if not (conflict_mask[gi] & combined_mask):
                    combined_mask |= (1 << gi)
        
        w_combined = score_mask(combined_mask)
        if w_combined > best_weight_frac:
            best_weight_frac = w_combined
            best_mask = combined_mask
    
    # Final local search on the best mask
    best_mask, best_weight_frac = local_search(best_mask, deadline - 2)
    
    # Convert mask to result list
    result = []
    for i in range(size):
        if best_mask & (1 << i):
            result.append(values[i])
    
    # Verify
    from math import lcm as lcm_func
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if lcm_func(result[i], result[j]) <= 256:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to greedy
        mask_fallback = greedy_construct(weight_order, deadline)
        result = []
        for i in range(size):
            if mask_fallback & (1 << i):
                result.append(values[i])
    
    return sorted(result)
