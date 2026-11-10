import time
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit

    N = 256
    values = list(range(2, N + 1))
    n = len(values)

    # 1. Precompute conflict graph
    # Two numbers a, b conflict if lcm(a, b) <= 256.
    # Note: For a, b in [2, 256], lcm(a, b) <= 256 implies that a and b are "close" in the divisibility lattice.
    # Specifically, if a | b, then lcm(a, b) = b <= 256, so they conflict.
    # If gcd(a, b) = 1, lcm(a, b) = a*b. If a*b <= 256, they conflict.
    
    # We want to select a subset S such that for all distinct a, b in S, lcm(a, b) > 256.
    # This is equivalent to finding an Independent Set in the conflict graph.
    # We want to maximize sum(1/a).

    # Build adjacency lists/masks
    # Using Python integers as bitsets is efficient for n=255.
    # index i corresponds to value values[i] = i + 2
    # Let's map value v to index v-2.
    
    adj = [0] * n
    
    for i in range(n):
        vi = values[i]
        for j in range(i + 1, n):
            vj = values[j]
            # Check conflict
            # Optimization: If vi > vj, swap logic, but here vi < vj.
            # lcm(vi, vj) = vi * vj // gcd(vi, vj)
            # We can check if vi * vj <= 256 * gcd(vi, vj)
            # Or simply compute lcm.
            # Given n=255, O(n^2) is ~65k ops, very fast.
            if lcm(vi, vj) <= N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # 2. Sort vertices by weight (1/value) descending
    # Higher weight = smaller value.
    # We want to include small values if possible.
    # Sort indices by value ascending (which means weight descending).
    order = list(range(n))
    order.sort(key=lambda i: values[i])
    
    # Reorder weights and adjacency to match this order
    # new_index k corresponds to original index order[k]
    weights = [1.0 / values[order[k]] for k in range(n)]
    
    # New adjacency masks in the sorted order
    # Let new_adj[k] be the bitmask of neighbors of vertex k in the sorted order.
    new_adj = [0] * n
    for k in range(n):
        orig_k = order[k]
        mask_orig = adj[orig_k]
        mask_new = 0
        for l in range(n):
            if mask_orig & (1 << order[l]):
                mask_new |= (1 << l)
        new_adj[k] = mask_new

    # 3. Branch and Bound
    # State: 
    #   k: next vertex to consider (in sorted order)
    #   avail_mask: bitmask of vertices that are compatible with the current selection
    #                and have index >= k (we only consider vertices >= k to avoid duplicates)
    #   current_sum: sum of weights of selected vertices
    #   current_set: list of selected indices (in sorted order)
    
    best_score = 0.0
    best_set = []
    
    # Greedy initialization for a good lower bound
    # Pick vertices in order if compatible
    greedy_mask = 0
    greedy_sum = 0.0
    greedy_set = []
    for k in range(n):
        if not (new_adj[k] & greedy_mask):
            greedy_set.append(k)
            greedy_mask |= (1 << k)
            greedy_sum += weights[k]
            
    best_score = greedy_sum
    best_set = list(greedy_set)
    
    # Precompute suffix sums of weights for a simple upper bound
    # suffix_sum[k] = sum(weights[k:])
    suffix_sum = [0.0] * (n + 1)
    for k in range(n - 1, -1, -1):
        suffix_sum[k] = suffix_sum[k + 1] + weights[k]
        
    # Better Upper Bound:
    # At node (k, avail_mask), the maximum additional weight we can get is bounded by
    # the sum of weights of vertices in avail_mask with index >= k.
    # We can compute this on the fly or precompute a data structure.
    # Given n=255, iterating through bits of avail_mask & ((1<<n)-1) >> k is fast enough?
    # Let's use a helper to compute the sum of weights in a mask quickly.
    # Since weights are fixed, we can just sum them.
    
    def sum_weights_in_mask(mask, start_idx):
        # Sum weights[i] for i in mask where i >= start_idx
        s = 0.0
        # Iterate through set bits
        m = mask
        while m:
            # Get lowest set bit
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if idx >= start_idx:
                s += weights[idx]
            m ^= lsb
        return s

    # To speed up B&B, we can use a stack.
    # State: (k, avail_mask, current_sum, current_set_tuple)
    # We push (k+1, ...) for exclude and include branches.
    
    # Optimization: 
    # If we include vertex k, we remove k and its neighbors from avail_mask.
    # If we exclude vertex k, we just move to k+1.
    
    # We use a stack for iterative DFS to manage recursion depth and time checking.
    
    # Initial state:
    # k=0
    # avail_mask: all vertices 0..n-1 are available initially? 
    # Yes, but we must ensure we only pick vertices >= k.
    # Actually, standard B&B on ordered vertices:
    # avail_mask contains vertices that are compatible with *selected* vertices.
    # We iterate k from 0 to n-1.
    # At step k, we decide if we include k.
    # If we include k, k must be in avail_mask.
    # The new avail_mask for subsequent steps (k+1...) is (avail_mask & ~new_adj[k] & ~(1<<k)) & (mask of vertices > k).
    # Wait, the "mask of vertices > k" is handled by the fact that we only consider indices >= k+1 in the next step.
    # However, avail_mask might contain bits for indices < k+1. We should mask them out to keep avail_mask clean?
    # Or just ignore them in the next step?
    # It's safer to keep avail_mask restricted to indices >= current_k.
    
    # Let's define the state clearly:
    # k: current index to consider.
    # avail_mask: bitmask of vertices that are compatible with all previously selected vertices.
    #             It may contain bits for indices < k, but we will only look at bits >= k.
    #             Actually, it's better to maintain avail_mask such that bits < k are 0.
    
    # Start:
    # k=0, avail_mask = (1 << n) - 1
    
    stack = [(0, (1 << n) - 1, 0.0, ())]
    
    while stack:
        if time.time() > deadline:
            break
            
        k, avail_mask, cur_sum, cur_set = stack.pop()
        
        # Prune: Upper bound
        # Max possible sum = cur_sum + sum of weights of vertices in avail_mask with index >= k
        # We can compute this sum.
        # Optimization: Precompute prefix sums? No, avail_mask changes.
        # Let's compute the bound.
        # To make it faster, we can use a simple bound: cur_sum + suffix_sum[k]
        # This is loose but fast.
        # A tighter bound: sum of weights in (avail_mask >> k)
        # Let's use the tighter bound but optimize the calculation.
        
        # Calculate tight bound
        # Mask out bits < k
        relevant_mask = avail_mask & ((1 << n) - 1) & ~((1 << k) - 1)
        
        # Sum weights in relevant_mask
        # We can do this quickly if we precompute weight sums for chunks?
        # Or just iterate. n=255, so max 255 iterations per node.
        # Number of nodes might be large.
        
        # Let's try a hybrid bound:
        # If cur_sum + suffix_sum[k] <= best_score: continue
        if cur_sum + suffix_sum[k] <= best_score:
            continue
            
        # If k == n, we have a complete set
        if k == n:
            if cur_sum > best_score:
                best_score = cur_sum
                best_set = list(cur_set)
            continue
        
        # Option 1: Exclude k
        # Next state: k+1, avail_mask (unchanged, but bits < k+1 don't matter), cur_sum, cur_set
        # We push this first so it's popped last (LIFO), exploring Include first?
        # Or push Include first so Exclude is popped first?
        # Usually Include (adding weight) is more promising.
        # Let's push Exclude first, then Include, so Include is popped first.
        
        # Exclude branch
        # avail_mask for next step: we can clear bit k to keep it clean?
        # avail_mask_exclude = avail_mask & ~(1 << k)
        stack.append((k + 1, avail_mask & ~(1 << k), cur_sum, cur_set))
        
        # Option 2: Include k
        if avail_mask & (1 << k):
            # Check if k is compatible (it is, by definition of avail_mask)
            # New avail_mask: remove k and its neighbors
            # neighbors of k in the sorted graph: new_adj[k]
            # We must also remove k itself.
            # new_avail = avail_mask & ~(new_adj[k] | (1 << k))
            
            new_avail = avail_mask & ~(new_adj[k] | (1 << k))
            new_sum = cur_sum + weights[k]
            new_set = cur_set + (k,)
            
            # Prune check for this branch
            # Tight bound:
            # relevant_mask_new = new_avail & ~((1 << (k+1)) - 1)
            # bound = new_sum + sum_weights(relevant_mask_new)
            
            # Quick check with suffix sum
            if new_sum + suffix_sum[k + 1] > best_score:
                # Push Include branch
                stack.append((k + 1, new_avail, new_sum, new_set))
                
    # Convert best_set (indices in sorted order) back to values
    result = [values[order[v]] for v in best_set]
    
    return result

# EVOLVE-BLOCK-END
