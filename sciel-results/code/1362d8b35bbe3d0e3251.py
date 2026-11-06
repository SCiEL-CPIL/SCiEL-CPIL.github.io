import time
import random
from fractions import Fraction
import sys

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin for 90s limit

    # 1. Build Conflict Graph
    # vertices are 0..N-1 representing numbers 1..N
    # conflict[u][v] is True if u, v, and some w form a 3-AP
    # Actually, for MWIS, we need edges between vertices that CANNOT coexist.
    # Two vertices u, v conflict if there exists w such that {u,v,w} is a 3-AP.
    # This happens if:
    # 1. u, v are ends: w = (u+v)/2 is integer and in [1,N]
    # 2. u is middle, v is end: w = 2u - v is in [1,N] and w != u, v
    # 3. v is middle, u is end: w = 2v - u is in [1,N] and w != u, v
    
    # We can precompute adjacency lists for efficiency in B&B.
    # adj[i] is a bitmask or set of j such that i and j conflict.
    # Using bitmasks for N=256 is feasible (4 words of 64 bits, or just Python int which handles big ints).
    
    adj = [0] * N  # adj[i] is a bitmask of conflicts for i
    
    for u in range(N):
        val_u = u + 1
        for v in range(u + 1, N):
            val_v = v + 1
            # Check if u, v conflict
            # Case 1: u, v are ends. Middle m = (val_u + val_v) / 2
            if (val_u + val_v) % 2 == 0:
                m_val = (val_u + val_v) // 2
                if 1 <= m_val <= N:
                    # Conflict
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Case 2: u is middle, v is end. Third w = 2*val_u - val_v
            w_val = 2 * val_u - val_v
            if 1 <= w_val <= N and w_val != val_u and w_val != val_v:
                # Conflict
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            # Case 3: v is middle, u is end. Third w = 2*val_v - val_u
            w_val = 2 * val_v - val_u
            if 1 <= w_val <= N and w_val != val_u and w_val != val_v:
                # Conflict
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    # Weights: w[i] = 1 / (i+1)
    weights = [1.0 / (i + 1) for i in range(N)]
    
    # Sort vertices by weight descending to improve B&B pruning?
    # Standard B&B often benefits from ordering. However, the conflict structure is tied to indices.
    # We can keep original indices for adjacency, but process nodes in order of weight.
    # For simplicity and speed in bit operations, let's stick to index order 0..N-1 
    # but maybe we can reorder? Reordering requires remapping adj.
    # Let's try reordering by weight descending.
    
    order = sorted(range(N), key=lambda i: weights[i], reverse=True)
    # Map old index to new position
    pos = [0] * N
    for new_idx, old_idx in enumerate(order):
        pos[old_idx] = new_idx
        
    new_adj = [0] * N
    new_weights = [0.0] * N
    
    for new_idx, old_idx in enumerate(order):
        new_weights[new_idx] = weights[old_idx]
        mask = adj[old_idx]
        new_mask = 0
        # Remap bits
        # This is slow if done bit by bit. 
        # Alternative: Don't reorder. Use standard B&B on 0..N-1.
        # Since weights are 1/x, small x (high weight) are at low indices.
        # B&B usually branches on the first available node. 
        # If we branch on node 0 (weight 1.0), then node 1 (0.5), etc., we explore high weights first.
        # This is naturally good for upper bound pruning if we take a greedy solution first.
        
        # Let's stick to original order 0..N-1 to avoid remapping overhead and potential bugs.
        # Node 0 is 1 (weight 1.0), Node 1 is 2 (weight 0.5), etc.
        
    # So we use adj and weights as is.
    
    # Precompute "available" masks?
    # In B&B, we maintain a mask of available vertices.
    
    best_score = 0.0
    best_set = set()
    
    # 1. Get a good initial lower bound using Greedy
    # Greedy: pick highest weight available, remove conflicts, repeat.
    def greedy_initial():
        available = (1 << N) - 1
        current_set = []
        current_score = 0.0
        while available:
            # Find highest weight vertex in available
            # Since weights are decreasing with index, the lowest index in available is best.
            # But wait, weights are 1/(i+1). So index 0 has weight 1, index 1 has 0.5.
            # So yes, lowest index = highest weight.
            
            # Find lowest set bit
            # (available & -available) gives the lowest set bit mask
            lsb_mask = available & -available
            # Index of lsb
            # int.bit_length() - 1
            idx = lsb_mask.bit_length() - 1
            
            current_set.append(idx)
            current_score += weights[idx]
            
            # Remove idx and its neighbors from available
            # available &= ~( (1 << idx) | adj[idx] )
            available &= ~( (1 << idx) | adj[idx] )
            
        return current_set, current_score

    init_set, init_score = greedy_initial()
    best_score = init_score
    best_set = set(init_set)
    
    # 2. Branch and Bound
    # State: (available_mask, current_score, current_set_mask)
    # We want to maximize current_score + sum(weights of selected from available)
    # Upper bound: current_score + sum(weights[i] for i in available)
    # To speed up upper bound calculation, we can precompute suffix sums of weights?
    # No, available is arbitrary subset.
    # However, we can use a greedy upper bound: 
    #   UB = current_score + greedy_bound(available)
    # where greedy_bound picks highest weight, removes neighbors, etc.
    # This is tighter than simple sum but expensive.
    
    # Alternative UB: 
    #   Sort available by weight desc.
    #   Pick first, remove neighbors.
    #   Pick next available, remove neighbors.
    #   ...
    #   Sum weights.
    
    # Let's implement a recursive B&B with time checks.
    
    sys.setrecursionlimit(5000)
    
    def greedy_bound(available):
        """Calculate an upper bound by greedily selecting non-conflicting vertices from available."""
        if available == 0:
            return 0.0
        score = 0.0
        avail = available
        # We can optimize this by just summing weights of a maximal independent set found greedily.
        # Since we just need an upper bound, any independent set sum is a lower bound on the optimal, 
        # but we need an UPPER bound on the optimal to prune.
        # Wait. 
        # If we find a valid set S in available, sum(S) is a LOWER bound on the optimal value of available.
        # We need an UPPER bound to prune.
        # Simple Upper Bound: Sum of all weights in available. (Too loose)
        # Better Upper Bound: 
        #   The optimal set is an independent set.
        #   We can compute a fractional packing or use the greedy independent set sum? 
        #   No, greedy independent set sum is <= Optimal. So it's a lower bound.
        #   We need something >= Optimal.
        #   
        #   Common UB for MWIS: 
        #   1. Sum of weights of all vertices in available.
        #   2. LP relaxation.
        #   3. For specific structures, maybe we can use the fact that if we pick v, we lose neighbors.
        #   
        #   Let's use a simple but effective heuristic UB:
        #   Process vertices in order of decreasing weight.
        #   If we pick v, we gain w(v) but lose w(neighbors).
        #   This is hard to sum up simply.
        #   
        #   Let's stick to: UB = current_score + sum(weights[i] for i in available).
        #   This is very loose. 
        #   
        #   Improvement: 
        #   We can use the result of the greedy algorithm on `available` as a LOWER bound to update best_score.
        #   For UPPER bound, we can use the sum of weights, but prune if current_score + sum_weights <= best_score.
        #   Given the weights drop off as 1/x, the sum of remaining weights might be small enough to prune effectively if best_score is high.
        
        # Let's try to compute a tighter UB using a simple heuristic:
        #   Assume we can take all vertices in available, but we must resolve conflicts.
        #   A common bound for independent set is the "degree-based" bound or similar.
        #   For now, let's use the sum of weights. It's O(N) to compute if we maintain a running sum?
        #   Or we can precompute prefix sums? No, available is a bitmask.
        #   
        #   Let's just compute sum(weights[i] for i in range(N) if (available >> i) & 1).
        #   This is O(N) per node. N=256. B&B nodes might be many.
        #   
        #   Optimization: Pass the sum of weights in available as a parameter?
        #   When we branch on v:
        #   Branch 1 (Include v): new_available = available & ~(1<<v) & ~adj[v]
        #   Branch 2 (Exclude v): new_available = available & ~(1<<v)
        #   
        #   We can maintain `sum_avail` as a parameter.
        #   sum_new_avail_1 = sum_avail - weights[v] - sum(weights[i] for i in neighbors(v) if i in available)
        #   sum_new_avail_2 = sum_avail - weights[v]
        
        return 0.0 # Placeholder, we will handle UB in the main loop logic

    # Let's implement B&B iteratively or recursively with explicit stack to avoid recursion depth issues and allow time checks.
    # Stack items: (available_mask, current_score, sum_avail_weights)
    
    # Initial sum
    total_weight = sum(weights)
    stack = [( (1 << N) - 1, 0.0, total_weight )]
    
    nodes_explored = 0
    
    while stack:
        if time.time() > deadline - 2.0:
            break
            
        available, cur_score, sum_avail = stack.pop()
        
        if available == 0:
            if cur_score > best_score:
                best_score = cur_score
                # Reconstruct set? We didn't track it. 
                # For the final answer, we need the set.
                # We can track the set in the stack items, but that's memory heavy.
                # Alternatively, we can just return the greedy set if B&B doesn't find better?
                # Or we can track the path.
                # Let's track the set as a bitmask in the stack.
                pass
            continue
            
        # Upper Bound Check
        # If cur_score + sum_avail <= best_score, prune
        if cur_score + sum_avail <= best_score:
            continue
            
        # Find a vertex to branch on.
        # Heuristic: Pick the vertex in `available` with highest weight (lowest index).
        lsb_mask = available & -available
        v = lsb_mask.bit_length() - 1
        
        # Update best_score with a greedy solution from this state?
        # This can improve the lower bound quickly.
        # Greedy on `available`:
        avail_temp = available
        greedy_score = 0.0
        greedy_mask = 0
        while avail_temp:
            lsb = avail_temp & -avail_temp
            idx = lsb.bit_length() - 1
            greedy_score += weights[idx]
            greedy_mask |= (1 << idx)
            avail_temp &= ~( (1 << idx) | adj[idx] )
        
        if cur_score + greedy_score > best_score:
            best_score = cur_score + greedy_score
            # We need to store the set. 
            # This is a problem. We are not tracking the set in the B&B state.
            # Let's add set_mask to the stack.
            
        # Branch 1: Include v
        # new_available = available & ~(1<<v) & ~adj[v]
        # new_sum_avail = sum_avail - weights[v] - sum(weights[i] for i in neighbors(v) in available)
        
        # To calculate new_sum_avail efficiently, we need to know which neighbors are in available.
        # neigh_in_avail = adj[v] & available
        # sum_neigh = sum(weights[i] for i in range(N) if (neigh_in_avail >> i) & 1)
        
        # This sum is O(N). 
        # Let's precompute a helper or just do it.
        
        neigh_in_avail = adj[v] & available
        
        # Calculate sum of weights in neigh_in_avail
        s_neigh = 0.0
        tmp = neigh_in_avail
        while tmp:
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            s_neigh += weights[idx]
            tmp ^= lsb
            
        new_avail_1 = available & ~( (1 << v) | adj[v] )
        new_sum_1 = sum_avail - weights[v] - s_neigh
        
        # Branch 2: Exclude v
        new_avail_2 = available & ~(1 << v)
        new_sum_2 = sum_avail - weights[v]
        
        # Push to stack. 
        # We need to track the set to return it.
        # Let's modify stack to hold (available, cur_score, sum_avail, set_mask)
        
        # This requires restarting the stack logic with set_mask.
        # Let's restart the B&B loop with set_mask included.
        
        pass
        
    # The above loop was a draft. Let's rewrite properly with set_mask.
    
    stack = [ ( (1 << N) - 1, 0.0, total_weight, 0 ) ]
    best_set_mask = 0
    
    # Re-initialize best from greedy
    init_set, init_score = greedy_initial()
    best_score = init_score
    init_mask = 0
    for idx in init_set:
        init_mask |= (1 << idx)
    best_set_mask = init_mask
    
    while stack:
        if time.time() > deadline - 2.0:
            break
            
        available, cur_score, sum_avail, set_mask = stack.pop()
        
        if available == 0:
            if cur_score > best_score:
                best_score = cur_score
                best_set_mask = set_mask
            continue
            
        # Prune
        if cur_score + sum_avail <= best_score:
            continue
            
        # Pick vertex v (lowest index in available)
        lsb_mask = available & -available
        v = lsb_mask.bit_length() - 1
        
        # Optional: Update best with greedy from this state
        # This is expensive but might tighten bound.
        # Given N=256, let's skip per-node greedy update to save time, 
        # relying on the initial greedy and the structure.
        # Actually, per-node greedy is O(N^2) worst case per node. Too slow.
        # Let's just rely on the sum_avail bound.
        
        # Branch 1: Include v
        neigh_in_avail = adj[v] & available
        s_neigh = 0.0
        tmp = neigh_in_avail
        while tmp:
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            s_neigh += weights[idx]
            tmp ^= lsb
            
        new_avail_1 = available & ~( (1 << v) | adj[v] )
        new_sum_1 = sum_avail - weights[v] - s_neigh
        new_score_1 = cur_score + weights[v]
        new_set_mask_1 = set_mask | (1 << v)
        
        # Branch 2: Exclude v
        new_avail_2 = available & ~(1 << v)
        new_sum_2 = sum_avail - weights[v]
        new_score_2 = cur_score
        new_set_mask_2 = set_mask
        
        # Push. Order: Push Exclude first so Include is processed first (LIFO)?
        # Or push Include first? 
        # Usually we want to explore promising branches first.
        # Include v has higher immediate score. 
        # If we push Exclude, then Include, Include is popped first.
        
        stack.append( (new_avail_2, new_score_2, new_sum_2, new_set_mask_2) )
        stack.append( (new_avail_1, new_score_1, new_sum_1, new_set_mask_1) )
        
        # Time check inside loop is already done at top.
        
    # Convert best_set_mask to list of integers
    result_set = []
    for i in range(N):
        if (best_set_mask >> i) & 1:
            result_set.append(i + 1)
            
    # Validate
    def _is_3ap_free(s):
        if not s:
            return True
        s_set = set(s)
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                c = 2 * b - a
                if c > b and c in s_set:
                    return False
        return True
        
    if not _is_3ap_free(result_set):
        # Fallback to greedy
        result_set = init_set
        
    if not result_set:
        return [1]
        
    return result_set

if __name__ == "__main__":
    res = construct()
    print(res)
    score = sum(1.0/x for x in res)
    print(f"Score: {score}")
    print(f"Size: {len(res)}")
