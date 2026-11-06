# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import heapq

    target = Fraction(5, 31)
    
    # Greedy for upper bound
    r = target
    greedy_vals = []
    while r > 0:
        d_g = int(1 / r) + 1
        greedy_vals.append(d_g)
        r -= Fraction(1, d_g)
    upper_bound = max(greedy_vals)
    
    # State: (max_denom, remainder, tuple_used, depth)
    start_state = (0, target, (), 0)
    pq = [start_state]
    
    while pq:
        cost, rem, used, depth = heapq.heappop(pq)
        
        if depth > 4:
            continue
            
        if rem == 0:
            return list(used)
            
        d_min = int(1 / rem) + 1
        
        for d in range(d_min, upper_bound + 1):
            if d in used:
                continue
            
            next_rem = rem - Fraction(1, d)
            if next_rem < 0:
                continue
                
            new_cost = max(cost, d)
            
            if new_cost >= upper_bound:
                continue
                
            if next_rem == 0:
                return list(used) + [d]
                
            heapq.heappush(pq, (new_cost, next_rem, used + (d,), depth + 1))
            
    return greedy_vals
# EVOLVE-BLOCK-END
