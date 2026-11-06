"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import heapq

    target = Fraction(5, 31)
    
    # Known optimal or near-optimal solution to use as upper bound
    # 1/7 + 1/62 + 1/434 = 5/31
    best_solution = [7, 62, 434]
    best_max = 434

    # Dijkstra's algorithm to find a solution with max denominator < best_max
    # State: (max_denominator_used, current_remainder, frozenset_of_used_denominators)
    start_state = (0, target, frozenset())
    heap = [start_state]
    
    while heap:
        current_max, remainder, used = heapq.heappop(heap)
        
        if remainder == 0:
            return list(used)
            
        # If the current maximum denominator is already >= our best known max,
        # and we haven't found a better solution, we can prune this branch
        # because any further steps will only increase or maintain the max.
        if current_max >= best_max:
            continue

        # The next denominator k must satisfy 1/k <= remainder
        # So k >= 1/remainder
        min_k = (remainder.denominator + remainder.numerator - 1) // remainder.numerator
        
        # We are looking for a solution with max < best_max
        # So k must be < best_max
        max_k = best_max - 1
        
        if min_k > max_k:
            continue
            
        for k in range(min_k, max_k + 1):
            if k in used:
                continue
                
            new_remainder = remainder - Fraction(1, k)
            
            if new_remainder < 0:
                # Should not happen if min_k is correct, but safe to break
                break
                
            new_max = max(current_max, k)
            
            # If new_max is not better than best_max, we can stop iterating k
            # because k is increasing, so new_max will only increase
            if new_max >= best_max:
                break
                
            new_used = used | frozenset([k])
            heapq.heappush(heap, (new_max, new_remainder, new_used))

    # If no better solution found, return the known best
    return best_solution
# EVOLVE-BLOCK-END
