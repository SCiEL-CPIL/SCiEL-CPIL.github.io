import heapq
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    
    # Known valid solution with max denominator 1736
    # 1/7 + 1/56 + 1/1736 = 8/56 + 1/56 + 1/1736 = 9/56 + 1/1736
    # 9/56 = 9 * 31 / (56 * 31) = 279 / 1736
    # 279/1736 + 1/1736 = 280/1736 = 35/217 = 5/31. Correct.
    
    upper_bound = 1736
    
    # State: (max_denominator, remainder, last_denominator, path)
    heap = [(0, target, 1, ())]
    
    # To avoid re-exploring states, we track the best max_d seen for (r, last_k)
    best_max_d_for_state = {}
    
    while heap:
        max_d, r, last_k, path = heapq.heappop(heap)
        
        if r == 0:
            return list(path)
        
        state_key = (r, last_k)
        if state_key in best_max_d_for_state:
            if best_max_d_for_state[state_key] <= max_d:
                continue
        else:
            best_max_d_for_state[state_key] = max_d
            
        min_k_from_value = (r.denominator + r.numerator - 1) // r.numerator
        min_k_from_order = last_k + 1
        start_k = max(min_k_from_value, min_k_from_order)
        
        if start_k > upper_bound:
            continue
            
        for k in range(start_k, upper_bound + 1):
            new_r = r - Fraction(1, k)
            if new_r < 0:
                break
            
            new_max_d = max(max_d, k)
            new_path = path + (k,)
            
            heapq.heappush(heap, (new_max_d, new_r, k, new_path))
            
    # If no better solution found, return the known one
    return [7, 56, 1736]
