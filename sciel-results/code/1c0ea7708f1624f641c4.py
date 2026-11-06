def construct():
    # Generate 5-smooth numbers up to 128
    smooth = []
    for a in range(8):  # 2^7 = 128
        for b in range(6):  # 3^5 = 243 > 128
            for c in range(4):  # 5^3 = 125
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 128:
                    smooth.append(val)
    smooth = sorted(set(smooth))
    
    best = [None] * 129
    
    # Helper to check if val can be added to current antichain
    def is_compatible(val, current_terms):
        for t in current_terms:
            if t % val == 0 or val % t == 0:
                return False
        return True

    # Recursive generator for antichains of size k
    def find_antichains(k, start_idx, current_terms, current_sum):
        if len(current_terms) == k:
            if current_sum <= 128:
                if best[current_sum] is None:
                    best[current_sum] = list(current_terms)
            return
            
        slots_needed = k - len(current_terms)
        if slots_needed > 0:
             # Check if we have enough elements left
             if start_idx + slots_needed > len(smooth):
                 return
             
             # Loose sum bound: minimum possible sum for remaining slots
             if current_sum + slots_needed * smooth[start_idx] > 128:
                 return

        for i in range(start_idx, len(smooth)):
            val = smooth[i]
            if current_sum + val > 128:
                break
            
            if is_compatible(val, current_terms):
                current_terms.append(val)
                find_antichains(k, i + 1, current_terms, current_sum + val)
                current_terms.pop()

    max_k = 10
    for k in range(1, max_k + 1):
        # Check if all are found
        if all(best[i] is not None for i in range(1, 129)):
            break
        find_antichains(k, 0, [], 0)
        
    return best[1:129]
