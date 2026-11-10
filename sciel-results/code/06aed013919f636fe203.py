def construct():
    from fractions import Fraction
    import bisect
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Split into two halves
    mid = n // 2
    left_denoms = denominators[:mid]
    right_denoms = denominators[mid:]
    
    # Generate all subset sums < 1 for a list of denominators
    # Returns list of (Fraction_sum, list_of_local_indices)
    def gen_subsets(denoms):
        # Start with empty subset
        sums = [(Fraction(0), [])]
        for i, d in enumerate(denoms):
            frac = Fraction(1, d)
            new_entries = []
            for s, idxs in sums:
                ns = s + frac
                if ns < 1:
                    new_entries.append((ns, idxs + [i]))
            sums.extend(new_entries)
        return sums
    
    left_sums = gen_subsets(left_denoms)
    right_sums = gen_subsets(right_denoms)
    
    # Sort right_sums by value for binary search
    right_sums.sort(key=lambda x: x[0])
    right_values = [x[0] for x in right_sums]
    
    best_deficit = Fraction(1, 1)  # Start with deficit 1
    best_indices = None
    
    for s_left, idxs_left in left_sums:
        remaining = Fraction(1, 1) - s_left
        if remaining <= 0:
            continue
        # Find largest right sum strictly less than remaining
        pos = bisect.bisect_left(right_values, remaining) - 1
        if pos >= 0:
            s_right = right_values[pos]
            idxs_right = right_sums[pos][1]
            total = s_left + s_right
            if total >= 1:
                continue
            deficit = Fraction(1, 1) - total
            if deficit < best_deficit:
                best_deficit = deficit
                # Map to global indices
                global_idxs = list(idxs_left) + [i + mid for i in idxs_right]
                if len(global_idxs) > 0:
                    best_indices = global_idxs
    
    if best_indices is None or len(best_indices) == 0:
        # Fallback: return index 0 (denominator 2)
        return [0]
    
    return best_indices
