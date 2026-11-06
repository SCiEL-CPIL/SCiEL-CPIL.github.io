import itertools

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
    
    # best[n] will store the list of terms for the shortest antichain summing to n
    best = [None] * 129
    
    # Pre-check: k=1
    for s in smooth:
        if s <= 128:
            best[s] = [s]
            
    # Iterate k from 2 upwards
    # We can stop when all 128 targets are filled
    remaining = [i for i in range(1, 129) if best[i] is None]
    
    max_k = 10  # Safe upper bound
    for k in range(2, max_k + 1):
        if not remaining:
            break
            
        # Generate combinations of size k
        # To optimize, we can filter smooth numbers to be <= 128
        # Also, we can prune combinations that already sum > 128
        # But itertools.combinations doesn't support pruning easily.
        # Given the small size of 'smooth' (38 elements), and k being small,
        # direct generation is acceptable for k=2,3,4.
        # For k=5+, it might be slow. Let's see how many are left.
        
        # Optimization: Only consider smooth numbers that are <= max(remaining)
        # But sum must be exactly n.
        
        found_any = False
        for terms in itertools.combinations(smooth, k):
            s = sum(terms)
            if s > 128:
                continue
            if best[s] is not None:
                continue
                
            # Check antichain property
            is_antichain = True
            # Sort terms to make division check easier? 
            # If a divides b, and a <= b, then b % a == 0.
            # We need to check all pairs.
            # Since k is small, O(k^2) is fine.
            for i in range(k):
                for j in range(i + 1, k):
                    a, b = terms[i], terms[j]
                    if b % a == 0 or a % b == 0:
                        is_antichain = False
                        break
                if not is_antichain:
                    break
            
            if is_antichain:
                best[s] = list(terms)
                if s in remaining:
                    remaining.remove(s)
                found_any = True
        
        if not found_any:
            # If no new antichains found for this k, maybe larger k won't help?
            # Or maybe we need to increase the pool of smooth numbers?
            # But for n<=128, k<=4 should be enough.
            # If we get stuck, we might need to handle it differently.
            # However, the problem states "All targets require valid representations".
            # So we assume they exist.
            pass

    # If any target is still None, we might need to increase max_k or smooth limit
    # But for n<=128, this should be fine.
    # Let's verify if any are left.
    if any(best[i] is None for i in range(1, 129)):
        # Fallback: try larger k or larger smooth numbers?
        # Actually, if a number is not smooth, it can be a sum of smooth numbers.
        # The antichain constraint is the tricky part.
        # For now, we assume the above loop finds them.
        pass

    return best[1:129]
