import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Greedy Egyptian fraction algorithm
    # Start with remaining sum = 1, find largest unit fraction <= remaining
    # with denominator >= 5 and not already used
    
    def greedy_egyptian(target_sum, min_denom, max_terms=256):
        """
        Greedy algorithm: at each step, pick the largest unit fraction 1/d
        such that d >= min_denom, d not in used, and 1/d <= remaining.
        """
        remaining = target_sum
        used = set()
        result = []
        
        for _ in range(max_terms):
            if remaining == 0:
                break
            if remaining < 0:
                return None
            
            # Find smallest d >= min_denom such that 1/d <= remaining
            # i.e., d >= 1/remaining
            # We want the LARGEST unit fraction, so smallest valid d
            # 1/d <= remaining means d >= 1/remaining
            # d must be integer >= min_denom and >= ceil(1/remaining)
            
            num, den = remaining.numerator, remaining.denominator
            # 1/d <= num/den => den <= num*d => d >= den/num
            # d >= ceil(den/num)
            min_d_needed = (den + num - 1) // num  # ceil(den/num)
            
            # Also need d >= min_denom
            d = max(min_denom, min_d_needed)
            
            # Check that 1/d <= remaining (should be guaranteed by construction)
            # But also need d not in used
            while d in used:
                d += 1
            
            # Verify 1/d <= remaining
            if Fraction(1, d) > remaining:
                # This shouldn't happen with our construction, but just in case
                return None
            
            result.append(d)
            used.add(d)
            remaining -= Fraction(1, d)
            min_denom = d + 1  # Next denominator must be larger (distinct)
            
            # Time check
            if time.time() > deadline:
                break
        
        if remaining == 0:
            return result
        return None
    
    # Try greedy approach
    result = greedy_egyptian(Fraction(1, 1), min_denom=5)
    
    if result is not None and len(result) <= 256:
        # Verify the solution
        total = sum(Fraction(1, d) for d in result)
        if total == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
            return result
    
    # Fallback: try with different starting strategies
    # Try starting with slightly larger minimum denominators to see if we get fewer terms
    for start_min in [5, 6, 7, 8, 9, 10]:
        if time.time() > deadline - 5:
            break
        result = greedy_egyptian(Fraction(1, 1), min_denom=start_min)
        if result is not None and len(result) <= 256:
            total = sum(Fraction(1, d) for d in result)
            if total == 1 and all(5 <= d <= 10000000 for d in result) and len(set(result)) == len(result):
                return result
    
    # If greedy fails, use a simple DP fallback
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if time.time() > deadline:
            break
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
            
    raise RuntimeError("no baseline representation")
