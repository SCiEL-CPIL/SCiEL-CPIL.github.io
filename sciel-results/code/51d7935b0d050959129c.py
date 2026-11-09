"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85  # 5s safety margin before 90s kill
    
    # Try to find minimal Egyptian fraction with denominators >= 5
    # Start with small k and search upward
    
    # Known: need at least 7 terms (6 max sum < 1, 7 can exceed 1)
    # Try k=7 first with a bounded search
    
    def search_european(k, min_den=5):
        """Search for k distinct denominators >= min_den summing to 1"""
        from fractions import Fraction
        target = Fraction(1, 1)
        
        def dfs(start, remaining, terms, chosen):
            if time.time() > deadline:
                return None
            if len(terms) == k:
                if remaining == 0:
                    return chosen[:]
                return None
            if len(terms) == k - 1:
                # Last term: need 1/d = remaining, so d = 1/remaining
                if remaining > 0:
                    num, den = remaining.numerator, remaining.denominator
                    if den % num == 0:
                        d = den // num
                        if d > start and d >= 5 and d <= 10000000:
                            return chosen + [d]
                return None
            
            # Max possible sum with (k - len(terms)) terms starting from start+1
            # Use greedy upper bound: sum of 1/(start+i) for i in range(k-len(terms))
            max_possible = Fraction(0, 1)
            for i in range(k - len(terms)):
                max_possible += Fraction(1, start + 1 + i)
            if max_possible < remaining:
                return None
            
            # Min possible: 1/(start+1) + ... + 1/(start+k-len(terms))
            # Actually we need to bound the search
            # d must satisfy: remaining >= (k-len(terms)) / d  => d >= (k-len(terms))/remaining
            # And remaining <= (k-len(terms)) / (start+1) roughly
            
            # d must be > start
            # Also 1/d <= remaining, so d >= 1/remaining
            # And (k-len(terms))/d >= remaining for feasibility, so d <= (k-len(terms))/remaining
            
            if remaining <= 0:
                return None
            
            d_min = max(start + 1, int(1 / remaining) + 1)
            # d_max: we need (k-len(terms)) terms each >= 1/d, so sum >= (k-len(terms))/d >= remaining
            # => d <= (k-len(terms))/remaining
            d_max = int((k - len(terms)) / remaining)
            d_max = min(d_max, 10000000)
            
            if d_min > d_max:
                return None
            
            for d in range(d_min, d_max + 1):
                new_remaining = remaining - Fraction(1, d)
                if new_remaining < 0:
                    break
                result = dfs(d, new_remaining, terms + [d], chosen + [d])
                if result is not None:
                    return result
            return None
        
        return dfs(min_den - 1, target, [], [])
    
    # Try increasing k from 7
    for k in range(7, 13):
        result = search_european(k)
        if result is not None:
            # Validate
            from fractions import Fraction
            s = Fraction(0, 1)
            for d in result:
                s += Fraction(1, d)
            if s == 1 and len(set(result)) == len(result) and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback: use the DP approach from original
    target, reached = 2520, {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total, values in list(reached.items()):
            if total + w <= target and total + w not in reached:
                reached[total + w] = values + [d]
        if target in reached:
            return reached[target]
    raise RuntimeError("no baseline representation")
# EVOLVE-BLOCK-END
