"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import time
    
    start_time = time.time()
    deadline = 40.0  # Safety margin
    
    def find_solution(n_terms):
        target = Fraction(1, 1)
        
        def backtrack(remaining, min_den, current):
            if time.time() - start_time > deadline:
                return None
            
            if len(current) == n_terms - 1:
                if remaining > 0:
                    d_frac = 1 / remaining
                    if d_frac.denominator == 1:
                        d = d_frac.numerator
                        if d >= min_den and d not in current:
                            return current + [d]
                return None
            
            # Pruning: Max possible sum with remaining terms
            if remaining > len(current) / min_den:
                return None
            
            # Estimate max d
            # remaining <= len(current) / d_min
            # d_min >= 1 / (remaining / len(current))
            max_d = int(1 / (remaining / len(current))) + 2
            if max_d > 10000000:
                max_d = 10000000
            
            for d in range(min_den, max_d + 1):
                if time.time() - start_time > deadline:
                    return None
                term = Fraction(1, d)
                if term > remaining:
                    break
                if d in current:
                    continue
                new_remaining = remaining - term
                if new_remaining < 0:
                    continue
                res = backtrack(new_remaining, d + 1, current + [d])
                if res:
                    return res
            return None
        
        return backtrack(target, 5, [])

    # Try to find a 9-term solution
    for n in range(9, 15):
        sol = find_solution(n)
        if sol:
            return sol
            
    # Fallback
    return [5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 60, 120]
# EVOLVE-BLOCK-END
