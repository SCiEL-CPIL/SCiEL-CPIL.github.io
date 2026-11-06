import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # wall-clock seconds; evaluator kills at 90s
    
    # Try to find an 11-term solution using backtracking with strong pruning
    # If that fails or takes too long, return a known 12-term solution
    
    def try_find_11_terms():
        """Try to find 11 distinct denominators in [5, 10^7] summing to 1."""
        best = [None]
        
        def bt(rem, min_d, chosen, k):
            if time.time() > DEADLINE:
                return
            if k == 0:
                if rem == 0:
                    best[0] = chosen[:]
                return
            if rem <= 0:
                return
            
            # Pruning: max possible sum from k terms starting at min_d
            if rem > k / min_d:
                return
            # If rem < k/10^7, impossible
            if rem < k / 10000000:
                return
            
            # Compute d_max from quadratic: rem <= 1/d + (k-1)/(d+1)
            # rem*d^2 + (rem - 1 - k)*d + (k - 1) <= 0
            a = rem
            b = rem - 1 - k
            c = k - 1
            disc = b*b - 4*a*c
            if disc < 0:
                return
            sqrt_disc = disc ** 0.5
            d_max = int((-b + sqrt_disc) / (2 * a)) + 1
            d_min = max(min_d, int(1 / rem) + 1)
            
            if d_min > d_max:
                return
            
            # Limit search range for speed
            d_max = min(d_max, d_min + 200)
            
            for d in range(d_min, d_max + 1):
                if d in chosen:
                    continue
                term = Fraction(1, d)
                if term > rem:
                    continue  # Try larger d
                if term <= 0:
                    break
                
                new_rem = rem - term
                chosen.append(d)
                bt(new_rem, d + 1, chosen, k - 1)
                chosen.pop()
                if best[0] is not None:
                    return
                if time.time() > DEADLINE:
                    return
        
        bt(Fraction(1, 1), 5, [], 11)
        return best[0]
    
    # Try 11 terms first
    sol_11 = try_find_11_terms()
    if sol_11:
        # Verify
        s = sum(Fraction(1, d) for d in sol_11)
        if s == 1 and len(set(sol_11)) == len(sol_11) and all(5 <= d <= 10000000 for d in sol_11):
            return sol_11
    
    # Fallback: known 12-term solution
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 + 1/840
    # Let's verify this sum:
    # Actually, let's use a simpler known solution or compute one.
    
    def get_fallback_12():
        # Use a known 12-term Egyptian fraction for 1 with all denominators >= 5
        # One such solution: 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 + 1/840
        # Let's verify this programmatically
        sol = [5, 6, 7, 10, 14, 35, 42, 70, 105, 210, 420, 840]
        s = sum(Fraction(1, d) for d in sol)
        if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
            return sol
        
        # If that doesn't work, compute one using dynamic programming
        target = 2520
        reached = {0: []}
        for d in range(5, target + 1):
            if target % d:
                continue
            w = target // d
            for total, values in list(reached.items()):
                if total + w <= target and total + w not in reached:
                    reached[total + w] = values + [d]
            if target in reached:
                sol = reached[target]
                s = sum(Fraction(1, d) for d in sol)
                if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
                    return sol
        raise RuntimeError("no baseline representation")
    
    return get_fallback_12()
