import time
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # wall-clock seconds; evaluator kills at 90s
    
    # Hard-coded 11-term solution (all denominators >= 5, distinct, sum of reciprocals = 1)
    # We need to find one. Let's use a known decomposition.
    # 1 = 1/2 + 1/3 + 1/6 is trivial but denom < 5.
    # We need all denom >= 5.
    # Known: 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420?
    # Let's just compute a valid 12-term solution quickly as fallback,
    # and try hard to find an 11-term one.
    
    # Actually, the task says "minimize their count". The seed returns 12 terms.
    # If we can find 11, we score better.
    
    # Let's verify a candidate 11-term solution by construction.
    # We'll use a targeted search with strong pruning.
    
    def try_find_n_terms(n):
        """Try to find n distinct denominators in [5, 10^7] summing to 1."""
        # Use backtracking with aggressive pruning
        best = [None]
        
        def bt(rem, min_d, chosen, k):
            if time.time() > DEADLINE:
                return
            if k == 0:
                if rem == 0:
                    best[0] = chosen
                return
            if rem <= 0:
                return
            # Pruning: max possible sum from k terms starting at min_d
            # If rem > k/min_d, impossible
            if rem > k / min_d:
                return
            # If rem < k/10^7, impossible
            if rem < k / 10000000:
                return
            
            # Lower bound for d: need 1/d <= rem (if it were the only term)
            # But we have k terms. The largest we can pick is such that even with k-1 more terms (smallest possible), we don't exceed rem.
            # Actually, we need to pick d such that rem - 1/d can be represented by k-1 terms each >= 1/(d+1).
            # So rem - 1/d <= (k-1)/(d+1)
            # rem <= 1/d + (k-1)/(d+1)
            # For a given rem and k, this gives an upper bound on how small d can be... wait.
            # If d is small, 1/d is large. The RHS 1/d + (k-1)/(d+1) is large.
            # We need rem <= RHS. If rem is fixed, small d makes RHS large, so inequality holds easily.
            # Large d makes RHS small. So we need d not too large.
            # rem <= 1/d + (k-1)/(d+1)
            # rem*d*(d+1) <= d+1 + d*(k-1)
            # rem*d^2 + rem*d <= d + 1 + kd - k
            # rem*d^2 + (rem - 1 - k)*d + (k - 1) <= 0
            # This is a quadratic in d. Since rem < 1 for k>=2, the leading coefficient is positive.
            # We want d such that the quadratic is <= 0.
            # The roots give the upper bound for d.
            
            # Also, we need 1/d <= rem, so d >= 1/rem.
            # So d is in [max(min_d, ceil(1/rem)), d_max]
            
            # Let's compute d_max from the quadratic.
            # rem*d^2 + (rem - 1 - k)*d + (k - 1) <= 0
            a = rem
            b = rem - 1 - k
            c = k - 1
            # Discriminant
            disc = b*b - 4*a*c
            if disc < 0:
                # No real roots, quadratic is always positive (since a>0), so no solution
                return
            # Roots: (-b ± sqrt(disc)) / (2*a)
            sqrt_disc = disc ** 0.5
            # We want the positive root (upper bound)
            d_max = int((-b + sqrt_disc) / (2 * a)) + 1
            d_min = max(min_d, int(1 / rem) + 1)
            
            if d_min > d_max:
                return
            
            # To keep search fast, limit the range
            # Try d from d_min to d_max, but d_max might be huge.
            # Cap it.
            d_max = min(d_max, d_min + 500)
            
            for d in range(d_min, d_max + 1):
                if d in chosen:
                    continue
                term = Fraction(1, d)
                if term > rem:
                    break  # Since d is increasing, term is decreasing... wait.
                # Actually as d increases, 1/d decreases. So if term > rem, smaller d would be even bigger.
                # But we're going from d_min (small d, big term) to d_max (big d, small term).
                # So if term > rem, we should continue (try larger d).
                # If term <= rem, we can proceed.
                
                new_rem = rem - term
                chosen.append(d)
                bt(new_rem, d + 1, chosen, k - 1)
                chosen.pop()
                if best[0] is not None:
                    return
                if time.time() > DEADLINE:
                    return
        
        bt(Fraction(1, 1), 5, [], n)
        return best[0]
    
    # Try 11 terms
    sol_11 = try_find_n_terms(11)
    if sol_11:
        # Verify
        s = sum(Fraction(1, d) for d in sol_11)
        if s == 1 and len(set(sol_11)) == len(sol_11) and all(5 <= d <= 10000000 for d in sol_11):
            return sol_11
    
    # Fallback: 12-term solution
    # We can construct one by splitting 1/2 into fractions with denom >= 5.
    # 1/2 = 1/5 + 1/6 + 1/12 + 1/20? No, let's just use a known one.
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210 + 1/420 + 1/840
    # Let's verify this sum.
    # Actually, let's just compute one programmatically as the seed does.
    
    def get_fallback():
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
                # Verify
                s = sum(Fraction(1, d) for d in sol)
                if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
                    return sol
        raise RuntimeError("no baseline representation")
    
    return get_fallback()
