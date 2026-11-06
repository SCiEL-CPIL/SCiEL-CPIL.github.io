"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    def verify(denoms):
        """Check if denominators are valid and sum to 1."""
        if not denoms or len(denoms) > 256:
            return False
        if len(set(denoms)) != len(denoms):
            return False
        for d in denoms:
            if d < 5 or d > 10000000:
                return False
        s = sum(Fraction(1, d) for d in denoms)
        return s == 1

    # Try to find minimal solutions with increasing term count
    # Use DFS with pruning
    
    def search_minimal(k, deadline):
        """Find representation of 1 as sum of k distinct unit fractions, denominators >= 5."""
        def dfs(remaining, min_d, terms_left, current):
            if time.time() > deadline:
                return None
            if terms_left == 0:
                return current if remaining == 0 else None
            if remaining == 0:
                return current if terms_left == 0 else None
            if remaining <= 0:
                return None
            
            # Lower bound for next denominator: 1/d < remaining => d > 1/remaining
            # Also d >= min_d
            d_min = max(min_d, 5)
            
            # Upper bound: we need remaining - 1/d to be achievable with (terms_left-1) terms
            # Each term <= 1/min_d, so remaining - 1/d <= (terms_left-1) * (1/min_d)
            # => 1/d >= remaining - (terms_left-1)/min_d
            # If remaining - (terms_left-1)/min_d > 0, then d <= 1/(remaining - (terms_left-1)/min_d)
            
            max_possible = Fraction(terms_left - 1, min_d) if min_d > 0 else 0
            if remaining > max_possible:
                return None
            
            # d must satisfy: 1/d < remaining => d > 1/remaining
            # And: remaining - 1/d > 0 => d > 1/remaining (same)
            # Also: remaining - 1/d must be >= 0, so d >= 1/remaining
            # For strict inequality (need more terms), d > 1/remaining
            
            # If this is the last term, 1/d must equal remaining exactly
            if terms_left == 1:
                if remaining.numerator == 1:
                    d = remaining.denominator
                    if d >= d_min and d >= 5 and d <= 10000000:
                        return current + [d]
                return None
            
            # d_min from remaining: d > 1/remaining
            # 1/remaining = remaining.denominator / remaining.numerator
            # d > remaining.denominator / remaining.numerator
            # d >= floor(remaining.denominator / remaining.numerator) + 1
            d_min_from_rem = remaining.denominator // remaining.numerator + 1
            d_min = max(d_min, d_min_from_rem)
            
            # Upper bound on d: we need remaining - 1/d to be representable
            # with (terms_left - 1) terms, each >= 1/10000000 (smallest possible reciprocal)
            # Actually each term is at most 1/d_min_next where d_min_next > d
            # But simpler: remaining - 1/d must be > 0, so d > 1/remaining (already handled)
            # And remaining - 1/d must be achievable: 
            # remaining - 1/d <= (terms_left-1) / (d+1)  [since next denom > d]
            # This gives: 1/d >= remaining - (terms_left-1)/(d+1)
            # Let's just cap d at a reasonable value
            
            # Upper bound: 1/d >= remaining - (terms_left-1) * (1/(d+1))
            # For safety, cap d at min(10000000, int(1/remaining) + 1000)
            # But 1/remaining could be large...
            
            # Better upper bound: since next denominators must be > d,
            # the maximum sum of remaining (terms_left-1) terms is (terms_left-1)/(d+1)
            # So we need: remaining - 1/d <= (terms_left-1)/(d+1)
            # => (d+1)(remaining - 1/d) <= terms_left-1
            # => (d+1)*remaining - (d+1)/d <= terms_left-1
            # => (d+1)*remaining - 1 - 1/d <= terms_left-1
            # => (d+1)*remaining <= terms_left + 1/d
            # => d+1 <= (terms_left + 1/d) / remaining
            # => d <= (terms_left + 1/d) / remaining - 1
            
            # This is implicit in d. Let's use a simpler bound:
            # d must be such that remaining - 1/d > 0, i.e., d > 1/remaining
            # And remaining - 1/d must be at least (terms_left-1) * (1/10000000)
            # (since each remaining term is at least 1/10000000)
            # So: remaining - 1/d >= (terms_left-1)/10000000
            # => 1/d <= remaining - (terms_left-1)/10000000
            # => d >= 1 / (remaining - (terms_left-1)/10000000)
            
            # This gives a lower bound, not upper. For upper bound on d:
            # We just need d to be reasonable. Let's cap at 10000000.
            
            d_max = 10000000
            
            # But we can tighten: since all remaining terms have denominators > d,
            # the sum of remaining (terms_left-1) terms is at most (terms_left-1)/(d+1)
            # So we need: remaining - 1/d <= (terms_left-1)/(d+1)
            # Let's solve for d:
            # (d+1)(remaining - 1/d) <= terms_left - 1
            # (d+1)*remaining - (d+1)/d <= terms_left - 1
            # (d+1)*remaining - 1 - 1/d <= terms_left - 1
            # (d+1)*remaining <= terms_left + 1/d
            # For large d, 1/d is small, so approximately:
            # d+1 <= terms_left / remaining
            # d <= terms_left / remaining - 1
            
            if remaining > 0:
                approx_d_max = (terms_left * remaining.denominator) // remaining.numerator
                if approx_d_max > d_min:
                    d_max = min(d_max, approx_d_max + 10)
            
            current_set = set(current)
            
            for d in range(d_min, d_max + 1):
                if d in current_set:
                    continue
                if time.time() > deadline:
                    return None
                
                new_rem = remaining - Fraction(1, d)
                if new_rem < 0:
                    break
                if new_rem == 0:
                    if terms_left == 1:
                        return current + [d]
                    else:
                        # Can't have zero remaining with terms left
                        continue
                
                result = dfs(new_rem, d + 1, terms_left - 1, current + [d])
                if result is not None:
                    return result
            
            return None
        
        return dfs(Fraction(1, 1), 5, k, [])
    
    # Try increasing k starting from small values
    # Minimum k: we need sum of k reciprocals (each <= 1/5) to equal 1
    # So k >= 5 (since 5 * 1/5 = 1, but we need distinct denominators)
    # With distinct denominators >= 5, the maximum sum for k terms is 1/5 + 1/6 + ... + 1/(5+k-1)
    # For k=5: 1/5+1/6+1/7+1/8+1/9 ≈ 0.2+0.167+0.143+0.125+0.111 = 0.746 < 1
    # For k=6: add 1/10 ≈ 0.1, total ≈ 0.846 < 1
    # For k=7: add 1/11 ≈ 0.091, total ≈ 0.937 < 1
    # For k=8: add 1/12 ≈ 0.083, total ≈ 1.02 > 1, so k=8 might work
    
    best_result = None
    for k in range(5, 20):
        if time.time() > deadline - 3.0:
            break
        result = search_minimal(k, deadline - 2.0)
        if result is not None:
            if verify(result):
                best_result = result
                break
    
    if best_result is None:
        # Fallback: use DP approach from current program
        target, reached = 2520, {0: []}
        for d in range(5, target + 1):
            if time.time() > deadline - 1.0:
                break
            if target % d:
                continue
            w = target // d
            for total, values in list(reached.items()):
                if total + w <= target and total + w not in reached:
                    reached[total + w] = values + [d]
            if target in reached:
                best_result = reached[target]
                break
    
    if best_result is None:
        # Last resort: hardcoded valid solution
        # Let's construct one using the DP result or a known identity
        # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 + 1/42 + 1/70 + 1/105 + 1/210
        # Let me verify: LCM = 210
        # 210/5=42, 210/6=35, 210/7=30, 210/10=21, 210/14=15, 210/35=6, 210/42=5, 210/70=3, 210/105=2, 210/210=1
        # Sum = 42+35+30+21+15+6+5+3+2+1 = 160. No.
        
        # Use a simple valid one found by DP or just return a longer valid solution
        # 1 = 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + 1/10 + 1/11 + 1/12 + 1/13 + 1/14 + 1/15 + 1/16 + 1/17 + 1/18 + 1/19 + 1/20 + 1/21 + 1/22 + 1/23 + 1/24
        # This likely doesn't sum to exactly 1.
        
        # Just use the DP fallback result if available, otherwise raise
        raise RuntimeError("no valid representation found")
    
    return best_result
# EVOLVE-BLOCK-END
