import time
from fractions import Fraction

def construct():
    deadline = time.time() + 80  # Safety margin within 90s
    
    def dfs(remaining, min_d, count, path):
        """Find 'count' more distinct denominators >= min_d that sum to 'remaining'."""
        if time.time() > deadline:
            return None
        if count == 0:
            if remaining == 0:
                return list(path)
            return None
        if remaining <= 0:
            return None
        
        # Pruning: max possible sum with 'count' terms starting from min_d
        if remaining > Fraction(count, min_d):
            return None
        
        # Lower bound for d: 1/d <= remaining -> d >= 1/remaining
        d_start = max(min_d, (remaining.numerator + remaining.denominator - 1) // remaining.denominator)
        
        # Upper bound for d:
        # We need remaining - 1/d <= (count-1)/(d+1)
        # remaining <= 1/d + (count-1)/(d+1)
        # For large d, LHS ~ (1 + count-1)/d = count/d
        # So d <= count / remaining
        # But we can be tighter: 1/d + (count-1)/(d+1) >= remaining
        # Let's use d_end = int(count * remaining.numerator / remaining.denominator) + 5
        d_end = int((count * remaining.numerator) // remaining.denominator) + 5
        d_end = min(d_end, 10000000)
        
        # If d_start > d_end, no solution
        if d_start > d_end:
            return None

        for d in range(d_start, d_end + 1):
            if time.time() > deadline:
                return None
            if d in path:
                continue
            
            # Check if 1/d <= remaining
            if Fraction(1, d) > remaining:
                continue
            
            new_remaining = remaining - Fraction(1, d)
            
            if new_remaining < 0:
                continue
                
            # Feasibility check for remaining 'count-1' terms
            if count - 1 == 0:
                if new_remaining == 0:
                    return list(path) + [d]
                else:
                    continue
            
            # Max sum of (count-1) terms from d+1 is (count-1)/(d+1)
            if new_remaining > Fraction(count - 1, d + 1):
                break
            
            path.append(d)
            result = dfs(new_remaining, d + 1, count - 1, path)
            path.pop()
            if result is not None:
                return result
        
        return None
    
    # Try n from 3 upward
    for n in range(3, 10):
        if time.time() > deadline:
            break
        result = dfs(Fraction(1), 5, n, [])
        if result is not None:
            # Verify
            s = sum(Fraction(1, d) for d in result)
            if s == 1 and len(set(result)) == n and all(5 <= d <= 10000000 for d in result):
                return result
    
    # Fallback: DP over divisors of a large number
    target = 2520
    reached = {0: []}
    for d in range(5, target + 1):
        if target % d:
            continue
        w = target // d
        for total in list(reached.keys()):
            new_total = total + w
            if new_total <= target and new_total not in reached:
                reached[new_total] = reached[total] + [d]
        if target in reached:
            sol = reached[target]
            s = sum(Fraction(1, d) for d in sol)
            if s == 1 and all(5 <= d <= 10000000 for d in sol) and len(set(sol)) == len(sol):
                return sol

    # Last resort: hardcoded valid solution
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35
    # Let me verify: LCM(5,6,7,10,14,35) = 210
    # 210/5 + 210/6 + 210/7 + 210/10 + 210/14 + 210/35
    # = 42 + 35 + 30 + 21 + 15 + 6 = 149. Not 210.
    # Let me find a valid one.
    # 1 = 1/5 + 1/6 + 1/7 + 1/42
    # LCM(5,6,7,42) = 210
    # 210/5 + 210/6 + 210/7 + 210/42 = 42 + 35 + 30 + 5 = 112. Not 210.
    # Let me use a known identity.
    # 1 = 1/5 + 1/6 + 1/8 + 1/120 + 1/1440 + 1/10080
    # This is getting complicated. Let me just return a valid set.
    # Actually, let me use the greedy algorithm to generate a valid solution.
    # Greedy: 1 = 1/5 + 4/5
    # 4/5 = 1/6 + 14/30 = 1/6 + 7/15
    # 7/15 = 1/7 + 4/105
    # 4/105 = 1/27 + 19/2835
    # This will take many terms. Let me just return a known valid solution.
    
    # Known valid: 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/14 + 1/35 doesn't work.
    # Let me compute: 1/5 + 1/6 = 11/30
    # 11/30 + 1/7 = (77+30)/210 = 107/210
    # 107/210 + 1/10 = (1070+210)/2100 = 1280/2100 = 128/210
    # 128/210 + 1/14 = (1792+210)/2940 = 2002/2940 = 1001/1470
    # 1001/1470 + 1/35 = (1001+42)/1470 = 1043/1470. Not 1.
    
    # Let me just use a simpler approach: return [5, 6, 7, 42] if it works, else use DP result.
    # Actually, the DP should find a solution. Let me trust it.
    # If DP fails, raise error.
    raise RuntimeError("no valid representation found")
