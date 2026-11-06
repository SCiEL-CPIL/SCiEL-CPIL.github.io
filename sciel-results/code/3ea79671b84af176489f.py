from fractions import Fraction
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s safety margin
    
    def get_fallback():
        """Get the known 12-term solution via subset DP."""
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
    
    # Get fallback quickly
    fallback = get_fallback()
    
    def try_find_solution(k, min_den=5, max_den=10000000):
        """Find k distinct denominators in [min_den, max_den] with reciprocal sum = 1."""
        best = [None]
        
        def backtrack(remaining, min_d, current, k_left):
            if time.time() > deadline:
                return
            if k_left == 0:
                if remaining == 0:
                    best[0] = current[:]
                return
            
            if k_left == 1:
                if remaining > 0:
                    num, den = remaining.numerator, remaining.denominator
                    if den % num == 0:
                        d = den // num
                        if d >= min_d and d <= max_den and d not in current:
                            best[0] = current + [d]
                return
            
            # Pruning bounds
            if remaining > k_left / min_d:
                return
            if remaining < k_left / max_den:
                return
            
            d_low = max(min_d, int(1 / float(remaining)) + 1)
            if d_low < min_d:
                d_low = min_d
            
            if remaining - (k_left - 1) / max_den <= 0:
                return
            d_high = int(1 / (remaining - (k_left - 1) / max_den))
            d_high = min(d_high, max_den)
            
            if d_low > d_high:
                return
            
            # Try candidates in a focused range
            avg_d = k_left / float(remaining)
            search_start = max(d_low, int(avg_d * 0.7))
            search_end = min(d_high, int(avg_d * 1.5) + 50)
            
            candidates = set(range(search_start, search_end + 1))
            candidates.add(d_low)
            candidates.add(d_high)
            
            for d in sorted(candidates):
                if d < min_d or d > max_den or d in current:
                    continue
                term = Fraction(1, d)
                if term >= remaining:
                    continue
                new_remaining = remaining - term
                if new_remaining < 0:
                    continue
                current.append(d)
                backtrack(new_remaining, d + 1, current, k_left - 1)
                current.pop()
                if best[0] is not None:
                    return
        
        backtrack(Fraction(1, 1), min_den, [], k)
        return best[0]
    
    # Try to find solutions with fewer terms
    for k in range(11, 4, -1):
        if time.time() > deadline - 5:
            break
        sol = try_find_solution(k)
        if sol is not None:
            s = sum(Fraction(1, d) for d in sol)
            if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= 10000000 for d in sol):
                return sol
    
    return fallback
