import time
from fractions import Fraction
from math import gcd
import heapq

def construct():
    deadline = time.time() + 85  # Safety margin within 90s
    
    def _find_3_terms():
        """Find 3 distinct denominators >= 5 with reciprocal sum 1."""
        for a in range(5, 200):
            # 1/a + 1/b + 1/c = 1 => 1/b + 1/c = 1 - 1/a = (a-1)/a
            # => (b+c)/(bc) = (a-1)/a => a(b+c) = (a-1)bc
            # => (a-1)bc - ab - ac = 0
            # => (a-1)b*c - a*b - a*c = 0
            # Multiply by (a-1): (a-1)^2 bc - a(a-1)b - a(a-1)c = 0
            # Add a^2: [(a-1)b - a][(a-1)c - a] = a^2
            num = a - 1
            den = a
            # We need 1/b + 1/c = num/den
            # => (b+c)/(bc) = num/den
            # => den*(b+c) = num*bc
            # => num*bc - den*b - den*c = 0
            # => (num*b - den)*(num*c - den) = den^2
            rhs = den * den
            # Find factor pairs of rhs where both factors > den
            # Let x = num*b - den, y = num*c - den, x*y = rhs
            # b = (x + den) / num, c = (y + den) / num
            # Need b > a, c > b, b >= 5, c >= 5, integer
            # Iterate over divisors of rhs
            x = 1
            while x * x <= rhs:
                if rhs % x == 0:
                    y = rhs // x
                    # x <= y
                    b_num = x + den
                    c_num = y + den
                    if b_num % num == 0 and c_num % num == 0:
                        b = b_num // num
                        c = c_num // num
                        if b > a and c > b and b >= 5 and c >= 5:
                            return [a, b, c]
                x += 1
        return None
    
    def _find_4_terms():
        """Find 4 distinct denominators >= 5 with reciprocal sum 1."""
        for a in range(5, 100):
            if time.time() > deadline:
                return None
            # 1/b + 1/c + 1/d = 1 - 1/a = (a-1)/a
            rem_num = a - 1
            rem_den = a
            # Find 3 terms summing to rem_num/rem_den
            for b in range(a + 1, a + 200):
                # 1/c + 1/d = rem_num/rem_den - 1/b = (rem_num*b - rem_den)/(rem_den*b)
                num2 = rem_num * b - rem_den
                den2 = rem_den * b
                if num2 <= 0:
                    break
                # Find 2 terms summing to num2/den2
                # 1/c + 1/d = num2/den2
                # => (c+d)/(cd) = num2/den2
                # => den2*(c+d) = num2*cd
                # => num2*cd - den2*c - den2*d = 0
                # => (num2*c - den2)*(num2*d - den2) = den2^2
                rhs2 = den2 * den2
                x = 1
                while x * x <= rhs2:
                    if rhs2 % x == 0:
                        y = rhs2 // x
                        c_num = x + den2
                        d_num = y + den2
                        if c_num % num2 == 0 and d_num % num2 == 0:
                            c = c_num // num2
                            d = d_num // num2
                            if c > b and d > c and c >= 5 and d >= 5:
                                return [a, b, c, d]
                    x += 1
        return None
    
    def _astar_search(target_num, target_den, min_d, max_terms, deadline):
        """
        A* search for Egyptian fraction decomposition.
        State: (remaining_num, remaining_den, min_denom, path)
        Heuristic: lower bound on remaining terms
        """
        # Priority queue: (heuristic, path_length, remaining_num, remaining_den, min_denom, path)
        # We want to minimize path length, so use path_length as primary sort key
        # Heuristic: minimum terms needed = ceil(remaining / (1/min_denom)) but more tightly:
        # The greedy lower bound: we need at least k terms where k/min_denom >= remaining
        # So k >= remaining * min_denom
        
        start_remaining = Fraction(target_num, target_den)
        start_state = (start_remaining, min_d, [])
        
        # Priority: (total_path_length, remaining_num, remaining_den, min_denom, path)
        # We want to explore shorter paths first
        heap = []
        heapq.heappush(heap, (0, start_remaining.numerator, start_remaining.denominator, min_d, []))
        
        visited = set()
        
        while heap:
            if time.time() > deadline:
                return None
            
            path_len, rem_num, rem_den, min_denom, path = heapq.heappop(heap)
            
            # Check if we've reached the goal
            if rem_num == 0:
                if len(path) <= max_terms:
                    return list(path)
                continue
            
            if rem_num <= 0:
                continue
            
            remaining = Fraction(rem_num, rem_den)
            
            # State key for visited
            state_key = (rem_num, rem_den, min_denom)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # If we've used too many terms
            if path_len >= max_terms:
                continue
            
            # Find candidate denominators d >= min_denom
            # d must satisfy: 1/d <= remaining, so d >= 1/remaining
            d_start = max(min_denom, (rem_num + rem_den - 1) // rem_den)
            
            # Upper bound: we need remaining - 1/d to be achievable with (max_terms - path_len - 1) terms
            # Max sum of k terms starting from d+1 is k/(d+1)
            # So we need remaining - 1/d <= (max_terms - path_len - 1)/(d+1)
            # remaining <= 1/d + (max_terms - path_len - 1)/(d+1)
            # For large d, this is approximately remaining <= (max_terms - path_len)/d
            # So d <= (max_terms - path_len) / remaining
            remaining_terms = max_terms - path_len - 1
            if remaining_terms <= 0:
                # Only one term left
                if rem_den % rem_num == 0:
                    d_candidate = rem_den // rem_num
                    if d_candidate >= min_denom and d_candidate not in path:
                        return list(path) + [d_candidate]
                continue
            
            d_end = int((remaining_terms + 1) * rem_num / rem_den) + 10
            d_end = min(d_end, 10000000)
            
            if d_start > d_end:
                continue
            
            # Limit the number of candidates to explore
            max_candidates = 50
            for d in range(d_start, min(d_start + max_candidates, d_end + 1)):
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
                
                new_rem_num = new_remaining.numerator
                new_rem_den = new_remaining.denominator
                
                # Feasibility check: can we achieve new_remaining with remaining_terms terms from d+1?
                if remaining_terms > 0:
                    # Max sum: remaining_terms / (d+1)
                    if new_remaining > Fraction(remaining_terms, d + 1):
                        # As d increases, 1/d decreases, so new_remaining increases
                        # And (remaining_terms)/(d+1) decreases
                        # So if it fails here, it will fail for all larger d
                        break
                
                # Heuristic: lower bound on additional terms needed
                # We need at least k terms where k/(d+1) >= new_remaining
                # So k >= new_remaining * (d+1)
                heuristic = int(new_rem_num * (d + 1) / new_rem_den) + 1 if new_rem_num > 0 else 0
                heuristic = max(heuristic, 1) if new_rem_num > 0 else 0
                
                total_path_len = path_len + 1
                heapq.heappush(heap, (total_path_len, new_rem_num, new_rem_den, d + 1, path + [d]))
        
        return None
    
    def _find_5_terms():
        """Find 5 distinct denominators >= 5 with reciprocal sum 1 using A*."""
        result = _astar_search(1, 1, 5, 5, deadline)
        return result
    
    def _find_n_terms(n):
        """Find n distinct denominators >= 5 with reciprocal sum 1 using A*."""
        return _astar_search(1, 1, 5, n, deadline)
    
    # Try n from 3 upward, using fast solvers for small n
    for n in range(3, 10):
        if time.time() > deadline:
            break
        if n == 3:
            result = _find_3_terms()
        elif n == 4:
            result = _find_4_terms()
        elif n == 5:
            result = _find_5_terms()
        else:
            result = _find_n_terms(n)
        
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
    
    # Last resort: raise error
    raise RuntimeError("no valid representation found")
