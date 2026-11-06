import time
from fractions import Fraction

def construct():
    deadline = time.time() + 80
    
    # We want to minimize the count k.
    # Theoretical minimum k is 8.
    # We will try k=8, 9, 10... until we find a solution.
    
    def find_solution(k):
        # dfs(idx, remaining, last_denom)
        # remaining: Fraction
        # last_denom: int, the last denominator used, next must be > last_denom
        # idx: current index in the list of denominators (0-based)
        
        best_sol = None
        
        def dfs(idx, remaining, last_denom):
            nonlocal best_sol
            if best_sol is not None:
                return
            if time.time() > deadline:
                return
            
            if idx == k:
                if remaining == 0:
                    best_sol = current_path[:]
                return
            
            if remaining <= 0:
                return
                
            # Pruning:
            # 1. Max possible sum with remaining slots
            #    Using the smallest possible distinct denominators: last_denom+1, last_denom+2, ...
            max_possible_sum = Fraction(0)
            for i in range(1, k - idx + 1):
                max_possible_sum += Fraction(1, last_denom + i)
            
            if remaining > max_possible_sum:
                return
            
            # 2. Min possible sum with remaining slots?
            #    We need to pick k-idx distinct integers > last_denom.
            #    The smallest sum is achieved by picking the largest possible integers? No, we don't have an upper bound on denominators except 10^7.
            #    However, if remaining is very small, we need large denominators.
            #    If 1/(last_denom + 1) > remaining, we can still pick a larger denominator.
            #    So the constraint is: there must exist d > last_denom such that 1/d <= remaining.
            #    This is always true for large enough d.
            #    But we also need to fill the rest.
            
            # Start from last_denom + 1
            # Upper bound for d:
            # 1/d <= remaining  => d >= 1/remaining
            # Also, we need to leave enough sum for the remaining k-idx-1 terms.
            # Min sum for remaining k-idx-1 terms (using smallest possible denominators d+1, d+2...)
            # must be <= remaining - 1/d.
            
            start_d = last_denom + 1
            # Optimization: if 1/start_d > remaining, we can jump to ceil(1/remaining)
            if Fraction(1, start_d) > remaining:
                start_d = int(1 / float(remaining)) + 1
                if start_d <= last_denom:
                    start_d = last_denom + 1

            for d in range(start_d, 10000000): # Cap at 10^7, but loop will break earlier
                if time.time() > deadline:
                    return
                frac = Fraction(1, d)
                if frac > remaining:
                    continue
                
                new_remaining = remaining - frac
                if new_remaining < 0:
                    continue
                
                # Check if new_remaining can be formed by k-idx-1 terms > d
                # Max sum check for next step
                if k - idx - 1 > 0:
                    # Max sum of next k-idx-1 terms starting from d+1
                    max_next = Fraction(0)
                    for j in range(1, k - idx):
                        max_next += Fraction(1, d + j)
                    if new_remaining > max_next:
                        # Even with smallest possible next denominators, we can't reach the target
                        # Wait, if new_remaining > max_next, it means we need MORE sum, but we can only get max_next.
                        # So we should skip this d? No, if we pick a smaller d, we get a larger frac, leaving a smaller new_remaining.
                        # If new_remaining is too large, we need a smaller d (larger frac).
                        # But we are iterating d upwards. So if d is too small (frac too large), new_remaining is small.
                        # If d is too large (frac too small), new_remaining is large.
                        # Here, if new_remaining > max_next, it means we left too much sum for the remaining terms.
                        # This implies our current d is too large (frac too small).
                        # Since we are iterating d in increasing order, subsequent d's will be even larger, frac even smaller, new_remaining even larger.
                        # So we can break?
                        # Yes, if new_remaining > max_next, increasing d will only increase new_remaining.
                        break
                else:
                    # Last term
                    if new_remaining != 0:
                        continue

                current_path.append(d)
                dfs(idx + 1, new_remaining, d)
                current_path.pop()
                
                if best_sol is not None:
                    return

        current_path = []
        dfs(0, Fraction(1, 1), 4)
        return best_sol

    # Try k=8 first
    sol = find_solution(8)
    if sol:
        return sol
        
    # If not found, try k=9
    sol = find_solution(9)
    if sol:
        return sol

    # Fallback: return a known valid solution if search fails (unlikely)
    # Example: 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12+1/13... 
    # Actually, just raise error or return empty if all fail, but we should find one.
    return []
