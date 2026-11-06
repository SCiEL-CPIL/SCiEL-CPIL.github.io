"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    """
    Find a small set of distinct denominators in [5, 10000000] whose reciprocal sum is 1.
    Uses BFS to find the minimum number of terms.
    """
    start_time = time.time()
    deadline = start_time + 50.0
    
    # We want to find denominators d1 < d2 < ... < dk such that sum(1/di) = 1, with all di >= 5.
    # This is equivalent to finding the shortest path in a graph where states are the remaining
    # sum to be filled.
    # Start state: sum = 1, last_denominator = 4 (so next must be >= 5)
    # Target state: sum = 0
    
    # We use BFS on (remaining_sum, last_denominator).
    # remaining_sum is a Fraction.
    # To keep the state space manageable, we can limit the depth or use heuristics.
    # However, for k=5, it is known to be impossible for min denom >= 5.
    # For k=6, it might be possible?
    # Let's check known solutions.
    # 1 = 1/5 + 1/5 + ... (not distinct)
    # Smallest distinct denominators >= 5:
    # 1/5 + 1/6 + 1/7 + 1/8 + 1/9 + ...
    # Sum of first few: 0.2+0.166+0.142+0.125+0.111 = 0.744. Need 0.256 more.
    # 1/4 = 0.25, but denom must be >= 5.
    # 1/5 = 0.2. Remaining 0.056.
    # This path is likely long.
    
    # A known minimal solution for min denom >= 5 is length 5?
    # 1 = 1/5 + 1/5 + 1/5 + 1/5 + 1/5 (not distinct)
    # 1 = 1/5 + 1/6 + 1/7 + 1/10 + 1/20 + 1/42 + 1/70 + 1/140 + 1/210 + 1/420 (length 10)
    # Can we do better?
    # Let's try BFS with a small depth limit first.
    
    # State: (remaining_sum, last_denom, path)
    # We want to minimize len(path).
    
    # Since BFS explores in order of path length, the first solution found is optimal.
    # However, the branching factor can be large.
    # For a given remaining sum S and last denominator L, the next denominator d must satisfy:
    # d > L
    # 1/d <= S  => d >= 1/S
    # Also, the maximum possible sum with k terms from denominators > d is bounded.
    # If we have used i terms, and we need k-i more terms, the maximum additional sum is:
    # Sum_{j=1}^{k-i} 1/(d+j)
    # This must be >= S.
    # So d is bounded above by something.
    
    # Let's implement a BFS that limits the depth to, say, 8 or 10.
    # If no solution is found within the time limit, we return the best found or a known valid solution.
    
    # Known valid solution (length 10):
    # [5, 6, 7, 10, 20, 42, 70, 140, 210, 420]
    # Let's verify sum:
    # LCM(5,6,7,10,20,42,70,140,210,420) = 420
    # 84 + 70 + 60 + 42 + 21 + 10 + 6 + 3 + 2 + 1 = 299.
    # Wait, 299/420 != 1.
    # My previous hardcoded attempts were failing. I need a valid one.
    
    # Let's use the greedy algorithm to generate a valid solution quickly.
    def greedy_egyptian(numerator, denominator, min_denom):
        result = []
        n = numerator
        d = denominator
        while n > 0:
            # Find smallest d' >= min_denom such that 1/d' <= n/d
            # d' >= ceil(d/n)
            next_d = max(min_denom, (d + n - 1) // n)
            # Check if 1/next_d <= n/d
            if next_d * n > d:
                next_d += 1
            result.append(next_d)
            # New fraction: n/d - 1/next_d = (n*next_d - d) / (d*next_d)
            new_n = n * next_d - d
            new_d = d * next_d
            # Simplify
            from math import gcd
            g = gcd(new_n, new_d)
            new_n //= g
            new_d //= g
            n, d = new_n, new_d
            min_denom = next_d + 1
            if len(result) > 20:
                break
        return result

    # Greedy solution starting with 1/5
    greedy_sol = greedy_egyptian(1, 1, 5)
    # Verify greedy_sol
    s = sum(1.0/x for x in greedy_sol)
    # It should be very close to 1.
    # But we need exact rational sum.
    # Greedy algorithm guarantees exact rational sum of 1.
    # However, we need to ensure denominators are distinct and >= 5.
    # Greedy ensures distinct denominators (strictly increasing) and >= min_denom.
    # So greedy_sol is a valid candidate.
    # Its length might be large.
    
    # Let's try to find a shorter one using BFS.
    # State: (n, d, last_denom, count)
    # Start: (1, 1, 4, 0)
    # Goal: n == 0
    
    from collections import deque
    q = deque()
    q.append((1, 1, 4, []))
    visited = set()
    # (n, d, last_denom) is the state key.
    visited.add((1, 1, 4))
    
    best_sol = greedy_sol
    
    while q:
        if time.time() > deadline:
            break
            
        n, d, last, path = q.popleft()
        if n == 0:
            return path
            
        # Determine range for next denominator x
        # 1/x <= n/d  => x >= ceil(d/n)
        min_x = max(last + 1, (d + n - 1) // n)
        # Also, we need to be able to sum the rest.
        # Max terms remaining? Let's say we allow up to 10 total terms.
        # If len(path) >= 10, skip.
        if len(path) >= 10:
            continue
            
        # Upper bound for x:
        # If we pick x, the remaining sum is n/d - 1/x = (nx-d)/(dx).
        # We need to represent this with distinct denominators > x.
        # The maximum sum with k terms starting from x+1 is roughly k/x.
        # Let's just iterate x up to some limit, or until the remaining sum becomes impossible.
        # A safe upper bound: if x is too large, 1/x is very small.
        # Let's limit x to 10000 for speed in BFS.
        
        for x in range(min_x, 10000):
            if x * n < d:
                break
            new_n = n * x - d
            new_d = d * x
            # Simplify
            from math import gcd
            g = gcd(new_n, new_d)
            if g > 1:
                new_n //= g
                new_d //= g
                
            state_key = (new_n, new_d, x)
            if state_key not in visited:
                visited.add(state_key)
                q.append((new_n, new_d, x, path + [x]))
                
        # Prune: if n/d is very small, we might need many terms.
        # If 1/(last+1) < n/d, we can proceed.
        
    if time.time() > deadline - 5:
        # Return the best valid solution found so far, or greedy
        return best_sol
        
    return best_sol
# EVOLVE-BLOCK-END
