import math
import time

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in divisors:
        L = lcm(L, d)
        
    max_val = 1000000
    
    # We search for the minimum width W
    # Minimum width for 10 distinct integers is 9
    min_width = 9
    
    # To handle the constraint x <= 1000000, we just need L + W <= 1000000
    # Since we start L from 0, this is safe for small W.
    
    # Pre-calculate indices sorted by divisor size descending (fewer candidates first)
    # This helps backtracking prune faster
    sorted_indices = sorted(range(n), key=lambda i: divisors[i], reverse=True)
    
    def find_solution(width, start):
        # Candidates for each divisor in [start, start + width]
        candidates = [[] for _ in range(n)]
        for i in range(n):
            d = divisors[i]
            # First multiple >= start
            m_start = math.ceil(start / d)
            # Last multiple <= start + width
            m_end = (start + width) // d
            
            if m_start <= m_end:
                for m in range(m_start, m_end + 1):
                    candidates[i].append(m * d)
            else:
                return None # No candidate for this divisor
        
        # Backtracking search
        # Order of assignment: sorted_indices
        used = set()
        assignment = [0] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            i = sorted_indices[idx]
            for x in candidates[i]:
                if x not in used:
                    used.add(x)
                    assignment[i] = x
                    if backtrack(idx + 1):
                        return True
                    used.remove(x)
            return False
        
        if backtrack(0):
            return assignment
        return None

    # Deadline guard
    start_time = time.time()
    deadline = 80.0 # 80 seconds limit to be safe
    
    # Iterate over possible widths
    # The optimal width is likely small. 
    # Upper bound on width? 
    # In the worst case, we might need a width up to LCM, but likely much smaller.
    # Let's iterate W from 9 upwards.
    
    for width in range(min_width, L + 1):
        if time.time() > deadline:
            # Fallback: return a valid solution if we found one previously?
            # Or just return a heuristic solution.
            # For now, if we timeout, we return the divisors themselves (which is valid but not optimal)
            # But the loop should be fast enough.
            break
            
        # Iterate over starting positions in one fundamental domain
        for start in range(0, L):
            if start + width > max_val:
                break
                
            sol = find_solution(width, start)
            if sol is not None:
                # Ensure types are Python int
                return [int(x) for x in sol]
                
    # If no solution found in LCM range (should not happen), fallback
    # Just return the divisors (valid, distinct, <= 1000000)
    return [int(d) for d in divisors]
