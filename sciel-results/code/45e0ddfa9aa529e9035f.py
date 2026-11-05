"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Find minimum number of distinct denominators in [5, 10000000] with reciprocal sum = 1
    # WLOG denominators are in strictly increasing order
    
    # BFS by depth with strong pruning
    # State: (num, den, last_d) where num/den is remaining fraction, last_d is last chosen denominator
    # Next denominator d must satisfy: d > last_d, d >= ceil(den/num), d >= 5, d <= 10000000
    
    def find_solution(max_depth):
        """Find a solution with exactly max_depth terms using BFS."""
        # Start: remainder 1/1, last_d = 4
        # We'll use a list for the current level and build next level
        # Each state: (num, den, last_d, path)
        
        # To limit memory, we only keep states at the current depth
        current_level = [(1, 1, 4, [])]
        visited = set()
        visited.add((1, 1, 4))
        
        for depth in range(max_depth):
            if time.time() > deadline:
                return None
            
            next_level = []
            for num, den, last_d, path in current_level:
                if time.time() > deadline:
                    return None
                
                if num == 0:
                    return path
                
                # Compute valid range for next denominator
                # d >= ceil(den/num), d > last_d, d >= 5
                min_d_from_frac = (den + num - 1) // num
                min_d = max(last_d + 1, min_d_from_frac, 5)
                
                if min_d > 10000000:
                    continue
                
                # Also, we need at least (max_depth - depth - 1) more terms after this
                # Each remaining term contributes at most 1/min_d_next
                # So we need: num/den <= (max_depth - depth - 1) * 1/(min_d+1) ... roughly
                # This is a loose bound, skip for now
                
                # Generate candidates: try a limited set
                # Strategy: try min_d, and a few strategic values
                candidates = []
                
                # The minimum
                candidates.append(min_d)
                
                # Try min_d + 1 through min_d + 20
                for k in range(1, 21):
                    d = min_d + k
                    if d > 10000000:
                        break
                    candidates.append(d)
                
                # Try some "nice" values
                for d in [6, 7, 8, 9, 10, 12, 14, 15, 18, 20, 21, 24, 28, 30, 35, 40, 42, 48, 56, 60, 70, 84, 90, 105, 120, 140, 168, 210, 240, 280, 420, 840, 1260, 2520, 5040, 10080, 20160, 40320, 80640, 161280, 322560, 645120, 1290240, 2580480, 5160960]:
                    if d >= min_d and d <= 10000000 and d not in candidates:
                        candidates.append(d)
                
                # Limit to 50 candidates to control branching
                if len(candidates) > 50:
                    candidates = candidates[:50]
                
                for d in candidates:
                    if d < min_d or d > 10000000:
                        continue
                    
                    # New remainder
                    new_num = num * d - den
                    new_den = den * d
                    
                    if new_num < 0:
                        continue
                    
                    if new_num == 0:
                        return path + [d]
                    
                    # Reduce
                    g = math.gcd(new_num, new_den)
                    new_num //= g
                    new_den //= g
                    
                    state_key = (new_num, new_den, d)
                    if state_key in visited:
                        continue
                    
                    visited.add(state_key)
                    next_level.append((new_num, new_den, d, path + [d]))
            
            current_level = next_level
            if not current_level:
                break
        
        return None
    
    # Try increasing depths
    for depth in range(1, 15):
        if time.time() > deadline:
            break
        result = find_solution(depth)
        if result is not None and 1 <= len(result) <= 256:
            s = sum(Fraction(1, x) for x in result)
            if s == 1:
                return result
    
    # Fallback: greedy
    num, den = 1, 1
    used = set()
    result = []
    while num > 0:
        if time.time() > deadline:
            break
        min_d = (den + num - 1) // num
        if min_d < 5:
            min_d = 5
        d = min_d
        while d in used:
            d += 1
        if d > 10000000:
            break
        result.append(d)
        used.add(d)
        num = num * d - den
        den = den * d
        if num != 0:
            g = math.gcd(num, den)
            num //= g
            den //= g
    
    if num == 0 and 1 <= len(result) <= 256:
        s = sum(Fraction(1, x) for x in result)
        if s == 1:
            return result
    
    raise RuntimeError("no valid representation")
# EVOLVE-BLOCK-END
