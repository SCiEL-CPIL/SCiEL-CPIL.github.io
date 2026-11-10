import time
from math import gcd
from functools import reduce
import heapq

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    LCM = reduce(lcm, divisors)  # 30030
    
    start_time = time.time()
    timeout = 85  # Safety margin for 90s limit
    
    # Precompute interesting L values once
    interesting_L = set()
    for d in divisors:
        for k in range(1, LCM // d + 1):
            interesting_L.add(k * d)
            if k * d - 1 >= 1:
                interesting_L.add(k * d - 1)
    interesting_L.add(1)
    interesting_L.add(LCM)
    interesting_L = sorted(interesting_L)
    
    # We search for minimal W (max - min)
    # Minimum W is 9 (10 distinct integers)
    # We will iterate W from 9 upwards
    for W in range(9, 300):
        if time.time() - start_time > timeout:
            # Fallback: return a valid solution
            return divisors[:]
        
        for L in interesting_L:
            if time.time() - start_time > timeout:
                return divisors[:]
            
            # For this window [L, L+W], determine candidates for each divisor
            all_candidates = []
            valid_window = True
            
            for i, d in enumerate(divisors):
                # Find first multiple >= L
                k = (L + d - 1) // d
                c1 = k * d
                
                if c1 > L + W:
                    valid_window = False
                    break
                
                # Collect all multiples of d in [L, L+W]
                candidates = []
                x = c1
                while x <= L + W:
                    candidates.append(x)
                    x += d
                
                if not candidates:
                    valid_window = False
                    break
                    
                all_candidates.append(candidates)
            
            if not valid_window:
                continue
            
            # Check if any value is forced (only one candidate)
            forced = [None] * n
            flexible = []
            for i in range(n):
                if len(all_candidates[i]) == 1:
                    forced[i] = all_candidates[i][0]
                else:
                    flexible.append(i)
            
            # Check for conflicts in forced assignments
            used_forced = set()
            has_conflict = False
            for i in range(n):
                if forced[i] is not None:
                    if forced[i] in used_forced:
                        has_conflict = True
                        break
                    used_forced.add(forced[i])
            
            if has_conflict:
                continue
            
            # If no flexible divisors, we're done
            if not flexible:
                result = forced[:]
                if len(set(result)) == n:
                    return result
                continue
            
            # Use bipartite matching (Hopcroft-Karp style) for the flexible part
            # Left side: flexible divisors (indices)
            # Right side: values in the window
            # We need to find a matching where each divisor gets a distinct multiple
            
            # Build adjacency: for each flexible divisor, which values can it take?
            # Values are in [L, L+W]
            val_to_idx = {}
            for i in flexible:
                for v in all_candidates[i]:
                    if v not in used_forced:
                        if v not in val_to_idx:
                            val_to_idx[v] = []
                        val_to_idx[v].append(i)
            
            # Use DFS-based bipartite matching
            match_val = {}  # value -> divisor index
            match_div = {}  # divisor index -> value
            
            def dfs(div_idx, visited):
                for v in all_candidates[div_idx]:
                    if v in used_forced or v in visited:
                        continue
                    visited.add(v)
                    if v not in match_val or dfs(match_val[v], visited):
                        match_val[v] = div_idx
                        match_div[div_idx] = v
                        return True
                return False
            
            found = True
            for div_idx in flexible:
                visited = set()
                if not dfs(div_idx, visited):
                    found = False
                    break
            
            if found:
                result = forced[:]
                for div_idx in flexible:
                    result[div_idx] = match_div[div_idx]
                
                # Verify
                if len(set(result)) == n:
                    for i in range(n):
                        if result[i] % divisors[i] != 0 or not (0 < result[i] <= 1000000):
                            break
                    else:
                        return result
    
    # Fallback if no solution found in time or range
    return divisors[:]
