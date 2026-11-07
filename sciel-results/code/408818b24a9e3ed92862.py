"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill
    
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Compute LCM of all divisors
    lcm_val = reduce(_lcm, divisors)
    
    # For each divisor, precompute the set of multiples within one LCM period
    # Multiple of d_i in [0, lcm_val): d_i, 2*d_i, ..., (lcm_val/d_i)*d_i
    # We work modulo lcm_val, so values are in [0, lcm_val - 1]
    
    # For a window [L, L+W] (mod lcm_val), a divisor d can use value v
    # if v % d == 0 and L <= v <= L+W (considering wrap-around)
    
    # Approach: iterate W from 0 upward. For each W, slide L across the period.
    # For each (L, W), check if there's a valid assignment via backtracking.
    
    # Optimization: Only need to check L values that are "interesting" - 
    # i.e., where at least one divisor's multiple aligns with the window boundary.
    # But for simplicity, let's slide L by 1 and use a fast feasibility check.
    
    # Better: For a fixed W, the window [L, L+W] contains a multiple of d
    # iff there exists k such that L <= k*d <= L+W (mod lcm_val).
    # This is equivalent to: floor((L+W)/d) - ceil(L/d) + 1 >= 1 (with mod adjustment)
    
    # Let's use a different approach: 
    # For each W, iterate over all possible "anchor" positions.
    # The key observation: if a solution exists with span W, then for some L,
    # each divisor d_i has at least one multiple in [L, L+W].
    # Moreover, we need distinct multiples.
    
    # Efficient check for a given (L, W):
    # For each divisor i, compute the list of multiples of d_i in [L, L+W] (mod lcm_val)
    # Then check if there's a system of distinct representatives (SDR)
    # This is a bipartite matching problem: left side = divisors, right side = candidate values
    
    def get_multiples_in_window(d, L, W, lcm_val):
        """Get all multiples of d in [L, L+W] considering modular arithmetic."""
        result = []
        # First multiple >= L
        if L == 0:
            first = d  # smallest positive multiple
        else:
            k = (L + d - 1) // d  # ceil(L/d)
            first = k * d
        # But we need to handle the case where L=0: multiples are d, 2d, ...
        # Actually, multiples are ..., -d, 0, d, 2d, ... but we want positive
        # In mod lcm_val, 0 is a valid position (it's a multiple of every d)
        # Wait, the values must be positive. But in mod lcm_val, we can map
        # 0 to lcm_val. Let me reconsider.
        
        # Let's work with values in [1, lcm_val]. A multiple of d in [1, lcm_val]
        # is d, 2d, ..., (lcm_val/d)*d.
        # For window [L, L+W] in modular space, we need to handle wrap-around.
        
        # Simpler: just enumerate. Since d >= 6 and W will be small, 
        # the number of multiples in any window of size W is at most W/min(d) + 1
        # which is small.
        
        # For modular window [L, L+W] (where L is in [0, lcm_val-1]):
        # If L + W < lcm_val: window is [L, L+W] in linear space
        # Else: window wraps around: [L, lcm_val-1] U [0, (L+W) % lcm_val]
        
        # But values must be positive, so 0 maps to lcm_val.
        # Let's just work in [0, lcm_val-1] and remember 0 means lcm_val.
        
        # Actually, let me simplify: work entirely in modular arithmetic.
        # A value v (mod lcm_val) is a multiple of d iff v % d == 0.
        # The window [L, L+W] mod lcm_val contains values that are within W of L
        # in the circular sense.
        
        # For efficiency, since W will be small (hopefully < 100), let's just
        # check each offset from 0 to W.
        
        for offset in range(W + 1):
            v = (L + offset) % lcm_val
            if v % d == 0:
                result.append(v)
        return result
    
    # For bipartite matching, use a simple DFS-based augmenting path algorithm
    # Left nodes: divisors (0..n-1), Right nodes: candidate values
    # Edge from i to v if v is a multiple of divisors[i] in the window
    
    def has_valid_assignment(L, W, lcm_val, divisors, n):
        """Check if there's a valid assignment of distinct values to divisors."""
        # Build candidate lists for each divisor
        candidates = []
        for i in range(n):
            cands = get_multiples_in_window(divisors[i], L, W, lcm_val)
            candidates.append(cands)
            if not cands:
                return False  # This divisor has no multiple in the window
        
        # Bipartite matching: left = divisors, right = values
        # Use DFS augmenting path
        # Map values to indices for the right side
        all_values = set()
        for cands in candidates:
            all_values.update(cands)
        val_to_idx = {v: i for i, v in enumerate(sorted(all_values))}
        m = len(all_values)
        
        # match_to[i] = which left node is matched to right node i, or -1
        match_to = [-1] * m
        # match_from[j] = which right node is matched to left node j, or -1
        match_from = [-1] * n
        
        def try_kun(m, seen):
            for u in candidates[m]:
                v = val_to_idx[u]
                if seen[v]:
                    continue
                seen[v] = True
                if match_to[v] == -1 or try_kun(match_to[v], seen):
                    match_to[v] = m
                    match_from[m] = v
                    return True
            return False
        
        for i in range(n):
            seen = [False] * m
            if not try_kun(i, seen):
                return False
        return True
    
    # Now search for minimum W
    best_W = None
    best_L = None
    best_assignment = None
    
    # W must be at least n-1 = 9 since values must be distinct
    for W in range(9, 200):  # Try up to W=199
        if time.time() > deadline:
            break
        
        # Slide L across the period. But we don't need to check all L values.
        # Key insight: if a solution exists for some L, then shifting L by 1
        # changes which multiples are in the window. We can optimize by only
        # checking L values where the window boundary aligns with a multiple.
        
        # For now, let's check all L from 0 to lcm_val-1, but that's 3M+ iterations.
        # Too slow. Need to be smarter.
        
        # Alternative: for a fixed W, the condition "d has a multiple in [L, L+W]"
        # is periodic in L with period d. The combined condition for all divisors
        # is periodic with period lcm(d1, ..., dn) = lcm_val.
        
        # Within one period, the set of L values where all divisors have at least
        # one multiple in the window is a union of intervals. We can compute this.
        
        # For divisor d, the set of L values where [L, L+W] contains a multiple of d:
        # A multiple kd is in [L, L+W] iff L <= kd <= L+W iff kd-W <= L <= kd
        # So for each multiple kd, L can be in [kd-W, kd].
        # Union over all k: this is a periodic pattern.
        
        # The "good" L values for divisor d are those where L mod d is in 
        # [d-W mod d, d-1] U [0, W mod d] (if W >= d, then all L are good)
        
        # Actually, simpler: for a given L, d has a multiple in [L, L+W] iff
        # floor((L+W)/d) >= ceil(L/d). This is equivalent to:
        # (L+W) // d >= (L + d - 1) // d
        
        # Let me just check a subset of L values. Since the pattern is periodic
        # with period lcm_val, and we want to find ANY valid L, we can check
        # L values that are "aligned" with multiples.
        
        # Key optimization: only check L values where L is a multiple of some divisor,
        # or L+1 is, etc. Actually, the most efficient approach:
        
        # For each W, the feasible L values form a periodic set. We can find one
        # by checking L from 0 to some bound. Since the density of feasible L values
        # should be reasonable for small W, let's check L in a smart order.
        
        # Let's check L values that are multiples of the gcd of all divisors, or 
        # just check L from 0 to min(lcm_val, 10000) first, then expand if needed.
        
        checked = 0
        for L in range(lcm_val):
            if time.time() > deadline:
                break
            
            if has_valid_assignment(L, W, lcm_val, divisors, n):
                best_W = W
                best_L = L
                # Reconstruct the assignment
                # Re-run matching to get the actual values
                candidates = []
                for i in range(n):
                    cands = get_multiples_in_window(divisors[i], L, W, lcm_val)
                    candidates.append(cands)
                
                all_values = set()
                for cands in candidates:
                    all_values.update(cands)
                val_to_idx = {v: i for i, v in enumerate(sorted(all_values))}
                m = len(all_values)
                
                match_to = [-1] * m
                match_from = [-1] * n
                
                def try_kun2(m, seen):
                    for u in candidates[m]:
                        v = val_to_idx[u]
                        if seen[v]:
                            continue
                        seen[v] = True
                        if match_to[v] == -1 or try_kun2(match_to[v], seen):
                            match_to[v] = m
                            match_from[m] = v
                            return True
                    return False
                
                for i in range(n):
                    seen = [False] * m
                    try_kun2(i, seen)
                
                # Extract assignment: for each divisor i, value is the matched one
                result = []
                for i in range(n):
                    v_idx = match_from[i]
                    v = list(val_to_idx.keys())[v_idx]  # This is wrong, need reverse map
                    result.append(v)
                
                # Fix: create proper reverse mapping
                idx_to_val = {v_idx: v for v, v_idx in val_to_idx.items()}
                result = []
                for i in range(n):
                    v_idx = match_from[i]
                    v = idx_to_val[v_idx]
                    # Convert 0 to lcm_val (since values must be positive)
                    if v == 0:
                        v = lcm_val
                    result.append(v)
                
                best_assignment = result
                break
        
        if best_W is not None:
            break
    
    if best_assignment is not None:
        return [int(x) for x in best_assignment]
    
    # Fallback: return the divisors themselves (valid but not optimal)
    return [int(d) for d in divisors]

# EVOLVE-BLOCK-END
