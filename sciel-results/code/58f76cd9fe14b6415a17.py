from fractions import Fraction
import itertools
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin

    # 1. Precompute all reciprocal identities
    # Identity: 1/x = sum_{y in T} 1/y, where x not in T, x,y in 1..24
    # We represent T as a frozenset or tuple of sorted elements.
    # We store (x, T)
    
    identities = []
    # We need to check all subsets T of {1..24} \ {x}
    # But iterating all subsets of 24 is 2^24 ~ 16M.
    # For each subset, compute sum of reciprocals.
    # Check if it equals 1/x for some x not in T.
    # This is feasible but might be slow in Python if not optimized.
    # 16M subsets * 24 elements avg size? No, average size is 12.
    # 16M * 12 additions = 200M operations. Might be too slow for 90s in pure Python.
    
    # Alternative: Generate identities by iterating x and finding representations.
    # 1/x = 1/a + 1/b + ...
    # This is the Egyptian fraction problem.
    # For x=1: 1 = 1/a + ...
    # For x=2: 1/2 = 1/a + ...
    
    # Let's try the subset sum approach but with pruning or just checking if it's fast enough.
    # Actually, we can iterate over all subsets of {1..24}.
    # There are 2^24 = 16,777,216 subsets.
    # For each subset, we compute the sum of reciprocals (Fraction).
    # Then we check if that sum is of the form 1/x for x in 1..24 and x not in subset.
    
    # Let's optimize the subset iteration.
    # We can use a dictionary to map Fraction -> list of subsets? No, too many subsets.
    # We just need to collect the (x, T) pairs.
    
    # Let's iterate k from 1 to 24 (size of T).
    # For each combination of k elements from 1..24:
    #   Compute sum S.
    #   Check if S == 1/x for some x not in T.
    
    # Number of combinations:
    # k=1: 24
    # k=2: 276
    # k=3: 2024
    # k=4: 10626
    # k=5: 42504
    # k=6: 134596
    # k=7: 346104
    # k=8: 735471
    # k=9: 1307504
    # k=10: 2003001
    # ...
    # Sum is 2^24 - 1.
    
    # This loop in Python might take a while.
    # Let's try to do it efficiently.
    
    # We can precompute reciprocals as Fractions.
    recips = {i: Fraction(1, i) for i in range(1, 25)}
    
    # To speed up, we can use a recursive generation or just itertools.combinations.
    # itertools.combinations is implemented in C, so it's fast.
    # The bottleneck is the Fraction addition.
    
    # Let's try to limit the search or use a faster check.
    # Actually, we only need to find the identities to build the ILP.
    # If the number of identities is small, the ILP is small.
    
    # Let's run the identity search.
    t0 = time.time()
    identities_set = set()
    
    # We iterate over all non-empty subsets of {1..24}
    # But we can prune: if sum > 1, and we are looking for 1/x where x>=1, 
    # then 1/x <= 1. So if sum > 1, it can't be 1/x for x>=1.
    # Wait, 1/1 = 1. 1/2 = 0.5.
    # So we only care about sums in [1/24, 1].
    # Actually, min sum is 1/24. Max sum is 1 (if 1 is in subset).
    # If 1 is in subset, sum >= 1. If sum == 1, then x=1. But 1 is in T, so x cannot be 1.
    # So if 1 in T, no identity for x=1.
    # If sum > 1, no identity.
    
    # Let's just iterate and check.
    
    elements = list(range(1, 25))
    
    # To make it faster, we can skip subsets that are too large or too small?
    # No, let's just run it.
    
    # Optimization: 
    # We can group by sum? No.
    
    # Let's try a different approach:
    # For each x in 1..24, find all subsets T of {1..24}\{x} such that sum(T) = 1/x.
    # This is a subset sum problem with rational numbers.
    # We can use DP? The sums are fractions with large denominators.
    # DP on fractions is hard.
    
    # Let's stick to iterating combinations.
    # To speed up Python, we can use numpy? No, Fractions are not numpy-friendly.
    
    # Let's try to see how long it takes.
    # If it's too slow, we can limit to small k.
    # Most identities likely have small k.
    # 1/x = 1/a + 1/b ...
    # For x=1: 1 = 1/2 + 1/3 + 1/6 (k=3)
    # For x=2: 1/2 = 1/3 + 1/6 (k=2), 1/4 + 1/4 (no, distinct), 1/4+1/5+1/20...
    
    # Let's iterate k from 1 to 10 first.
    # If we find enough, we can stop or continue.
    
    # Actually, let's just do the full search but with a time check.
    
    found_ids = []
    
    # Precompute reciprocals
    # Use a list for speed
    rec = [None] + [Fraction(1, i) for i in range(1, 25)]
    
    # Iterate k
    for k in range(1, 12): # Up to 11? 
        if time.time() > deadline - 10:
            break
        for comb in itertools.combinations(elements, k):
            s = Fraction(0)
            for y in comb:
                s += rec[y]
                if s > 1: # Prune
                    break
            if s > 1:
                continue
            
            # Check if s == 1/x for some x not in comb
            # s = 1/x => x = 1/s
            # x must be integer
            if s.numerator == 1:
                x = s.denominator
                if 1 <= x <= 24 and x not in comb:
                    # Identity found: x and comb
                    # Store (x, frozenset(comb))
                    key = (x, frozenset(comb))
                    if key not in identities_set:
                        identities_set.add(key)
                        found_ids.append(key)
    
    # Convert to list of (x, tuple(T)) for ILP
    # We need to map elements to indices 0..23
    # x is 1..24, index x-1
    # y in T is 1..24, index y-1
    
    # Build ILP
    n_vars = 24
    c = np.ones(n_vars) # Maximize sum z_i
    
    # Constraints
    # z_x + sum_{y in T} z_y <= |T|
    # Indices:
    # x_idx = x - 1
    # y_idx = y - 1
    
    A_rows = []
    b_vals = []
    
    for x, T in identities_set:
        row = np.zeros(n_vars)
        row[x-1] = 1
        for y in T:
            row[y-1] = 1
        A_rows.append(row)
        b_vals.append(len(T))
        
    if A_rows:
        A = np.array(A_rows)
        b = np.array(b_vals)
        
        # Bounds: 0 <= z_i <= 1
        bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))
        
        # Linear constraint: A z <= b
        # milp requires LinearConstraint
        # lb is -inf, ub is b
        lin_cons = LinearConstraint(A, lb=-np.inf, ub=b)
        
        # Integrality
        integrality = np.ones(n_vars)
        
        try:
            res = milp(c=c, constraints=lin_cons, integrality=integrality, bounds=bounds)
            if res.success:
                # Extract solution
                z = res.x
                # z should be close to 0 or 1
                z_rounded = np.round(z).astype(int)
                
                # Verify solution
                S = [i+1 for i in range(n_vars) if z_rounded[i] == 1]
                
                # Double check validity with exact fractions
                if _is_valid_exact(S):
                    return S
                else:
                    # Fallback to greedy if ILP solution is invalid (shouldn't happen)
                    pass
            
        except Exception as e:
            pass
    
    # Fallback to greedy if ILP failed or timed out
    return _greedy_construct()

def _is_valid_exact(S):
    S_set = set(S)
    for x in S:
        target = Fraction(1, x)
        others = [y for y in S if y != x]
        # Check all subsets of others
        # This is expensive, but S is small (<=24)
        # We can use the precomputed identities?
        # Or just iterate subsets of others.
        # |others| <= 23. 2^23 is 8M. Too slow for repeated checks.
        # But we only check once.
        # Better: use the identities we found?
        # We didn't return them. Let's just do a quick check.
        # Actually, if we found the identities, we can check if any identity (x, T) is satisfied.
        # i.e., is T subset of S_set?
        # But we didn't store identities globally.
        
        # Let's just do the subset sum check for this specific S.
        # Since S is likely ~20 elements, 2^19 = 524k subsets.
        # For each x, we check subsets of S\{x}.
        # 20 * 2^19 * 20 ops = 200M ops. Might be slow in Python.
        
        # Let's use a set of reachable sums.
        sums = {Fraction(0)}
        for y in others:
            new_sums = set()
            for s in sums:
                ns = s + Fraction(1, y)
                if ns == target:
                    return False
                if ns < target:
                    new_sums.add(ns)
            sums.update(new_sums)
            if min(sums) > target:
                break
    return True

def _greedy_construct():
    S = []
    for x in range(24, 0, -1):
        candidate = S + [x]
        if _is_valid_exact(candidate):
            S.append(x)
    return S
