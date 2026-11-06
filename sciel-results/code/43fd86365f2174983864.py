"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from fractions import Fraction
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s limit
    
    # Precompute LCM and integer values
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for i in range(1, 25):
        L = lcm(L, i)
        
    val = {i: L // i for i in range(1, 25)}
    
    def is_subsum_free(S):
        """Check if set S is subsum-free using integer arithmetic."""
        S_list = sorted(S)
        n = len(S_list)
        for i, x in enumerate(S_list):
            target = val[x]
            # Only elements y > x can form 1/x
            # In sorted S_list, these are S_list[i+1:]
            candidates = [val[y] for y in S_list[i+1:]]
            
            # Check if target is a subset sum of candidates
            # Prune sums > target
            if not candidates:
                continue
            if sum(candidates) < target:
                continue
                
            achievable = {0}
            for c in candidates:
                if c > target:
                    continue
                new_sums = set()
                for s in achievable:
                    ns = s + c
                    if ns <= target:
                        new_sums.add(ns)
                achievable |= new_sums
                if target in achievable:
                    return False
        return True

    def is_safe_in(S, x):
        """Check if x is safe to add to S (i.e., S U {x} is subsum-free)."""
        # 1. Check if x is subsumed by S
        target = val[x]
        candidates = [val[y] for y in S if y > x]
        if candidates:
            if sum(candidates) >= target:
                achievable = {0}
                found = False
                for c in candidates:
                    if c > target:
                        continue
                    new_sums = set()
                    for s in achievable:
                        ns = s + c
                        if ns <= target:
                            new_sums.add(ns)
                    achievable |= new_sums
                    if target in achievable:
                        return False
        
        # 2. Check if adding x makes any y < x subsumed
        # For y < x, x is a new candidate (since x > y)
        for y in S:
            if y >= x:
                continue
            target_y = val[y]
            # Candidates for y are {z in S U {x} | z > y}
            # This is {z in S | z > y} U {x}
            # We know S is currently subsum-free, so target_y is not a sum of {z in S | z > y}
            # We need to check if target_y is a sum of ({z in S | z > y} U {x})
            # This is equivalent to checking if target_y or target_y - val[x] is a sum of {z in S | z > y}
            
            # Optimization: if val[x] > target_y, x cannot be part of the sum (since all positive)
            if val[x] > target_y:
                continue
                
            candidates_y = [val[z] for z in S if z > y]
            # We need to check if target_y is in SubsetSums(candidates_y U {val[x]})
            # This is true if target_y is in SubsetSums(candidates_y) OR (target_y - val[x]) is in SubsetSums(candidates_y)
            
            # We know target_y is NOT in SubsetSums(candidates_y) because S is valid.
            # So we only need to check if (target_y - val[x]) is in SubsetSums(candidates_y)
            
            rem_target = target_y - val[x]
            if rem_target < 0:
                continue
            
            if rem_target == 0:
                # This means val[x] alone forms the sum? 
                # Wait, subset can be {x}. 
                # If rem_target == 0, it means {x} sums to target_y? 
                # Yes, if val[x] == target_y.
                # But x > y implies val[x] < val[y] = target_y. So rem_target > 0 always.
                pass

            # Check if rem_target is a subset sum of candidates_y
            if sum(candidates_y) < rem_target:
                continue
                
            achievable = {0}
            found = False
            for c in candidates_y:
                if c > rem_target:
                    continue
                new_sums = set()
                for s in achievable:
                    ns = s + c
                    if ns <= rem_target:
                        new_sums.add(ns)
                achievable |= new_sums
                if rem_target in achievable:
                    found = True
                    break
            if found:
                return False
                
        return True

    def greedy_remove(S):
        """Iteratively remove subsumed elements until stable."""
        S = set(S)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Sort to ensure deterministic order, smallest first
            for x in sorted(S):
                if time.time() >= deadline:
                    break
                # Check if x is subsumed by others in S
                # Only y > x matter
                others = [y for y in S if y > x]
                if not others:
                    continue
                    
                target = val[x]
                candidates = [val[y] for y in others]
                
                if sum(candidates) < target:
                    continue
                
                achievable = {0}
                found = False
                for c in candidates:
                    if c > target:
                        continue
                    new_sums = set()
                    for s in achievable:
                        ns = s + c
                        if ns <= target:
                            new_sums.add(ns)
                    achievable |= new_sums
                    if target in achievable:
                        found = True
                        break
                
                if found:
                    S.remove(x)
                    changed = True
        return S

    best = set()
    
    # Strategy 1: Greedy remove from full set
    S0 = set(range(1, 25))
    S1 = greedy_remove(S0)
    if len(S1) > len(best):
        best = S1
        
    # Strategy 2: Randomized restarts
    # Remove a random subset first, then greedy remove, then try to add back
    for _ in range(1000):
        if time.time() >= deadline:
            break
        
        # Random perturbation
        S_pert = set(range(1, 25))
        # Randomly remove 0 to 5 elements
        num_remove = random.randint(0, 5)
        to_remove = random.sample(list(S_pert), num_remove)
        S_pert -= set(to_remove)
        
        S_curr = greedy_remove(S_pert)
        
        # Try to add back all elements not in S_curr
        remaining = set(range(1, 25)) - S_curr
        added = True
        while added:
            added = False
            # Try adding in random order
            for x in random.sample(list(remaining), len(remaining)):
                if time.time() >= deadline:
                    break
                if is_safe_in(S_curr, x):
                    S_curr.add(x)
                    remaining.remove(x)
                    added = True
                    
        if len(S_curr) > len(best):
            best = S_curr
            
    # Final verification
    if not is_subsum_free(best):
        # Fallback to a known valid set if something went wrong
        # The set of all numbers > 12 is likely valid?
        # Let's just return a small valid set
        best = {24}
        
    return list(best)
# EVOLVE-BLOCK-END
