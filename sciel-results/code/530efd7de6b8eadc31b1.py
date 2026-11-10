import random
import time
from fractions import Fraction
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Precompute reciprocals as Fractions for exact arithmetic
    recip = {i: Fraction(1, i) for i in range(1, 25)}
    
    def is_valid(S):
        """
        Returns True if S is reciprocal-subsum-free.
        For every x in S, 1/x is not the sum of reciprocals of any subset of S\{x}.
        Uses Meet-in-the-Middle for subset sum check.
        """
        S_list = list(S)
        n = len(S_list)
        if n < 2:
            return True
            
        for i, x in enumerate(S_list):
            target = recip[x]
            others = S_list[:i] + S_list[i+1:]
            if not others:
                continue
                
            # Check if target can be formed by subset sum of recip[others]
            # Meet in the middle
            m = len(others)
            if m > 20: # Fallback to simple check if too large? No, 24 is max.
                 pass
            
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            # Generate all subset sums for left
            left_sums = set()
            # Use iterative DP or bit manipulation?
            # Since m is small, just iterate 2^half
            # But we can prune? No, just generate.
            # Optimization: store as list, sort, then use two-pointer or set lookup.
            
            # Actually, for m=23, half=11. 2^11 = 2048. Very fast.
            left_sums = [0]
            for y in left:
                ry = recip[y]
                # Add ry to all existing sums
                new_sums = [s + ry for s in left_sums]
                left_sums.extend(new_sums)
                
            right_sums = [0]
            for y in right:
                ry = recip[y]
                new_sums = [s + ry for s in right_sums]
                right_sums.extend(new_sums)
                
            # Check if any l + r == target
            # Convert right_sums to a set for O(1) lookup
            right_set = set(right_sums)
            
            found = False
            for l in left_sums:
                if (target - l) in right_set:
                    # We found a subset that sums to 1/x
                    # But we must ensure the subset is not empty? 
                    # "sum of reciprocals of a subset of the OTHER selected elements"
                    # If the subset is empty, sum is 0. 1/x != 0. So empty subset is safe.
                    # However, if l=0 and r=target, the subset is from right only.
                    # If l=target and r=0, the subset is from left only.
                    # This is a valid violation.
                    found = True
                    break
                    
            if found:
                return False
                
        return True

    def find_violation(S):
        """
        Returns (x, subset) if violation found, else None.
        """
        S_list = list(S)
        for i, x in enumerate(S_list):
            target = recip[x]
            others = S_list[:i] + S_list[i+1:]
            m = len(others)
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            left_sums = [0]
            for y in left:
                ry = recip[y]
                new_sums = [s + ry for s in left_sums]
                left_sums.extend(new_sums)
                
            right_sums = [0]
            for y in right:
                ry = recip[y]
                new_sums = [s + ry for s in right_sums]
                right_sums.extend(new_sums)
                
            right_set = set(right_sums)
            
            # We need to reconstruct the subset if we want to remove specific elements.
            # For now, just knowing it's invalid is enough for the repair step.
            # But to repair, we should know which elements are involved.
            # Let's just return True/False for now and use a simpler repair: remove random element.
            for l in left_sums:
                if (target - l) in right_set:
                    return True
        return False

    best_set = set()
    best_size = 0
    
    # Strategy 1: Start with full set, repair
    for attempt in range(20): # Try a few times
        if time.time() > deadline:
            break
            
        current_set = set(range(1, 25))
        
        # Repair loop
        max_repair_steps = 100
        for _ in range(max_repair_steps):
            if time.time() > deadline:
                break
            
            if is_valid(current_set):
                break
            
            # Find a violation and remove an element involved
            # Simplified: Pick a random element from current_set and remove it?
            # Better: Identify x that has a violation, and remove a random element from the subset T that sums to 1/x.
            # Since is_valid doesn't return the subset, let's just remove a random element.
            # This might be slow.
            
            # Let's try a smarter repair:
            # Iterate through x. If violation exists for x, remove a random element from S\{x}.
            S_list = list(current_set)
            repaired = False
            for x in S_list:
                target = recip[x]
                others = [y for y in S_list if y != x]
                # Check if valid for this x
                m = len(others)
                half = m // 2
                left = others[:half]
                right = others[half:]
                
                left_sums = [0]
                left_indices = [] # Track indices? Too complex.
                
                # Just check existence
                left_sums_val = [0]
                for y in left:
                    ry = recip[y]
                    left_sums_val.extend([s + ry for s in left_sums_val])
                    
                right_sums_val = [0]
                for y in right:
                    ry = recip[y]
                    right_sums_val.extend([s + ry for s in right_sums_val])
                    
                right_set = set(right_sums_val)
                
                violation_found = False
                for l in left_sums_val:
                    if (target - l) in right_set:
                        violation_found = True
                        break
                        
                if violation_found:
                    # Remove a random element from others (or x)
                    # Removing x is safest to break this specific violation
                    # But removing an element from the subset T is also valid.
                    # Let's remove x to be safe and fast.
                    current_set.remove(x)
                    repaired = True
                    break
                    
            if not repaired:
                # Should not happen if is_valid is consistent
                break
                
        if is_valid(current_set):
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
                
        # Try to regrow: add elements back from [1..24] \ current_set
        candidates = list(set(range(1, 25)) - current_set)
        random.shuffle(candidates)
        for c in candidates:
            if time.time() > deadline:
                break
            new_set = current_set | {c}
            if is_valid(new_set):
                current_set = new_set
                # Update best if larger
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
                    
    # Fallback if no valid set found (should not happen, singleton is always valid)
    if not best_set:
        best_set = {24}
        
    return list(best_set)
