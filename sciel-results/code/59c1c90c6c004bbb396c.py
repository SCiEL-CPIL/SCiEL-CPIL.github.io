"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction
from itertools import combinations

def _time_remaining(deadline):
    return deadline - time.time()

def _lcm(a, b):
    return a * b // gcd(a, b)

def _can_form_target(target_num, target_den, recips_ints, recips_dens):
    """Check if target_num/target_den can be formed as sum of a subset of recips.
    recips_ints[i]/recips_dens[i] are the reciprocals.
    Uses integer scaling via LCM of all denominators.
    """
    n = len(recips_ints)
    if n == 0:
        return target_num == 0
    
    # Scale all fractions to common denominator
    common_den = recips_dens[0]
    for d in recips_dens[1:]:
        common_den = _lcm(common_den, d)
    
    # Target in scaled form
    target_scaled = target_num * (common_den // target_den)
    
    # Each reciprocal in scaled form
    scaled_vals = [recips_ints[i] * (common_den // recips_dens[i]) for i in range(n)]
    
    # Check if target_scaled is a subset sum of scaled_vals
    # Use bitset DP for efficiency when values are small
    # But values can be large, so use a set-based approach with pruning
    
    # For small n, just enumerate subsets
    if n <= 20:
        # Use meet-in-the-middle for n up to ~40
        if n <= 20:
            # Simple subset sum check
            sums = {0}
            for v in scaled_vals:
                new_sums = set()
                for s in sums:
                    new_sums.add(s + v)
                sums |= new_sums
                if target_scaled in sums:
                    return True
            return target_scaled in sums
        else:
            # Meet in the middle
            half = n // 2
            left = scaled_vals[:half]
            right = scaled_vals[half:]
            
            left_sums = {0}
            for v in left:
                new_sums = set()
                for s in left_sums:
                    new_sums.add(s + v)
                left_sums |= new_sums
            
            right_sums = {0}
            for v in right:
                new_sums = set()
                for s in right_sums:
                    new_sums.add(s + v)
                right_sums |= new_sums
            
            for ls in left_sums:
                if (target_scaled - ls) in right_sums:
                    return True
            return False
    else:
        # For larger n, use greedy approximation or just check small subsets
        # Check if any single element equals target
        for v in scaled_vals:
            if v == target_scaled:
                return True
        # Check pairs
        for i in range(n):
            for j in range(i+1, n):
                if scaled_vals[i] + scaled_vals[j] == target_scaled:
                    return True
        # Check triples for small cases
        if n <= 30:
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if scaled_vals[i] + scaled_vals[j] + scaled_vals[k] == target_scaled:
                            return True
        return False

def _is_subsum_free(candidate_set):
    """Check if candidate_set is reciprocal-subsum-free."""
    n = len(candidate_set)
    if n <= 1:
        return True
    
    # For each element, check if its reciprocal equals sum of subset of others' reciprocals
    for i in range(n):
        target = Fraction(1, candidate_set[i])
        others = [candidate_set[j] for j in range(n) if j != i]
        
        # Reciprocals of others
        recips_ints = [1 for _ in others]
        recips_dens = [others[j] for j in range(len(others))]
        
        if _can_form_target(target.numerator, target.denominator, recips_ints, recips_dens):
            return False
    return True

def construct():
    deadline = time.time() + 85  # 5s safety margin
    
    # Strategy: Greedy construction from largest to smallest
    # Start with largest elements (safest - least likely to cause subsums)
    # Then try to add smaller elements if they don't break the property
    
    best_set = list(range(24, 0, -1))[:1]  # Start with {24}
    
    # Try to build up greedily
    current_set = [24]
    
    # Try adding elements from 23 down to 1
    for elem in range(23, 0, -1):
        if _time_remaining(deadline) < 2:
            break
        
        trial_set = current_set + [elem]
        if _is_subsum_free(trial_set):
            current_set = trial_set
    
    if len(current_set) > len(best_set):
        best_set = current_set
    
    # Try a different order: start with medium elements
    if _time_remaining(deadline) > 30:
        current_set2 = []
        for elem in range(24, 0, -1):
            if _time_remaining(deadline) < 2:
                break
            trial_set = current_set2 + [elem]
            if _is_subsum_free(trial_set):
                current_set2 = trial_set
        if len(current_set2) > len(best_set):
            best_set = current_set2
    
    # Try starting with specific patterns
    if _time_remaining(deadline) > 20:
        # Try powers of 2 and nearby
        for start_elem in [24, 16, 12, 8]:
            if _time_remaining(deadline) < 5:
                break
            current_set3 = [start_elem]
            for elem in range(start_elem - 1, 0, -1):
                if _time_remaining(deadline) < 2:
                    break
                trial_set = current_set3 + [elem]
                if _is_subsum_free(trial_set):
                    current_set3 = trial_set
            if len(current_set3) > len(best_set):
                best_set = current_set3
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
