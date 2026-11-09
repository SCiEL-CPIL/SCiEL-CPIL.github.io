"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations
from math import gcd
from functools import lru_cache

# EVOLVE-BLOCK-START

def _all_subset_sums_int(weights):
    """Generate all 2^n subset sums of a list of positive integers."""
    n = len(weights)
    sums = [0]
    for w in weights:
        sums = sums + [s + w for s in sums]
    return sums

def _is_dissociated_int(weights):
    """Check if all subset sums of positive integers are distinct."""
    sums = _all_subset_sums_int(weights)
    return len(sums) == len(set(sums))

def _min_pairwise_diff(weights):
    """Return the minimum absolute difference between any two distinct subset sums.
    Returns inf if all are distinct (no zero diff), else the min positive diff."""
    sums = _all_subset_sums_int(weights)
    s = sorted(sums)
    min_diff = float('inf')
    for i in range(1, len(s)):
        d = s[i] - s[i-1]
        if d == 0:
            return 0
        if d < min_diff:
            min_diff = d
    return min_diff

def _lcm_list(lst):
    """Compute LCM of a list of integers."""
    from math import gcd
    def _lcm(a, b):
        return a * b // gcd(a, b)
    result = 1
    for x in lst:
        result = _lcm(result, x)
    return result

def _evaluate_set(indices):
    """Given a set of indices into [1..22], compute weights = LCM/idx and check dissociation.
    Returns (is_valid, min_diff) where min_diff is the min pairwise subset-sum diff."""
    if not indices:
        return True, float('inf')
    L = _lcm_list(indices)
    weights = [L // i for i in indices]
    # Check dissociation
    sums = _all_subset_sums_int(weights)
    if len(sums) != len(set(sums)):
        return False, 0
    s = sorted(sums)
    min_diff = float('inf')
    for i in range(1, len(s)):
        d = s[i] - s[i-1]
        if d < min_diff:
            min_diff = d
    return True, min_diff

def construct():
    """Return a nonempty subset of [1,22] such that ALL reciprocal subset sums are distinct.
    Maximize cardinality."""
    deadline = time.time() + 80  # 80s deadline, 10s margin before 90s kill
    
    # Greedy initialization: try to build a large set
    # Start with a known good set
    current = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    current_valid, current_min_diff = _evaluate_set(current)
    best_set = list(current)
    best_min_diff = current_min_diff
    best_size = len(current)
    
    # Try to greedily add elements
    all_indices = list(range(1, 23))
    current_set = set(current)
    
    # Greedy: try adding each unused element if it maintains dissociation
    for idx in sorted(all_indices, key=lambda x: -x):  # try larger indices first (smaller weights)
        if idx in current_set:
            continue
        trial = current_set | {idx}
        valid, md = _evaluate_set(list(trial))
        if valid:
            current_set = trial
            current_min_diff = md
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
                best_min_diff = md
    
    # Now do simulated annealing to try to improve
    if time.time() < deadline - 10:
        # Simulated annealing: swap elements in and out
        current_set = set(best_set)
        current_valid, current_min_diff = _evaluate_set(list(current_set))
        
        # Temperature schedule
        T = 1000.0
        T_min = 1e-6
        cooling_rate = 0.995
        max_iter = 50000
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Choose a random move: add, remove, or swap
            move_type = random.random()
            
            if move_type < 0.3:
                # Try adding a random element
                candidates = [i for i in all_indices if i not in current_set]
                if not candidates:
                    continue
                idx = random.choice(candidates)
                trial = current_set | {idx}
                trial_valid, trial_md = _evaluate_set(list(trial))
                if trial_valid:
                    # Accept if improves min_diff or same size with better margin
                    if len(trial) > len(current_set):
                        current_set = trial
                        current_valid = True
                        current_min_diff = trial_md
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = list(current_set)
                            best_min_diff = trial_md
                    elif trial_md > current_min_diff:
                        current_set = trial
                        current_min_diff = trial_md
                        if len(current_set) > best_size or (len(current_set) == best_size and trial_md > best_min_diff):
                            best_set = list(current_set)
                            best_min_diff = trial_md
                            best_size = len(current_set)
                # If invalid, reject (or accept with probability based on T)
            elif move_type < 0.5:
                # Try removing a random element
                if len(current_set) <= 1:
                    continue
                idx = random.choice(list(current_set))
                trial = current_set - {idx}
                if not trial:
                    continue
                trial_valid, trial_md = _evaluate_set(list(trial))
                if trial_valid:
                    # Only accept if it leads to a better overall solution later
                    # For now, accept if min_diff improves significantly
                    if trial_md > current_min_diff * 1.5:
                        current_set = trial
                        current_min_diff = trial_md
            else:
                # Try swapping: remove one, add another
                if len(current_set) < 2:
                    continue
                idx_out = random.choice(list(current_set))
                candidates_in = [i for i in all_indices if i not in current_set]
                if not candidates_in:
                    continue
                idx_in = random.choice(candidates_in)
                trial = (current_set - {idx_out}) | {idx_in}
                trial_valid, trial_md = _evaluate_set(list(trial))
                if trial_valid:
                    if trial_md > current_min_diff:
                        current_set = trial
                        current_min_diff = trial_md
                        if len(current_set) > best_size or (len(current_set) == best_size and trial_md > best_min_diff):
                            best_set = list(current_set)
                            best_min_diff = trial_md
                            best_size = len(current_set)
                    else:
                        # Accept with probability exp(-(current_min_diff - trial_md)/T)
                        delta = current_min_diff - trial_md
                        if delta > 0 and T > T_min:
                            prob = min(1.0, delta / max(T, 1e-10))
                            if random.random() < prob:
                                current_set = trial
                                current_min_diff = trial_md
            
            T *= cooling_rate
    
    # Final polish: try all single swaps from the best set
    if time.time() < deadline - 2:
        for idx_in in all_indices:
            if idx_in in best_set:
                continue
            for idx_out in best_set:
                if len(best_set) <= 1:
                    break
                trial = set(best_set) - {idx_out} | {idx_in}
                valid, md = _evaluate_set(list(trial))
                if valid and len(trial) == len(best_set) and md > best_min_diff:
                    best_set = list(trial)
                    best_min_diff = md
    
    # Ensure nonempty
    if not best_set:
        best_set = [1]
    
    return sorted(best_set)

# EVOLVE-BLOCK-END
