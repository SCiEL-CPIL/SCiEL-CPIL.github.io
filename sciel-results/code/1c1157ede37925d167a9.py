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
        Uses Meet-in-the-Middle for subset sum check with early pruning.
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
                
            # Quick check: if target > sum of all reciprocals of others, skip
            total_recip = sum(recip[y] for y in others)
            if target > total_recip:
                continue
            # If target < min reciprocal, it can't be formed (all reciprocals are positive)
            min_recip = min(recip[y] for y in others)
            if target < min_recip:
                continue
                
            m = len(others)
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            left_sums = [Fraction(0)]
            for y in left:
                ry = recip[y]
                new_sums = [s + ry for s in left_sums]
                left_sums.extend(new_sums)
                
            right_sums = [Fraction(0)]
            for y in right:
                ry = recip[y]
                new_sums = [s + ry for s in right_sums]
                right_sums.extend(new_sums)
                
            right_set = set(right_sums)
            
            found = False
            for l in left_sums:
                if (target - l) in right_set:
                    found = True
                    break
                    
            if found:
                return False
                
        return True

    def find_violation_with_subset(S):
        """
        Returns (x, offending_subset) if violation found, else None.
        offending_subset is the list of elements y in S\{x} such that sum(1/y for y in offending_subset) == 1/x.
        """
        S_list = list(S)
        for i, x in enumerate(S_list):
            target = recip[x]
            others = S_list[:i] + S_list[i+1:]
            if not others:
                continue
            m = len(others)
            half = m // 2
            left = others[:half]
            right = others[half:]
            
            # Build left sums with tracking of which elements are included
            # left_sums[i] = (sum_value, bitmask) where bitmask indicates which left elements are used
            left_sums = [(Fraction(0), 0)]
            for j, y in enumerate(left):
                ry = recip[y]
                bit = 1 << j
                new_sums = [(s + ry, mask | bit) for (s, mask) in left_sums]
                left_sums.extend(new_sums)
                
            right_sums = [(Fraction(0), 0)]
            for j, y in enumerate(right):
                ry = recip[y]
                bit = 1 << j
                new_sums = [(s + ry, mask | bit) for (s, mask) in right_sums]
                right_sums.extend(new_sums)
                
            right_set = {}
            for s, mask in right_sums:
                right_set[s] = mask
            
            for l, lmask in left_sums:
                rem = target - l
                if rem in right_set:
                    rmask = right_set[rem]
                    # Reconstruct the subset
                    subset = []
                    for j, y in enumerate(left):
                        if lmask & (1 << j):
                            subset.append(y)
                    for j, y in enumerate(right):
                        if rmask & (1 << j):
                            subset.append(y)
                    return (x, subset)
        return None

    def greedy_construct(order):
        """
        Build a valid set by greedily adding elements in the given order.
        """
        current_set = set()
        for c in order:
            if time.time() > deadline:
                break
            new_set = current_set | {c}
            if is_valid(new_set):
                current_set = new_set
        return current_set

    def repair_and_regrow(initial_set):
        """
        Start with initial_set, repair until valid, then try to regrow.
        """
        current_set = set(initial_set)
        
        # Repair loop
        max_repair_steps = 200
        for _ in range(max_repair_steps):
            if time.time() > deadline:
                break
            
            if is_valid(current_set):
                break
            
            viol = find_violation_with_subset(current_set)
            if viol is None:
                break
            
            x, offending_subset = viol
            # Try multiple removal strategies and pick the best
            candidates_to_remove = set()
            if offending_subset:
                candidates_to_remove.update(offending_subset)
            candidates_to_remove.add(x)
            
            best_size_after_removal = -1
            best_removal = None
            for rem in candidates_to_remove:
                test_set = current_set - {rem}
                # Count how many more violations would remain (approximate by checking validity)
                if is_valid(test_set):
                    # This removal fixes it completely
                    best_size_after_removal = len(test_set)
                    best_removal = rem
                    break
                else:
                    # Count violations roughly: just track size for now
                    if len(test_set) > best_size_after_removal:
                        best_size_after_removal = len(test_set)
                        best_removal = rem
            
            if best_removal is not None:
                current_set.remove(best_removal)
            else:
                # Fallback: remove largest
                current_set.remove(max(current_set))
                
        if is_valid(current_set):
            # Try to regrow with multiple passes
            for _pass in range(3):
                if time.time() > deadline:
                    break
                candidates = list(set(range(1, 25)) - current_set)
                random.shuffle(candidates)
                added_any = False
                for c in candidates:
                    if time.time() > deadline:
                        break
                    new_set = current_set | {c}
                    if is_valid(new_set):
                        current_set = new_set
                        added_any = True
                if not added_any:
                    break
                    
        return current_set

    best_set = set()
    best_size = 0
    
    # Strategy 1: Start with full set, repair with smarter removal
    if time.time() < deadline:
        result = repair_and_regrow(set(range(1, 25)))
        if len(result) > best_size:
            best_size = len(result)
            best_set = set(result)
    
    # Strategy 2: Greedy construction with different orderings
    orderings = [
        list(range(24, 0, -1)),  # Descending: 24, 23, ..., 1
        list(range(1, 25)),      # Ascending: 1, 2, ..., 24
    ]
    
    # Add a few random orderings
    for _ in range(5):
        if time.time() > deadline:
            break
        rand_order = list(range(1, 25))
        random.shuffle(rand_order)
        orderings.append(rand_order)
    
    for order in orderings:
        if time.time() > deadline:
            break
        result = greedy_construct(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = set(result)
        # Also try to regrow from this greedy result
        if time.time() < deadline:
            candidates = list(set(range(1, 25)) - result)
            random.shuffle(candidates)
            for c in candidates:
                if time.time() > deadline:
                    break
                new_set = result | {c}
                if is_valid(new_set):
                    result = new_set
            if len(result) > best_size:
                best_size = len(result)
                best_set = set(result)
    
    # Strategy 3: More repair attempts with random perturbations
    for attempt in range(10):
        if time.time() > deadline:
            break
        
        # Start with a random subset of size ~12-16
        size = random.randint(12, 16)
        initial = random.sample(range(1, 25), size)
        result = repair_and_regrow(set(initial))
        if len(result) > best_size:
            best_size = len(result)
            best_set = set(result)
    
    # Strategy 4: Local search with swaps on the best set found
    if best_set and time.time() < deadline:
        current_best = set(best_set)
        # Try swap operations: remove one, add one
        for _swap_pass in range(3):
            if time.time() > deadline:
                break
            improved = False
            removed_elements = list(current_best)
            random.shuffle(removed_elements)
            for rem in removed_elements:
                if time.time() > deadline:
                    break
                base_set = current_best - {rem}
                # Try adding each candidate not in base_set
                candidates = [c for c in range(1, 25) if c not in base_set]
                random.shuffle(candidates)
                for add in candidates:
                    if time.time() > deadline:
                        break
                    new_set = base_set | {add}
                    if is_valid(new_set) and len(new_set) > len(current_best):
                        current_best = new_set
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        if len(current_best) > best_size:
            best_size = len(current_best)
            best_set = set(current_best)
    
    # Strategy 5: Simulated annealing-like perturbation and repair
    if time.time() < deadline:
        for _sa_iter in range(15):
            if time.time() > deadline:
                break
            # Perturb: randomly add/remove 1-2 elements
            perturbed = set(best_set)
            # Remove 1-2 random elements
            to_remove = random.sample(list(perturbed), min(random.randint(1, 2), len(perturbed)))
            for r in to_remove:
                perturbed.discard(r)
            # Add 1-2 random elements
            available = [c for c in range(1, 25) if c not in perturbed]
            if available:
                to_add = random.sample(available, min(random.randint(1, 2), len(available)))
                perturbed.update(to_add)
            
            result = repair_and_regrow(perturbed)
            if len(result) > best_size:
                best_size = len(result)
                best_set = set(result)
    
    # Fallback if no valid set found (should not happen, singleton is always valid)
    if not best_set:
        best_set = {24}
        
    return list(best_set)
