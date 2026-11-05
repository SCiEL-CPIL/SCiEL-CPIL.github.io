import time
import math
import random
from math import lcm
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 55.0
    
    # Universe
    universe = list(range(1, 23))
    n_universe = len(universe)
    
    # Precompute which pairs are "conflicting" - i.e., if a and b are in the set,
    # they might create a linear dependency with some other element.
    # Key insight: if 1/a = 1/b + 1/c for distinct b,c, then {a,b,c} cannot all be in the set.
    # More generally, we need to check for any linear dependency among reciprocals.
    
    # Build a conflict graph: edge between a and b if {a,b} is "bad" in some sense
    # Actually, the real constraint is more complex. Let's use a different approach.
    
    # For each subset S, we need all 2^|S| subset sums of {1/x : x in S} to be distinct.
    # This is equivalent to: no non-empty subset of S has sum 0 in the group of rationals,
    # which means no linear relation with coefficients in {-1,0,1} sums to 0.
    
    # Efficient check: use the fact that if we sort the elements, we can use meet-in-the-middle
    # for small sets, and probabilistic checks for larger sets.
    
    def get_scale(S):
        """Get the LCM of all elements in S."""
        if not S:
            return 1
        s = 1
        for x in S:
            s = lcm(s, x)
        return s
    
    def is_dissociated_fast(S):
        """Check if all subset sums of reciprocals of S are distinct using integer arithmetic."""
        if not S:
            return True
        if len(S) > 22:
            return False
        
        scale = get_scale(S)
        weights = [scale // x for x in S]
        n = len(S)
        
        if n <= 16:
            # Direct check with bit manipulation
            sums = [0] * (1 << n)
            for i in range(1, 1 << n):
                # Use lowest set bit to compute
                lb = i & (-i)
                j = lb.bit_length() - 1
                sums[i] = sums[i ^ lb] + weights[j]
            return len(set(sums)) == (1 << n)
        elif n <= 20:
            # Meet in the middle
            half = n // 2
            left_weights = weights[:half]
            right_weights = weights[half:]
            
            left_sums = set()
            for mask in range(1 << half):
                s = 0
                for i in range(half):
                    if mask & (1 << i):
                        s += left_weights[i]
                left_sums.add(s)
            
            right_sums = set()
            for mask in range(1 << (n - half)):
                s = 0
                for i in range(n - half):
                    if mask & (1 << i):
                        s += right_weights[i]
                right_sums.add(s)
            
            # Check if any left sum + right sum collision exists
            # This is equivalent to checking if left_sums and -right_sums intersect
            # But we need to be careful: we're checking if all sums are distinct
            # Actually, the total number of distinct sums should be 2^n
            # Let's just generate all sums and check
            all_sums = set()
            for ls in left_sums:
                for rs in right_sums:
                    all_sums.add(ls + rs)
            return len(all_sums) == (1 << n)
        else:
            # Probabilistic check for large sets
            return _probabilistic_check(weights)
    
    def _probabilistic_check(weights):
        """For large sets, use a faster heuristic check."""
        if len(set(weights)) != len(weights):
            return False
        
        n = len(weights)
        random.seed(12345)
        sample_size = min(2**n, 5000)
        sampled_sums = set()
        for _ in range(sample_size):
            subset_sum = 0
            for i in range(n):
                if random.random() < 0.5:
                    subset_sum += weights[i]
            if subset_sum in sampled_sums:
                return False
            sampled_sums.add(subset_sum)
        return True
    
    def try_add(S, x):
        """Check if S union {x} is dissociated."""
        new_S = list(S)
        new_S.append(x)
        return is_dissociated_fast(new_S)
    
    def greedy_build(order):
        """Build a dissociated set by greedily adding elements in the given order."""
        current = []
        for x in order:
            if try_add(current, x):
                current.append(x)
        return current
    
    def expand_set(S):
        """Try to expand a dissociated set by adding any remaining element."""
        remaining = [x for x in universe if x not in S]
        added = True
        while added:
            added = False
            for x in remaining[:]:
                if try_add(S, x):
                    S.append(x)
                    remaining.remove(x)
                    added = True
                    if time.time() - start_time > timeout:
                        break
            if time.time() - start_time > timeout:
                break
        return S
    
    # Strategy 1: Greedy with descending order (largest elements first)
    candidates_desc = sorted(universe, reverse=True)
    greedy_set = greedy_build(candidates_desc)
    greedy_set = expand_set(greedy_set)
    best_set = list(greedy_set)
    best_size = len(greedy_set)
    
    # Strategy 2: Greedy with ascending order
    candidates_asc = sorted(universe)
    greedy_set_asc = greedy_build(candidates_asc)
    greedy_set_asc = expand_set(greedy_set_asc)
    if len(greedy_set_asc) > best_size:
        best_set = list(greedy_set_asc)
        best_size = len(greedy_set_asc)
    
    # Strategy 3: Backtracking search for maximum cardinality
    def backtrack_search(target_size, current_set, start_idx, candidates):
        """Recursively try to build a set of target_size."""
        if time.time() - start_time > timeout:
            return None
        if len(current_set) == target_size:
            return list(current_set)
        
        remaining_needed = target_size - len(current_set)
        available = candidates[start_idx:]
        if len(available) < remaining_needed:
            return None
        
        for i, x in enumerate(available):
            if try_add(current_set, x):
                current_set.append(x)
                result = backtrack_search(target_size, current_set, start_idx + i + 1, candidates)
                if result is not None:
                    return result
                current_set.pop()
        
        return None
    
    # Try to find sets of increasing size
    for target_size in range(best_size + 1, 15):
        if time.time() - start_time > timeout:
            break
        
        ordered_candidates = sorted(universe, reverse=True)
        result = backtrack_search(target_size, [], 0, ordered_candidates)
        if result is not None:
            best_set = result
            best_size = target_size
            continue
    
    # Strategy 4: Randomized restarts with various heuristics
    random.seed(42)
    restarts = 150
    for r in range(restarts):
        if time.time() - start_time > timeout:
            break
        
        if r % 4 == 0:
            order = sorted(universe, reverse=True)
        elif r % 4 == 1:
            order = sorted(universe)
        elif r % 4 == 2:
            order = universe[:]
            random.shuffle(order)
        else:
            order = sorted(universe, key=lambda x: abs(x - 11.5) + random.random() * 5)
        
        current_set = greedy_build(order)
        current_set = expand_set(current_set)
        
        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(current_set)
    
    # Strategy 5: Local search - try swapping elements to improve
    if time.time() - start_time < timeout - 10:
        for _ in range(30):
            if time.time() - start_time > timeout:
                break
            
            current = list(best_set)
            improved = True
            while improved:
                improved = False
                for i in range(len(current)):
                    for x in universe:
                        if x in current:
                            continue
                        test_set = [current[j] for j in range(len(current)) if j != i]
                        test_set.append(x)
                        if is_dissociated_fast(test_set):
                            expanded = expand_set(test_set)
                            if len(expanded) > len(current):
                                current = expanded
                                improved = True
                                break
                    if improved:
                        break
            
            if len(current) > best_size:
                best_set = current
                best_size = len(current)
    
    return best_set
