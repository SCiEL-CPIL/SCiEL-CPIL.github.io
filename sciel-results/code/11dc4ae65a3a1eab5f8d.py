import time
import random
from itertools import combinations

def construct():
    """
    Finds a set of distinct integers in 1..256 maximizing the size,
    such that all non-empty subset sums are distinct and no two distinct
    subset sums divide one another.
    """
    start_time = time.time()
    time_limit = 56  # Keep 4s margin for safety
    
    best_set = [31, 32, 34]
    best_size = len(best_set)
    
    def compute_sums(S):
        """Compute all subset sums of S (including 0)."""
        sums = {0}
        for x in S:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums = sums | new_sums
        return sums

    def is_antichain(sums_set):
        """
        Check if sums_set (excluding 0) is an antichain under divisibility.
        Optimized: For each u, check if any multiple of u is in the set.
        """
        if not sums_set or len(sums_set) <= 1:
            return True
        
        # If 1 is in sums_set, it divides everything
        if 1 in sums_set and len(sums_set) > 1:
            return False
            
        max_sum = max(sums_set)
        
        # Sort to potentially early-exit? No, divisibility is order-independent.
        # Key optimization: only check u where 2*u <= max_sum
        # For large u, there are few multiples to check
        
        for u in sums_set:
            if u == 1:
                continue
            if 2 * u > max_sum:
                continue
            # Check multiples 2u, 3u, ... up to max_sum
            multiple = 2 * u
            while multiple <= max_sum:
                if multiple in sums_set:
                    return False
                multiple += u
        return True

    def check_validity_incremental(S, x, sums_S):
        """
        Checks if S U {x} is valid, given S is valid.
        Returns (True, new_sums) if valid, (False, None) otherwise.
        """
        if x in S:
            return False, None
            
        # Generate new sums: x + s for s in sums_S
        # sums_S contains 0.
        new_sums = {x + s for s in sums_S}
        
        # Check distinctness: sums_S and new_sums must be disjoint
        if sums_S & new_sums:
            return False, None
        
        all_sums = sums_S | new_sums
        non_empty_all = all_sums - {0}
        
        # Check divisibility antichain
        if not is_antichain(non_empty_all):
            return False, None
        
        return True, all_sums

    def local_search_improve(S, sums_S, remaining_candidates, deadline):
        """
        Try to improve the current set by swapping or adding elements.
        Uses a limited backtracking approach.
        """
        nonlocal best_set, best_size
        
        current_S = list(S)
        current_sums = set(sums_S)
        current_size = len(current_S)
        
        # Try to add one more element from remaining candidates
        # Sort candidates by "safety" - prefer larger numbers (fewer divisors)
        sorted_candidates = sorted(remaining_candidates, reverse=True)
        
        for x in sorted_candidates:
            if time.time() > deadline:
                break
            
            valid, new_sums = check_validity_incremental(current_S, x, current_sums)
            if valid:
                current_S.append(x)
                current_sums = new_sums
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_S[:]
                
                # Try to add another element
                for y in sorted_candidates:
                    if y == x or time.time() > deadline:
                        continue
                    valid2, new_sums2 = check_validity_incremental(current_S, y, current_sums)
                    if valid2:
                        current_S.append(y)
                        current_sums = new_sums2
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_set = current_S[:]
                        current_S.pop()
                        current_sums = current_sums - {y + s for s in current_sums if (y+s) not in (current_sums - {y + s2 for s2 in current_sums})}
                        # Recompute properly
                        current_sums = compute_sums(current_S)
                        current_size = len(current_S)
        
        # Try swapping: remove one element, try to add two
        for i in range(len(current_S)):
            if time.time() > deadline:
                break
            
            removed = current_S[i]
            S_without = current_S[:i] + current_S[i+1:]
            sums_without = compute_sums(S_without)
            
            # Try adding two elements
            for x in sorted_candidates:
                if x == removed or time.time() > deadline:
                    continue
                valid1, sums_with_x = check_validity_incremental(S_without, x, sums_without)
                if not valid1:
                    continue
                
                for y in sorted_candidates:
                    if y == x or y == removed or time.time() > deadline:
                        continue
                    valid2, sums_with_xy = check_validity_incremental(S_without + [x], y, sums_with_x)
                    if valid2:
                        new_S = S_without + [x, y]
                        if len(new_S) > best_size:
                            best_size = len(new_S)
                            best_set = new_S[:]
        
        return current_S, current_sums

    # Search Strategy
    random.seed(42)
    
    num_restarts = 5000
    for attempt in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        S = []
        sums_S = {0}
        
        candidates = list(range(1, 257))
        
        # Different strategies for different attempts
        if attempt % 5 == 0:
            # Descending (largest first) - most effective
            candidates.sort(reverse=True)
        elif attempt % 5 == 1:
            # Random shuffle
            random.shuffle(candidates)
        elif attempt % 5 == 2:
            # Bias towards larger numbers
            candidates.sort(key=lambda x: random.random() ** 0.5 * x, reverse=True)
        elif attempt % 5 == 3:
            # Start with medium-large, then random
            large = list(range(128, 257))
            small = list(range(1, 128))
            random.shuffle(large)
            random.shuffle(small)
            candidates = large + small
        else:
            # Primorial-like bias: prefer numbers with few small prime factors
            # Actually, prefer numbers that are "independent" - not multiples of each other
            # Sort by number of divisors (fewer is better for antichain)
            def count_divisors(n):
                count = 0
                for i in range(1, int(n**0.5) + 1):
                    if n % i == 0:
                        count += 2
                        if i != n // i:
                            count -= 1
                return count
            candidates.sort(key=lambda x: (count_divisors(x), -x))
        
        for x in candidates:
            if time.time() - start_time > time_limit:
                break
            
            valid, new_sums = check_validity_incremental(S, x, sums_S)
            if valid:
                S.append(x)
                sums_S = new_sums
                if len(S) > best_size:
                    best_size = len(S)
                    best_set = S[:]
        
        # After greedy, try local improvement
        if len(S) >= best_size - 1:
            remaining = [c for c in range(1, 257) if c not in S]
            S, sums_S = local_search_improve(S, sums_S, remaining, start_time + time_limit)
    
    # Final verification
    if best_set:
        final_sums = compute_sums(best_set)
        non_empty_final = final_sums - {0}
        if len(non_empty_final) != (1 << len(best_set)) - 1:
            best_set = [31, 32, 34]
            best_size = 3
        elif not is_antichain(non_empty_final):
            best_set = [31, 32, 34]
            best_size = 3
    
    return list(best_set)

if __name__ == "__main__":
    res = construct()
    print(res)
