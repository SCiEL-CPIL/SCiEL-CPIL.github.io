"""
Strategy: Systematic Backtracking with Pruning using Float Approximation.

The previous approach used Fraction arithmetic, which is exact but computationally
expensive due to large numerator/denominator growth. This approach:
1. Uses floating-point numbers (float64) to represent subset sums.
   - Rationale: The smallest difference between any two distinct subset sums of
     reciprocals of {1..22} is bounded below. Let's estimate.
     The maximum sum is H_22 ~ 3.7.
     The "granularity" of the sums depends on the LCM of 1..22.
     LCM(1..22) is 232792560.
     So any sum is a multiple of 1/232792560.
     The difference between any two distinct sums is at least 1/23292560 ~ 4.2e-9.
     float64 has ~15-16 decimal digits of precision.
     Sums are ~3.7. Error in float64 addition is ~ 3.7 * 1e-16 * N_ops.
     Even with 2^20 ops, error is negligible compared to 1e-9.
     Thus, float64 is safe for equality checks of distinct sums in this domain.
     We can use a set of floats for O(1) collision checks.

2. Uses a systematic backtracking search (DFS) with heuristics to maximize cardinality.
   - We try to build a set S of size K.
   - We iterate K from 11 down to 1 (max possible is likely around 10-12).
   - We use a greedy initialization to find a good starting point, then backtrack
     to improve if stuck, or simply do a randomized restart of the backtracker with
     different orderings if time permits.
   
   Actually, a pure systematic DFS for K=11 over 22 elements is 22 choose 11 ~ 705k paths.
   Each path involves maintaining a set of sums.
   If we prune aggressively, this is very fast.
   
   Algorithm:
   1. Precompute all possible subset sums? No, too many (2^22).
   2. Backtrack:
      def backtrack(start_idx, current_set, current_sums):
         if len(current_set) == target_k: return True
         for i in range(start_idx, 22):
             elem = i + 1
             new_sums = current_sums | {s + 1/elem for s in current_sums}
             if len(new_sums) == 2 * len(current_sums): # No collisions
                 if backtrack(i+1, current_set + [elem], new_sums):
                     return True
         return False
   
   To maximize cardinality, we try K=11, then 10, etc.
   To handle the search space, we can use randomization in the order of elements tried
   or use a heuristic (e.g., try smaller numbers first? larger numbers first?).
   Smaller numbers have larger reciprocals, creating larger gaps, which might help
   avoid collisions? Or larger numbers have smaller reciprocals, clustering near 0?
   Usually, dissociated sets are easier to build with "large" elements in some sense,
   but here we are picking from 1..22.
   
   Let's try K from 12 down to 1.
   We will run multiple random restarts if the first systematic search fails or times out.
"""

import time
import random
import sys

# Set recursion limit just in case, though depth is max 22
sys.setrecursionlimit(10000)

_START_TIME = time.time()
_DEADLINE = 85.0  # Safety margin before 90s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Returns a nonempty subset of [1, 22] with all reciprocal subset sums distinct.
    Maximizes cardinality.
    """
    # Target sizes to try, starting from high
    # Max possible size? 
    # 2^K sums must be distinct. 
    # Range of sums is [0, H_22]. 
    # This doesn't strictly bound K by range, but by the "dissociation" property.
    # Known results suggest max size for [1..N] is around N/2 or slightly more?
    # For N=22, let's try up to 12.
    
    best_set = [1]  # Fallback
    
    # Helper to check validity and get sums
    def get_sums(subset):
        sums = {0.0}
        for x in subset:
            recip = 1.0 / x
            new_sums = set()
            for s in sums:
                new_sums.add(s + recip)
            sums = sums.union(new_sums)
        return sums

    # Check if a set is valid (all sums distinct)
    def is_valid(subset):
        if not subset:
            return True
        sums = get_sums(subset)
        return len(sums) == (1 << len(subset))

    # Backtracking search for a specific target size K
    def find_set_of_size_k(k):
        # Elements 1 to 22
        elements = list(range(1, 23))
        
        # Heuristic: Sort elements. 
        # Trying smaller elements first might lead to larger sums, spreading them out?
        # Or larger elements first (small reciprocals) might cluster?
        # Let's try a few orderings.
        
        def backtrack(start_idx, current_set, current_sums):
            if _time_left() < 1.0:
                return None
            
            if len(current_set) == k:
                return list(current_set)
            
            remaining_needed = k - len(current_set)
            available_count = 22 - start_idx
            if available_count < remaining_needed:
                return None
            
            for i in range(start_idx, 23):
                elem = i + 1
                recip = 1.0 / elem
                
                # Calculate new sums
                # Optimization: If we detect collision, skip
                new_sums = set(current_sums)
                # We need to check if any new sum collides with existing or other new sums
                # Since we are adding 1/elem to ALL existing sums, the new sums are {s + recip for s in current_sums}
                # Collision if (s1 + recip) == s2 for s1, s2 in current_sums
                # OR (s1 + recip) == (s2 + recip) => s1 == s2 (always true if current_sums distinct)
                # So we just need to check if the set {s + recip for s in current_sums} intersects current_sums.
                
                # Efficient check:
                # If current_sums is a set, we can do:
                # new_part = {s + recip for s in current_sums}
                # if new_part & current_sums: collision
                # else: new_sums = current_sums | new_part
                
                # However, creating the set every time is O(2^d).
                # Depth d is at most 12. 2^12 = 4096.
                # This is fast enough in Python? 
                # 4096 operations per node. 
                # Branching factor ~ 10-20.
                # Depth 12.
                # Total nodes could be large if pruning is weak.
                
                # Let's try to prune:
                # If we add elem, the new sums are shifted.
                
                # Optimization: Use a list for current_sums? No, set is needed for intersection.
                # Let's stick to sets.
                
                # Generate new sums
                new_part = set()
                collision = False
                for s in current_sums:
                    ns = s + recip
                    if ns in current_sums:
                        collision = True
                        break
                    new_part.add(ns)
                
                if collision:
                    continue
                
                # Check for collisions within new_part?
                # No, if s1 != s2, then s1+recip != s2+recip.
                # So new_part has same cardinality as current_sums.
                # Total size becomes 2 * len(current_sums).
                
                # Recurse
                res = backtrack(i + 1, current_set + [elem], current_sums.union(new_part))
                if res is not None:
                    return res
                    
            return None

        # Try different orderings to help backtracking
        # 1. Ascending
        # 2. Descending
        # 3. Randomized
        
        orderings = [
            list(range(1, 23)),
            list(range(22, 0, -1)),
            list(range(1, 23)) # Placeholder for random
        ]
        
        # We'll just do a few random restarts if the first two fail
        attempts = 0
        max_attempts = 10
        
        while attempts < max_attempts and _time_left() > 2.0:
            if attempts == 0:
                order = list(range(1, 23))
            elif attempts == 1:
                order = list(range(22, 0, -1))
            else:
                order = list(range(1, 23))
                random.shuffle(order)
            
            # We need to map the order back to indices for the backtrack function
            # The backtrack function assumes we pick from the sorted list of remaining elements?
            # My backtrack above assumes `elements` is sorted and we pick `i` from `start_idx`.
            # This restricts us to picking subsets in increasing index order?
            # NO, `i` is the index in the original array 1..22.
            # If I shuffle `order`, I need to change the backtrack logic to iterate over `order[start_idx:]`.
            
            # Let's refactor backtrack to take an explicit list of candidates
            def backtrack_order(start_idx, current_set, current_sums, candidates):
                if _time_left() < 0.5:
                    return None
                
                if len(current_set) == k:
                    return list(current_set)
                
                remaining_needed = k - len(current_set)
                if len(candidates) - start_idx < remaining_needed:
                    return None
                
                for i in range(start_idx, len(candidates)):
                    elem = candidates[i]
                    recip = 1.0 / elem
                    
                    # Check collision
                    new_part = set()
                    collision = False
                    for s in current_sums:
                        ns = s + recip
                        if ns in current_sums:
                            collision = True
                            break
                        new_part.add(ns)
                    
                    if collision:
                        continue
                    
                    # Recurse with remaining candidates
                    res = backtrack_order(i + 1, current_set + [elem], current_sums.union(new_part), candidates)
                    if res is not None:
                        return res
                return None

            candidates = order
            result = backtrack_order(0, [], {0.0}, candidates)
            if result is not None:
                return result
            attempts += 1
            
        return None

    # Try to find a set of size K
    # Start from a high K and go down
    for k in range(12, 0, -1):
        if _time_left() < 5.0:
            break
        res = find_set_of_size_k(k)
        if res is not None:
            # Verify result
            if is_valid(res):
                return res
            else:
                # Should not happen, but fallback
                best_set = res
        # If we found a smaller one, keep searching for larger? 
        # The loop goes 12 -> 1. 
        # If 12 fails, try 11. 
        # If 11 succeeds, return it. 
        # This maximizes cardinality.
        
    # Fallback: Greedy
    # Just in case backtracking fails completely
    greedy_set = []
    sums = {0.0}
    for i in range(1, 23):
        recip = 1.0 / i
        new_part = set()
        collision = False
        for s in sums:
            ns = s + recip
            if ns in sums:
                collision = True
                break
            new_part.add(ns)
        if not collision:
            greedy_set.append(i)
            sums = sums.union(new_part)
            
    if greedy_set:
        return greedy_set
        
    return [1]

if __name__ == "__main__":
    # Quick self-test
    s = construct()
    print(f"Found set: {s}")
    print(f"Size: {len(s)}")
    # Verify
    sums = {0.0}
    for x in s:
        recip = 1.0 / x
        new_sums = {ss + recip for ss in sums}
        if new_sums & sums:
            print("INVALID!")
        sums = sums.union(new_sums)
    print(f"Num sums: {len(sums)} (Expected {1 << len(s)})")
