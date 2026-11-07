import time
import random
import sys

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin for 90s limit
    
    # We want to maximize the size of the set.
    # Theoretical upper bound:
    # For a set of size k, the sums for cardinality c range roughly from 
    # sum of smallest c to sum of largest c.
    # To keep them disjoint, we need "gaps".
    # A simple lower bound on the largest element x_k for size k is often exponential.
    # However, with 256 as the max value, k is likely small (around 10-15).
    # Current best is 2.25 * seed. If seed is ~10, best is ~22? No, score is value/seed.
    # If seed program returns 10, score 1.0. Current program score 2.25 => value 22.5?
    # Wait, "raw value is the number of integers returned".
    # If score is 2.25 and seed is 10, value is 22.5? Impossible, must be int.
    # Maybe seed is 8? 2.25 * 8 = 18.
    # Let's assume we are targeting k=18, 19, 20...
    
    # Strategy:
    # 1. Start with a small valid set (e.g., powers of 2).
    # 2. Try to add a new element.
    # 3. Use local search to adjust existing elements to make room for the new one.
    # 4. Objective: Minimize the "collision penalty" where sums from different cardinalities overlap.
    
    # Validity Check using DP
    def check_validity(s):
        n = len(s)
        if n == 0:
            return True, 0
        # dp[c] is a set of sums for cardinality c
        dp = [set() for _ in range(n + 1)]
        dp[0].add(0)
        for x in s:
            for c in range(n - 1, -1, -1):
                if dp[c]:
                    # Optimization: if sets are large, this is slow.
                    # But n is small (< 30), so 2^n is up to 10^9 which is too slow for python sets if done naively per step.
                    # Actually, number of subsets is 2^n. For n=20, 10^6 subsets.
                    # DP with sets: sum of sizes of sets.
                    # Max size of dp[c] is C(n,c). Sum is 2^n.
                    # For n=20, 2^20 = 1,000,000. This is feasible in Python if optimized.
                    # But we do this in a loop of local search steps.
                    # We need a faster validity check or penalty calculation.
                    
                    # Alternative: Compute min and max sums for each cardinality.
                    # If intervals [min_sum_c, max_sum_c] are disjoint, it's valid.
                    # This is a sufficient condition, not necessary.
                    # But it's O(n^2) to compute mins/maxs.
                    # Let's use interval disjointness as the primary fast filter.
                    pass
        
        # Fast Interval Check
        # Sort s
        s_sorted = sorted(s)
        n = len(s_sorted)
        prefix_sums = [0] * (n + 1)
        for i in range(n):
            prefix_sums[i+1] = prefix_sums[i] + s_sorted[i]
            
        # Min sum for cardinality c is sum of first c elements
        # Max sum for cardinality c is sum of last c elements
        min_sums = [0] * (n + 1)
        max_sums = [0] * (n + 1)
        for c in range(n + 1):
            min_sums[c] = prefix_sums[c]
            max_sums[c] = prefix_sums[n] - prefix_sums[n-c]
            
        # Check if intervals [min_sums[c], max_sums[c]] are disjoint for all c != d
        # Since min_sums is increasing and max_sums is increasing, 
        # it suffices to check if max_sums[c] < min_sums[c+1] for all c?
        # No, that's only for adjacent cardinalities.
        # Actually, if max_sums[c] < min_sums[c+1] for all c, then all are disjoint.
        # Is this condition necessary? No. But it's very restrictive.
        # Example: {1, 2, 4}. 
        # c=0: [0,0]
        # c=1: [1,4]
        # c=2: [3,6]
        # c=3: [7,7]
        # max(0)=0 < min(1)=1. OK.
        # max(1)=4 < min(2)=3? NO. 4 is not < 3.
        # So interval check fails, but {1,2,4} is valid.
        # Sums:
        # 0: {0}
        # 1: {1,2,4}
        # 2: {3,5,6}
        # 3: {7}
        # All disjoint.
        # So interval check is too strong. It rejects valid sets.
        
        # We need exact DP for validity, but it's slow.
        # Let's use a hybrid: 
        # 1. Interval check for fast rejection (if it passes, it's definitely valid? No, if it fails, it might still be valid).
        # Actually, if intervals overlap, sums MIGHT overlap. 
        # If intervals are disjoint, sums ARE disjoint.
        # So:
        # If intervals disjoint -> Valid (Fast Path)
        # Else -> Exact DP Check (Slow Path)
        
        # Wait, if intervals are disjoint, is it always valid?
        # Yes, because all sums of cardinality c are within [min_sums[c], max_sums[c]].
        # If these intervals don't intersect, the sets of sums don't intersect.
        
        # So, strategy:
        # 1. Try to satisfy the "Interval Disjointness" condition. This is a linear programming problem!
        #    max_sums[c] < min_sums[c+1]
        #    sum(s[n-c+1..n]) < sum(s[1..c+1])  (1-based indexing)
        #    This is exactly the constraint in the current MILP program!
        #    The current program uses MILP to satisfy this sufficient condition.
        #    If we can satisfy this, we are done.
        
        # Why did the current program only get 2.25?
        # Maybe the MILP solver is too slow for larger k, or the condition is too restrictive to allow k > 18?
        # If the condition is too restrictive, we might miss valid sets where intervals overlap but sums don't.
        
        # New Strategy:
        # 1. Use the MILP result (or a greedy construction satisfying the interval condition) as a starting point.
        # 2. Use Local Search to "relax" the interval condition, allowing intervals to overlap, 
        #    while ensuring via Exact DP that sums don't actually collide.
        # 3. Objective: Minimize collisions. If 0 collisions, it's valid.
        
        # Let's implement a Local Search that starts with a set satisfying the interval condition (if possible)
        # or a random set, and tries to reduce collisions.
        
        # Exact DP Check (Optimized)
        def exact_check(s):
            n = len(s)
            dp = [set() for _ in range(n + 1)]
            dp[0].add(0)
            for x in s:
                for c in range(n - 1, -1, -1):
                    if dp[c]:
                        # Add x to all sums in dp[c] and add to dp[c+1]
                        # To speed up, we can check if new sums collide with existing dp[c+1]
                        # But we need the full set for future steps.
                        for v in dp[c]:
                            dp[c+1].add(v + x)
            
            # Check for collisions between different cardinalities
            seen = set()
            for c in range(n + 1):
                if not seen.isdisjoint(dp[c]):
                    return False, len(seen.intersection(dp[c]))
                seen.update(dp[c])
            return True, 0

        # Fast Interval Check
        def interval_check(s):
            s_sorted = sorted(s)
            n = len(s_sorted)
            prefix_sums = [0] * (n + 1)
            for i in range(n):
                prefix_sums[i+1] = prefix_sums[i] + s_sorted[i]
            
            min_sums = [0] * (n + 1)
            max_sums = [0] * (n + 1)
            for c in range(n + 1):
                min_sums[c] = prefix_sums[c]
                max_sums[c] = prefix_sums[n] - prefix_sums[n-c]
                
            for c in range(n):
                if max_sums[c] >= min_sums[c+1]:
                    return False
            return True

        # Local Search Objective: 
        # We want to find a set of size k with 0 collisions.
        # Penalty = Number of collisions * 1000 + (max_val - 256 if max_val > 256 else 0) * 1000 + (min_val < 1 if min_val < 1 else 0) * 1000
        # Actually, we constrain values to [1, 256] during moves.
        
        # Start with a small valid set
        current_set = [1, 2, 4, 8, 16, 32, 64, 128]
        
        # Try to grow the set
        max_k_found = len(current_set)
        
        # We will try to increase size by 1, then local search to fix collisions.
        # If we fail, we try smaller k? No, we want max k.
        
        # Let's try to construct sets of size k from 10 to 25.
        # For each k, we start with a random set or a structured one, and local search.
        
        # Better: Start with the best known valid set (from interval condition if we can find one fast, else random).
        # Let's try to generate a set satisfying the interval condition greedily first.
        # x_1 = 1
        # x_{i+1} > sum(x_1...x_i) + 1 ? No, that's superincreasing, gives 2^n sums, all distinct.
        # Superincreasing: x_{i+1} > sum(x_1...x_i).
        # Then all subset sums are distinct. This is a sufficient condition for validity (even stronger).
        # Max size for superincreasing in 1..256:
        # 1, 2, 4, 8, 16, 32, 64, 128. Next would be 256 (if 128*2+1 > 256? No, 257 > 256).
        # So max size 8 for superincreasing.
        
        # The interval condition is weaker.
        # Let's use the MILP logic to find a starting point for larger k, but with a very short timeout.
        # If MILP fails, we use a random start for local search.
        
        # However, the prompt says "Maximize set size using exact cardinality-indexed subset-sum dynamic programming".
        # This implies we should use DP for the check, which we do.
        
        # Let's run a multi-start local search for k from 10 to 25.
        
        def local_search(start_set, k, time_limit):
            s = start_set[:]
            random.shuffle(s) # Randomize order? No, set is unordered.
            s.sort()
            
            # Ensure size k
            # If len(s) < k, add random values
            while len(s) < k:
                s.append(random.randint(1, 256))
                s.sort()
            if len(s) > k:
                s = s[:k]
                s.sort()
                
            # Initial penalty
            valid, collisions = exact_check(s)
            current_penalty = collisions * 1000
            # Add penalty for duplicates or out of bounds (shouldn't happen if we manage moves)
            
            best_set = s[:]
            best_penalty = current_penalty
            
            temp = 10.0
            cool_rate = 0.95
            
            t_start = time.time()
            no_improve_count = 0
            
            while time.time() - t_start < time_limit:
                # Generate a move
                # 1. Change one element
                # 2. Swap two elements?
                # 3. Add/Remove? (Keep size fixed)
                
                i = random.randint(0, k-1)
                old_val = s[i]
                # Perturbation: random walk in value space
                delta = random.randint(-10, 10)
                new_val = old_val + delta
                if new_val < 1: new_val = 1
                if new_val > 256: new_val = 256
                
                # Ensure distinctness? 
                # If new_val is already in s (at j != i), it's invalid.
                # We can allow it in the move but penalize heavily, or reject.
                # Let's reject if duplicate to keep search space clean.
                if s.count(new_val) > 0 and s.index(new_val) != i:
                    # Try again or skip
                    continue
                
                s[i] = new_val
                s.sort()
                
                # Check validity
                valid, collisions = exact_check(s)
                new_penalty = collisions * 1000
                
                # Acceptance
                if new_penalty < current_penalty:
                    current_penalty = new_penalty
                    no_improve_count = 0
                    if new_penalty < best_penalty:
                        best_penalty = new_penalty
                        best_set = s[:]
                else:
                    # Metropolis criterion
                    if temp > 0:
                        delta_p = new_penalty - current_penalty
                        if random.random() < pow(2.718, -delta_p / temp):
                            current_penalty = new_penalty
                    # If rejected, revert
                    s[i] = old_val
                    s.sort()
                    
                temp *= cool_rate
                if temp < 0.1:
                    temp = 0.1 # Floor to allow some exploration
            
            return best_set, best_penalty

        # Main Loop
        # We try to find the largest k.
        # We know k=8 is easy. 
        # Let's try k=10, 11, ..., up to 20.
        
        # We will run local search for each k.
        # To save time, we run them in parallel? No, single thread.
        # We allocate time budget based on k.
        
        best_global_set = current_set
        best_global_size = len(current_set)
        
        # Try k from 9 to 25
        # We give more time to larger k? Or equal time?
        # Let's give 5 seconds per k attempt, but stop if we find a valid one.
        
        # Actually, let's try to jump to larger k using the interval condition as a hint.
        # If we can find a set satisfying the interval condition for k, it's valid.
        # Let's try a greedy construction for the interval condition first, as it's faster than MILP.
        
        def greedy_interval_set(k):
            # x_1 = 1
            # x_{i+1} must be > sum of largest c ... 
            # This is complex. Let's just use random restarts for local search.
            return None

        # Let's just do local search for k = 9 to 25.
        # We start with a random set of size k.
        
        for k in range(9, 26):
            if time.time() > deadline - 5.0:
                break
                
            # Time budget for this k
            # If we have 80s left, and 17 k's to try, ~4s each.
            # But we should prioritize finding ANY valid set for larger k.
            # Let's do 3 restarts for each k, with 1s each.
            
            found = False
            for restart in range(3):
                if time.time() > deadline - 5.0:
                    break
                    
                # Start with a random set of size k
                # To make it better, start with values spread out
                s_init = random.sample(range(1, 257), k)
                
                res_set, penalty = local_search(s_init, k, 2.0)
                
                if penalty == 0:
                    # Valid!
                    if k > best_global_size:
                        best_global_size = k
                        best_global_set = res_set
                    found = True
                    # If we found a valid set for k, we can break the restart loop for this k
                    # and move to k+1
                    break
            
            # If we found a valid set for k, we continue to k+1
            # If not, we continue to k+1 (maybe a different random start works)
            
        return [int(x) for x in best_global_set]

    # Fallback if something goes wrong
    return [1, 2, 4, 8, 16, 32, 64, 128]

# Ensure the function is named construct
def construct():
    start = time.time()
    deadline = start + 85.0
    
    def exact_check(s):
        n = len(s)
        dp = [set() for _ in range(n + 1)]
        dp[0].add(0)
        for x in s:
            for c in range(n - 1, -1, -1):
                if dp[c]:
                    for v in dp[c]:
                        dp[c+1].add(v + x)
        
        seen = set()
        for c in range(n + 1):
            if not seen.isdisjoint(dp[c]):
                return False, len(seen.intersection(dp[c]))
            seen.update(dp[c])
        return True, 0

    def local_search(start_set, k, time_limit):
        s = start_set[:]
        s.sort()
        
        while len(s) < k:
            s.append(random.randint(1, 256))
            s.sort()
        if len(s) > k:
            s = s[:k]
            s.sort()
            
        valid, collisions = exact_check(s)
        current_penalty = collisions * 1000
        best_set = s[:]
        best_penalty = current_penalty
        
        temp = 10.0
        cool_rate = 0.95
        t_start = time.time()
        
        while time.time() - t_start < time_limit:
            i = random.randint(0, k-1)
            old_val = s[i]
            delta = random.randint(-10, 10)
            new_val = old_val + delta
            if new_val < 1: new_val = 1
            if new_val > 256: new_val = 256
            
            # Check duplicate
            if new_val in s and s.index(new_val) != i:
                continue
                
            s[i] = new_val
            s.sort()
            
            valid, collisions = exact_check(s)
            new_penalty = collisions * 1000
            
            if new_penalty < current_penalty:
                current_penalty = new_penalty
                if new_penalty < best_penalty:
                    best_penalty = new_penalty
                    best_set = s[:]
            else:
                if temp > 0:
                    delta_p = new_penalty - current_penalty
                    if random.random() < pow(2.718, -delta_p / temp):
                        current_penalty = new_penalty
                    else:
                        s[i] = old_val
                        s.sort()
            
            temp *= cool_rate
            if temp < 0.1:
                temp = 0.1
        
        return best_set, best_penalty

    best_global_set = [1, 2, 4, 8, 16, 32, 64, 128]
    best_global_size = len(best_global_set)
    
    for k in range(9, 26):
        if time.time() > deadline - 5.0:
            break
            
        found = False
        for restart in range(3):
            if time.time() > deadline - 5.0:
                break
                
            s_init = random.sample(range(1, 257), k)
            res_set, penalty = local_search(s_init, k, 2.0)
            
            if penalty == 0:
                if k > best_global_size:
                    best_global_size = k
                    best_global_set = res_set
                found = True
                break
                
    return [int(x) for x in best_global_set]

