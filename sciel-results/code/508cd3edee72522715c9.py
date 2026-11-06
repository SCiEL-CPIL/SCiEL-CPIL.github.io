import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    max_n = 512
    max_size = 48

    def can_add(x, S, dp):
        """
        Checks if x can be added to S without violating constraints.
        dp[k] is a bitset of achievable sums using k elements from S.
        Returns True if x can be added, False otherwise.
        """
        n_s = len(S)
        
        # Check 1: x is not the mean of any subset T of S with |T| >= 2
        # sum(T) = x * |T|
        for k in range(2, n_s + 1):
            target_sum = x * k
            if (dp[k] >> target_sum) & 1:
                return False
        
        # Check 2: No y in S becomes the mean of a subset involving x
        # y is mean of T' = T U {x}, T subset S \ {y}, |T| = j >= 1
        # sum(T) + x = y * (j + 1) => sum(T) = y * (j + 1) - x
        # We check if sum(T) is achievable by j elements from S.
        # Conservative: If achievable from S, assume it's achievable from S\{y} (reject).
        # This ensures validity.
        for y in S:
            for j in range(1, n_s + 1):
                target_sum = y * (j + 1) - x
                if target_sum < 0:
                    continue
                if (dp[j] >> target_sum) & 1:
                    return False
                    
        return True

    def greedy_build(order):
        """
        Builds a set using the given order of candidates.
        Returns the set of integers.
        """
        S = []
        dp = [0] * (max_size + 1)
        dp[0] = 1
        
        for x in order:
            if len(S) >= max_size:
                break
            if time.time() > deadline - 5.0:
                break
                
            if can_add(x, S, dp):
                S.append(x)
                # Update DP
                # Iterate backwards to avoid using x multiple times
                for k in range(len(S) - 1, -1, -1):
                    dp[k + 1] |= (dp[k] << x)
                    
        return S

    def try_fill(S):
        """
        Attempts to add more elements to S by scanning all candidates.
        """
        if time.time() > deadline - 5.0:
            return S
            
        S = list(S)
        dp = [0] * (max_size + 1)
        dp[0] = 1
        for x in S:
            for k in range(len(S) - 1, -1, -1):
                dp[k + 1] |= (dp[k] << x)
                
        changed = True
        while changed and len(S) < max_size:
            changed = False
            for x in range(1, max_n + 1):
                if x in S:
                    continue
                if time.time() > deadline - 5.0:
                    break
                if can_add(x, S, dp):
                    S.append(x)
                    for k in range(len(S) - 1, -1, -1):
                        dp[k + 1] |= (dp[k] << x)
                    changed = True
                    break # Restart scan to ensure consistency
        return S

    def sat_search(initial_set, target_size):
        """
        SAT-based conflict-driven search to extend initial_set to target_size.
        Uses a simple backtracking with conflict learning.
        """
        if time.time() > deadline - 5.0:
            return list(initial_set)
        
        S = list(initial_set)
        S_set = set(S)
        dp = [0] * (max_size + 1)
        dp[0] = 1
        for x in S:
            for k in range(len(S) - 1, -1, -1):
                dp[k + 1] |= (dp[k] << x)
        
        # Conflict tracking: for each element, track which other elements caused conflicts
        conflict_count = defaultdict(int)
        # Track which candidates have been tried and failed
        tried_and_failed = set()
        
        # Precompute conflict scores for ordering
        def get_conflict_score(x):
            """Estimate how many conflicts x would cause if added."""
            score = 0
            for y in S:
                # Check if x would make y a mean of some subset
                for j in range(1, len(S) + 1):
                    target_sum = y * (j + 1) - x
                    if target_sum >= 0 and (dp[j] >> target_sum) & 1:
                        score += 1
                        break
                # Check if x is mean of subset involving y
                for k in range(2, len(S) + 1):
                    target_sum = x * k
                    if (dp[k] >> target_sum) & 1:
                        score += 1
                        break
            return score
        
        # Backtracking search with conflict-driven variable ordering
        def backtrack(current_S, current_dp, depth):
            if time.time() > deadline - 5.0:
                return None
            if len(current_S) >= target_size:
                return list(current_S)
            
            # Get candidates not in current_S
            candidates = [x for x in range(1, max_n + 1) if x not in current_S]
            
            # Order by conflict score (ascending - least conflicts first)
            candidates.sort(key=lambda x: get_conflict_score(x))
            
            for x in candidates:
                if x in tried_and_failed:
                    continue
                
                # Try adding x
                if can_add(x, current_S, current_dp):
                    new_S = current_S + [x]
                    new_dp = current_dp[:]
                    for k in range(len(new_S) - 1, -1, -1):
                        new_dp[k + 1] |= (new_dp[k] << x)
                    
                    result = backtrack(new_S, new_dp, depth + 1)
                    if result is not None:
                        return result
                    else:
                        # Conflict: mark x as failed with current_S
                        tried_and_failed.add(x)
                        # Update conflict counts
                        conflict_count[x] += 1
                else:
                    # Direct conflict, skip
                    tried_and_failed.add(x)
                    conflict_count[x] += 1
            
            return None
        
        result = backtrack(S, dp, 0)
        if result is not None:
            return result
        return S

    best_set = []
    
    # 1. Deterministic Greedy with different orderings
    
    # Descending
    order_desc = range(max_n, 0, -1)
    S_desc = greedy_build(order_desc)
    best_set = S_desc
    best_size = len(best_set)
    
    # Ascending
    order_asc = range(1, max_n + 1)
    S_asc = greedy_build(order_asc)
    if len(S_asc) > best_size:
        best_set = S_asc
        best_size = len(S_asc)
        
    # 2. Heuristic: "Edge-First"
    # Numbers far from the center are less likely to be the mean of two other numbers.
    # Center is 256.5. Distance = abs(x - 256.5)
    # Sort by distance descending (edges first)
    order_edge = sorted(range(1, max_n + 1), key=lambda x: -abs(x - 256.5))
    S_edge = greedy_build(order_edge)
    if len(S_edge) > best_size:
        best_set = S_edge
        best_size = len(S_edge)
        
    # 3. Heuristic: "Safe" order based on pre-computed 3AP conflict counts
    # Score(x) = number of y in [1, 512]\{x} such that 2x-y is in [1, 512]\{x,y}
    # This is the number of 3-term APs centered at x.
    # Lower score = safer to include.
    scores = {}
    for x in range(1, max_n + 1):
        count = 0
        # u < x < v, u+v = 2x
        # u ranges from max(1, 2x-512) to x-1
        # v = 2x - u
        # v <= 512 => 2x - u <= 512 => u >= 2x - 512
        # v > x => 2x - u > x => x > u (always true for u < x)
        # u >= 1
        start_u = max(1, 2*x - 512)
        end_u = x - 1
        if start_u <= end_u:
            count = end_u - start_u + 1
        scores[x] = count
        
    # Sort by score ascending (safest first)
    order_safe = sorted(range(1, max_n + 1), key=lambda x: scores[x])
    S_safe = greedy_build(order_safe)
    if len(S_safe) > best_size:
        best_set = S_safe
        best_size = len(S_safe)
        
    # 4. SAT-based search to extend the best greedy result
    if best_size < max_size and time.time() < deadline - 10.0:
        sat_result = sat_search(best_set, max_size)
        if len(sat_result) > best_size:
            best_set = sat_result
            best_size = len(best_set)
        
    # 5. Randomized Greedy with Heuristic Bias
    random.seed(42)
    
    while time.time() < deadline - 15.0:
        # Create a biased order:
        # With probability 0.5, pick from the top 50% safest numbers.
        # With probability 0.5, pick from all numbers.
        # This is complex to implement efficiently in a single pass.
        
        # Simpler: Just try more random restarts.
        # But also try the "safe" order with some noise.
        
        # Let's try a few specific heuristic variations if time permits
        if len(best_set) < max_size:
            # Try a "mixed" order: first half safe, second half random
            safe_half = order_safe[:max_n//2]
            rest = [x for x in range(1, max_n+1) if x not in set(safe_half)]
            random.shuffle(rest)
            mixed_order = safe_half + rest
            S_mixed = greedy_build(mixed_order)
            if len(S_mixed) > best_size:
                best_set = S_mixed
                best_size = len(S_mixed)
                if best_size >= max_size:
                    break

        # Standard random restart
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        S_rand = greedy_build(order)
        if len(S_rand) > best_size:
            best_set = S_rand
            best_size = len(best_set)
            if best_size >= max_size:
                break
                
    # 6. Try to fill gaps in the best set found
    if time.time() < deadline - 5.0:
        filled_set = try_fill(best_set)
        if len(filled_set) > best_size:
            best_set = filled_set
            best_size = len(filled_set)
            
    # 7. Final SAT attempt on the filled set
    if best_size < max_size and time.time() < deadline - 3.0:
        final_sat = sat_search(best_set, max_size)
        if len(final_sat) > best_size:
            best_set = final_sat
            best_size = len(best_set)
        
    return best_set
