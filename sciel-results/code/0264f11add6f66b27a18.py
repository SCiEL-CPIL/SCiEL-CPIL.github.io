import time
import heapq
from fractions import Fraction

def construct():
    """
    Finds an Egyptian fraction expansion of 5/31 minimizing the largest denominator.
    Strategy: Dijkstra-like search over states (remaining_sum, min_denom, used_denoms).
    The priority is the current max denominator used (or 0 if none).
    We prune branches where the theoretical minimum possible next denominator 
    exceeds our current best solution's max denominator.
    """
    target = Fraction(5, 31)
    deadline = time.time() + 85  # Safety margin: 90s limit, 5s buffer
    
    # State: (max_denom_so_far, current_min_denom, remaining_fraction, used_denoms_tuple)
    # We want to minimize max_denom_so_far.
    # However, max_denom_so_far is non-decreasing. 
    # A better priority might be the 'current_min_denom' or just BFS by depth?
    # Actually, minimizing the max denominator is equivalent to finding a solution 
    # where all denominators are <= K. 
    # We can iterate K? No, that's slow.
    
    # Let's use a priority queue where the key is the maximum denominator used so far.
    # If we pop a state with max_denom D, and D is already >= best_found, we can stop?
    # No, because a state with larger max_denom might lead to a smaller final max? 
    # No, max_denom is monotonic. If we have used D, the final max is at least D.
    # So if we pop a state with max_denom D >= best_solution_max, we can prune it.
    
    # Initial state:
    # max_denom = 0
    # min_denom = 1
    # remaining = 5/31
    # used = ()
    
    # We need to be careful with the state space explosion.
    # used_denoms is a set. We can't easily hash large sets in a priority queue 
    # if we have many states. But the depth is limited (<= 256).
    # For 5/31, the depth will be small (likely < 10).
    
    # Optimization: 
    # The greedy algorithm gives a specific solution. 
    # Let's compute the greedy solution first to get an upper bound (best_max_den).
    
    def greedy_solution(target):
        r = target
        sol = []
        used = set()
        min_d = 1
        while r > 0:
            # Next denominator d such that 1/d <= r
            # d >= 1/r
            # d = ceil(1/r)
            d = (r.denominator + r.numerator - 1) // r.numerator
            if d < min_d:
                d = min_d
            while d in used:
                d += 1
            sol.append(d)
            used.add(d)
            r -= Fraction(1, d)
            min_d = d + 1
            if len(sol) > 256:
                break
        if r == 0:
            return sol
        return None

    greedy_sol = greedy_solution(target)
    if greedy_sol:
        best_max_den = max(greedy_sol)
        best_solution = greedy_sol
    else:
        best_max_den = float('inf')
        best_solution = None

    # BFS / Dijkstra
    # Queue items: (max_denom, min_denom, remaining, used_tuple)
    # We use a heap.
    # To avoid revisiting states, we can cache (remaining, min_denom, used_tuple) -> best_max_den_seen.
    # But used_tuple is large. 
    # Alternative: Since depth is small, we can just do a DFS with pruning, but Dijkstra is safer for "minimize max".
    
    # Actually, a simpler and often more effective approach for "minimize max denominator" 
    # is to iterate the target maximum denominator K from a lower bound upwards, 
    # and check if a solution exists with all denominators <= K.
    # Lower bound for K: 
    # Sum of reciprocals of 1..K is approx ln(K). 
    # 5/31 is approx 0.16.
    # ln(K) >= 0.16 => K >= e^0.16 ~ 1.17. So K is small.
    # But we need distinct denominators.
    # The greedy solution gives us an upper bound K_max.
    # We can binary search or linear search K from lower_bound to K_max.
    
    # What is a good lower bound for K?
    # If we use n terms, the sum is at most H_K (harmonic number) if we pick the smallest available.
    # But we are constrained by distinctness.
    # Let's just use the Dijkstra approach, it's more general.
    
    # To make Dijkstra efficient, we note that the number of terms is small.
    # We can store the state as (remaining, min_denom, frozenset(used)).
    # The number of unique remaining fractions is limited by the divisors of the denominator of target.
    
    visited = {} # state -> best_max_denom_seen
    
    # Start state
    start_state = (target, 1, frozenset())
    start_key = (0, start_state) # (max_denom, state)
    
    heap = [(0, 1, target, frozenset())] # (max_denom, min_denom, remaining, used)
    
    # We want to find ANY solution. The first one popped from the heap 
    # with a valid completion is not necessarily the global optimum if we don't 
    # ensure we've checked all states with smaller max_denom.
    # However, since max_denom is the priority, the first time we pop a state 
    # that can be completed to a valid solution, is it the best?
    # Not necessarily. A state with max_denom=10 might lead to a final max=100.
    # A state with max_denom=11 might lead to a final max=11.
    # So we cannot stop at the first valid solution. We must continue until 
    # the heap's min max_denom >= best_solution_max.
    
    while heap:
        if time.time() > deadline:
            break
            
        cur_max, cur_min, cur_rem, cur_used = heapq.heappop(heap)
        
        # Prune if current max is already worse than best found
        if cur_max >= best_max_den:
            break
            
        # Check if we've seen this state with a better or equal max_denom
        state_key = (cur_rem, cur_min, cur_used)
        if state_key in visited:
            if visited[state_key] <= cur_max:
                continue
        else:
            visited[state_key] = cur_max
            
        if cur_rem == 0:
            # Found a solution
            # The denominators are in cur_used. 
            # Wait, cur_used is a frozenset. We need the list.
            # We didn't store the list in the heap. 
            # We can reconstruct it? No, we need the actual denominators.
            # Let's store the tuple of denominators in the heap instead of just max.
            # But that makes the heap entries large.
            # Alternative: Store the path?
            # Let's restart with a slightly different state representation:
            # Heap: (max_denom, len(used), max_denom, min_denom, remaining, used_tuple)
            # Ties broken by length to encourage shorter expansions? No, we minimize max.
            
            # Actually, if cur_rem == 0, we have a valid solution.
            # The denominators are exactly the elements in cur_used.
            # We need to return them.
            sol_list = list(cur_used)
            # Verify
            if sum(Fraction(1, d) for d in sol_list) == target:
                if len(sol_list) <= 256 and all(1 <= d <= 10000000 for d in sol_list):
                    best_max_den = cur_max
                    best_solution = sol_list
                    # We don't break immediately because there might be a solution 
                    # with a smaller max_denom? 
                    # No, the heap is sorted by max_denom. 
                    # If we popped this, all other items in the heap have max_denom >= cur_max.
                    # So this is the minimal possible max_denom for any solution 
                    # that passes through this state. 
                    # But could there be a solution with a smaller max_denom 
                    # that hasn't been generated yet?
                    # Yes, if it comes from a different branch.
                    # But the heap priority is the max_denom *so far*.
                    # If a solution has final max M, all its prefixes have max <= M.
                    # So the state representing the full solution will have max=M.
                    # Since we pop in increasing order of max, the first valid solution 
                    # we pop is indeed the one with the globally minimal max denominator.
                    break

        if cur_rem < 0:
            continue
            
        # Determine next denominator d
        # 1/d < cur_rem  => d > 1/cur_rem
        # d must be >= cur_min
        # d must not be in cur_used
        
        # Lower bound for d:
        min_d = max(cur_min, int(1 / cur_rem) + 1)
        
        # Upper bound for d:
        # If 1/d is too small, the remaining sum might not be reachable with distinct denominators?
        # Also, if d > best_max_den, we can prune.
        max_d = best_max_den - 1
        
        if min_d > max_d:
            continue
            
        # Iterate possible next denominators
        # To keep the search space small, we should not iterate all d up to max_d.
        # We can limit the number of candidates.
        # However, for small fractions, the number of valid d is small.
        # Let's iterate d from min_d to max_d.
        # If max_d is large (e.g. 10000), this loop is slow.
        # But best_max_den starts from the greedy solution.
        # Greedy for 5/31:
        # 1/7 = 0.142, rem = 5/31 - 1/7 = (35-31)/217 = 4/217
        # 1/55 = 0.018, rem = 4/217 - 1/55 = (220-217)/11935 = 3/11935
        # 1/4000 approx.
        # Greedy max is around 4000.
        # So max_d is around 4000.
        # Iterating 4000 values per state might be too slow if we have many states.
        
        # Optimization: Only try a few candidates for d.
        # 1. The smallest possible d (min_d)
        # 2. d such that 1/d is close to cur_rem (greedy step)
        # 3. Randomized? No, deterministic is better for reproducibility.
        
        # Let's try a limited set of d values.
        # Candidates:
        # - min_d
        # - min_d + 1
        # - ... up to min_d + 50?
        # - Also, if we are close to a 2-term solution, we can solve it exactly.
        
        # Let's add a specific check for 2-term completion.
        # If cur_rem = 1/a + 1/b, we can solve for a,b directly.
        # This avoids iterating d.
        
        def solve_two_terms(r):
            if r <= 0:
                return None
            p = r.numerator
            q = r.denominator
            N = q * q
            limit = int(N ** 0.5) + 1
            for x in range(1, limit + 1):
                if N % x != 0:
                    continue
                y = N // x
                if (x + q) % p != 0:
                    continue
                if (y + q) % p != 0:
                    continue
                a = (x + q) // p
                b = (y + q) // p
                if a == b:
                    continue
                if a < 1 or b < 1:
                    continue
                if a > b:
                    a, b = b, a
                # Check distinctness with used
                if a in cur_used or b in cur_used:
                    continue
                # Check min_denom
                if a < cur_min or b < cur_min:
                    continue
                # Check max_denom
                if a > best_max_den or b > best_max_den:
                    continue
                return (a, b)
            return None

        pair = solve_two_terms(cur_rem)
        if pair:
            a, b = pair
            new_used = cur_used | {a, b}
            new_max = max(cur_max, a, b)
            if new_max < best_max_den:
                # We found a complete solution
                sol_list = list(new_used)
                if sum(Fraction(1, d) for d in sol_list) == target:
                    if len(sol_list) <= 256 and all(1 <= d <= 10000000 for d in sol_list):
                        best_max_den = new_max
                        best_solution = sol_list
                        break # Since heap is sorted, this is optimal
            
        # If 2-term solution didn't work or wasn't found, try extending by 1 term
        # Limit the number of d's to explore to keep it fast
        # We'll explore d in [min_d, min(min_d + 20, max_d)]
        # And also the greedy choice d_greedy = ceil(1/cur_rem)
        
        d_candidates = set()
        d_candidates.add(min_d)
        d_candidates.add(min_d + 1)
        d_candidates.add(min_d + 2)
        d_candidates.add(min_d + 5)
        d_candidates.add(min_d + 10)
        d_candidates.add(min_d + 20)
        
        d_greedy = (cur_rem.denominator + cur_rem.numerator - 1) // cur_rem.numerator
        if d_greedy >= cur_min:
            d_candidates.add(d_greedy)
            d_candidates.add(d_greedy + 1)
            d_candidates.add(d_greedy - 1)
            
        for d in d_candidates:
            if d < min_d or d > max_d:
                continue
            if d in cur_used:
                continue
                
            new_rem = cur_rem - Fraction(1, d)
            if new_rem < 0:
                continue
                
            new_min = d + 1
            new_used = cur_used | {d}
            new_max = max(cur_max, d)
            
            # Prune if new_max >= best_max_den
            if new_max >= best_max_den:
                continue
                
            # Check if this state is worth exploring
            # Simple heuristic: if new_rem is small, it's likely solvable
            # We just push it to the heap
            heapq.heappush(heap, (new_max, new_min, new_rem, new_used))

    if best_solution is None:
        # Fallback to greedy if Dijkstra failed to find a better one or timed out
        if greedy_sol:
            return greedy_sol
        # If even greedy failed (shouldn't happen), return empty or error
        return []

    return best_solution
