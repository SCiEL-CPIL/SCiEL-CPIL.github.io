import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 128
    target_lo = 128
    target_hi = 256
    target_range = list(range(target_lo, target_hi + 1))
    num_targets = len(target_range)
    
    def calculate_counts(A):
        """Calculate the representation counts for all target sums given set A."""
        counts = [0] * (target_hi + 1)
        A_list = list(A)
        for i in range(len(A_list)):
            ai = A_list[i]
            # Include equal summands (2*ai) and pairs with j > i
            for j in range(i, len(A_list)):
                aj = A_list[j]
                s = ai + aj
                if target_lo <= s <= target_hi:
                    counts[s] += 1
        return counts

    def get_peak_and_uncovered(counts):
        """Get the maximum count (peak) and number of uncovered targets."""
        peak = 0
        uncovered = 0
        for s in target_range:
            c = counts[s]
            if c == 0:
                uncovered += 1
            elif c > peak:
                peak = c
        return peak, uncovered

    def greedy_minimax_build():
        """
        Deterministic greedy construction.
        1. Start with an empty set.
        2. At each step, pick x that minimizes the new peak count.
           Tie-breaker: Maximize coverage of currently zero-count targets.
           Tie-breaker: Minimize the sum of squares of new counts (variance reduction).
        """
        A = set()
        counts = [0] * (target_hi + 1)
        
        # Initial seed: Randomly pick a few elements to break symmetry?
        # Actually, pure greedy from empty is deterministic. 
        # Let's start with a random small set to allow multiple restarts.
        # But the prompt asks for a structural change. 
        # Let's do a "Greedy Minimax" that is deterministic given the start.
        # We will run multiple restarts with different random initial seeds.
        
        # For this specific function, we assume A is already populated with initial seeds.
        # We will implement the loop here.
        
        max_iter = 200
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            current_peak, current_uncovered = get_peak_and_uncovered(counts)
            
            best_x = None
            best_new_peak = float('inf')
            best_new_uncovered = -1
            best_var_penalty = float('inf')
            
            # Evaluate all candidates x in [0, 128] not in A
            for x in range(n + 1):
                if x in A:
                    continue
                
                # Calculate impact of adding x
                # New pairs: (x, z) for z in A, and (x, x)
                # We need to simulate the counts change.
                # To avoid copying the array, we calculate the max count that would result.
                
                # The new counts for sums s = x + z (z in A) and s = 2x
                # will be counts[s] + 1.
                # All other counts remain the same.
                
                max_new_count = current_peak
                new_uncovered = current_uncovered
                sum_sq_new = 0
                # We need sum of squares for tie-breaking? 
                # It's O(N) to compute from scratch. 
                # Let's just use max_new_count and new_uncovered.
                # If we want variance, we can compute it for the final candidate only.
                
                # Check coverage of zeros
                # If counts[s] was 0 and becomes 1, uncovered decreases.
                
                # Iterate through potential sums created by x
                # Sums: 2*x and x+z for all z in A
                seen_sums = set()
                
                s2 = 2 * x
                if target_lo <= s2 <= target_hi:
                    seen_sums.add(s2)
                    
                for z in A:
                    sz = x + z
                    if target_lo <= sz <= target_hi:
                        seen_sums.add(sz)
                
                # Calculate new peak and uncovered
                local_max = current_peak
                local_uncovered = current_uncovered
                
                for s in seen_sums:
                    old_c = counts[s]
                    new_c = old_c + 1
                    if new_c > local_max:
                        local_max = new_c
                    if old_c == 0:
                        local_uncovered -= 1
                
                # Heuristic: We want to minimize local_max.
                # If local_max < best_new_peak, it's better.
                # If local_max == best_new_peak, we want to maximize coverage (minimize local_uncovered).
                # If both same, we might want to prefer x that spreads load?
                # For now, let's just use (local_max, local_uncovered).
                
                is_better = False
                if local_max < best_new_peak:
                    is_better = True
                elif local_max == best_new_peak:
                    if local_uncovered > best_new_uncovered:
                        is_better = True
                    # Tie-breaker: If uncovered is same, prefer x that doesn't increase the sum of counts?
                    # Or just random to break symmetry? No, deterministic is better for reproducibility in restarts.
                    # Let's add a tiny penalty for high total load?
                    # Actually, if local_max is the same, the "shape" matters.
                    # Let's stick to (max, uncovered).
                
                if is_better:
                    best_new_peak = local_max
                    best_new_uncovered = local_uncovered
                    best_x = x
            
            if best_x is None:
                break
                
            # Apply best_x
            A.add(best_x)
            
            # Update counts
            s2 = 2 * best_x
            if target_lo <= s2 <= target_hi:
                counts[s2] += 1
            for z in A:
                if z == best_x:
                    continue
                sz = best_x + z
                if target_lo <= sz <= target_hi:
                    counts[sz] += 1
            
            # Early stopping if we have a very good solution?
            # If peak is 1, we can't do better (unless we can cover all with 1s, which is a perfect factorization).
            # If peak is small and uncovered is 0, we might stop.
            # But we want to minimize peak. If we are at peak 2, can we get to 1? Unlikely for this density.
            # Let's run for a fixed number of steps or until no improvement.
            
            if current_uncovered == 0 and current_peak <= 2:
                # If we are already very good, stop adding to avoid increasing peak?
                # Adding elements increases counts. 
                # Wait, the greedy adds elements. 
                # If we have full coverage, adding more elements will ONLY increase counts.
                # So if uncovered == 0, we should STOP adding.
                break

        return A, counts

    def local_search_refine(A, counts, time_limit):
        """
        Refine the set by removing elements that don't hurt coverage and reduce peak,
        or swapping elements to reduce peak.
        Since the objective is to minimize the MAX count, and adding elements increases counts,
        the optimal set is likely minimal in size for a given peak.
        
        Strategy:
        1. Try to remove elements. If removal doesn't create 0s in counts, and reduces peak, accept.
        2. Try to swap elements if it reduces peak without creating 0s.
        """
        A = set(A)
        counts = counts[:]
        
        curr_peak, curr_uncovered = get_peak_and_uncovered(counts)
        
        if curr_uncovered > 0:
            # We need to ensure coverage first. 
            # The greedy build should have handled this, but if not, 
            # this local search might break coverage. 
            # For safety, we only do removals if uncovered == 0.
            # If uncovered > 0, we should probably return as is or try to add?
            # The greedy build stops when uncovered == 0. 
            # So we assume uncovered == 0 here.
            pass

        best_A = set(A)
        best_counts = counts[:]
        best_peak = curr_peak
        best_uncovered = curr_uncovered
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try Removals
            # We want to remove x if:
            # - After removal, no target sum has count 0.
            # - After removal, the peak is lower (or equal, but we prefer smaller set? No, objective is peak).
            # Actually, if peak is same, removing an element reduces the "pressure" on future swaps?
            # Let's just look for peak reduction.
            
            candidates = list(A)
            random.shuffle(candidates)
            
            for x in candidates:
                # Simulate removal of x
                # New counts: counts[s] - 1 for s in {x+z, 2x}
                
                # Check if any count becomes 0
                valid = True
                new_peak = 0
                new_uncovered = 0
                
                # We need to calculate the new peak.
                # Instead of full recalculation, we can track the max.
                # But removals can only decrease counts.
                # So the new peak is either the old peak (if the element with old peak is not involved in the max count sums) 
                # or lower.
                
                # Let's just recalculate peak and uncovered for simplicity, as |A| is small (~50-100).
                # O(|A|^2) per removal check is too slow if we do it for every x.
                # |A| ~ 100. |A| removals * 100^2 ops = 10^6 ops. 
                # In Python, 10^6 ops takes ~0.1s. 
                # If we do this in a loop, it might be slow.
                
                # Optimization: 
                # Only remove x if x is "redundant".
                # x is redundant if for every sum s=x+z, there is another pair (u,v) != (x,z) such that u+v=s.
                # This is hard to check quickly.
                
                # Let's use a faster heuristic:
                # Only consider removing x if the current peak is > 1.
                # And only if x is involved in the sums that have the peak count.
                
                # For now, let's just do a full check but limit the number of removal attempts per iteration.
                
                # Simulate
                temp_counts = counts[:]
                s2 = 2 * x
                if target_lo <= s2 <= target_hi:
                    temp_counts[s2] -= 1
                    if temp_counts[s2] < 0: # Should not happen
                        valid = False
                for z in A:
                    if z == x: continue
                    sz = x + z
                    if target_lo <= sz <= target_hi:
                        temp_counts[sz] -= 1
                        if temp_counts[sz] < 0:
                            valid = False
                
                if not valid:
                    continue
                
                # Check uncovered
                new_unc = 0
                new_pk = 0
                for s in target_range:
                    c = temp_counts[s]
                    if c == 0:
                        new_unc += 1
                    else:
                        if c > new_pk:
                            new_pk = c
                
                if new_unc > 0:
                    continue # Cannot remove if it breaks coverage
                
                if new_pk < best_peak:
                    # Accept removal
                    A.remove(x)
                    counts = temp_counts
                    best_peak = new_pk
                    best_uncovered = new_unc
                    best_A = set(A)
                    best_counts = counts[:]
                    improved = True
                    break # Restart loop with new state
                
            if improved:
                continue
            
            # 2. Try Swaps (if no removal improved)
            # Swap x in A with y not in A.
            # This is more complex. 
            # Given the time limit, if removals don't work, we might stop.
            # The greedy build is the main driver.
            
            # If we haven't improved in a while, break?
            # For now, just break if no removal works.
            break
            
        return best_A, best_counts

    best_overall_A = None
    best_overall_score = float('inf') # score = peak * 1000 + uncovered
    
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # 1. Initialize with random seeds
        # The number of elements needed is roughly sqrt(2 * 128) ~ 16 for perfect packing?
        # But we have constraints. 
        # Let's start with a random set of size 20-40.
        init_size = random.randint(20, 40)
        A_init = set(random.sample(range(n + 1), init_size))
        
        # 2. Greedy Minimax Build
        A_greedy, counts_greedy = greedy_minimax_build()
        
        # Check if coverage is complete. If not, the greedy might have failed to cover all.
        # The greedy stops when uncovered == 0. 
        # If it doesn't reach 0, we have a problem.
        # Let's check.
        peak_g, unc_g = get_peak_and_uncovered(counts_greedy)
        
        if unc_g > 0:
            # Try to fix by adding more? 
            # The greedy should have added until covered. 
            # If it didn't, it means no single element could cover the remaining zeros without increasing peak?
            # No, greedy picks the BEST x. If best_x is None, it means no x available? 
            # Or it ran out of iterations.
            # Let's just force add random elements if uncovered > 0.
            while unc_g > 0 and time.time() < deadline - 5.0:
                x = random.randint(0, n)
                if x in A_greedy:
                    continue
                A_greedy.add(x)
                # Update counts
                s2 = 2 * x
                if target_lo <= s2 <= target_hi:
                    counts_greedy[s2] += 1
                for z in A_greedy:
                    if z == x: continue
                    sz = x + z
                    if target_lo <= sz <= target_hi:
                        counts_greedy[sz] += 1
                peak_g, unc_g = get_peak_and_uncovered(counts_greedy)
        
        # 3. Local Search Refine (Removals)
        remaining_time = deadline - time.time()
        sa_time = max(1.0, remaining_time * 0.5)
        A_final, counts_final = local_search_refine(A_greedy, counts_greedy, time.time() + sa_time)
        
        peak_f, unc_f = get_peak_and_uncovered(counts_final)
        
        # Score: Primary is Peak. Secondary is Uncovered (should be 0).
        # If uncovered > 0, it's invalid effectively, so high penalty.
        score = peak_f * 10000 + unc_f
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_A = A_final
            
        # Early stop if we find a perfect solution (Peak 1 is theoretically impossible for this range/size? 
        # If peak 2 is found, it's very good.)
        if peak_f <= 2 and unc_f == 0:
             break

    if best_overall_A is None:
        # Fallback: Return a dense set to ensure coverage, even if peak is high.
        return list(range(64, 129))
        
    return list(best_overall_A)

# Keep the required interface
# The current program has a weird structure with _construct_impl. 
# I will just define construct() directly.

# Wait, the prompt says "Keep the function names and signatures of the seed program."
# The seed program has:
# def construct():
#     ...
# def construct(): # duplicate? No, it was a mess.
# The final definition in the seed was:
# def construct():
#     ...
#     return best_overall_A

# So I just need to provide a single `construct()` function.

if __name__ == "__main__":
    # Quick self-test
    A = construct()
    print(f"Constructed A of size {len(A)}")
    # Verify
    counts = [0] * 257
    A_list = list(A)
    for i in range(len(A_list)):
        for j in range(i, len(A_list)):
            s = A_list[i] + A_list[j]
            if 128 <= s <= 256:
                counts[s] += 1
    max_c = 0
    unc = 0
    for s in range(128, 257):
        if counts[s] == 0:
            unc += 1
        else:
            if counts[s] > max_c:
                max_c = counts[s]
    print(f"Max Count: {max_c}, Uncovered: {unc}")
