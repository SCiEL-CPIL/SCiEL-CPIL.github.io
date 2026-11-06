import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 128
    target_lo = 128
    target_hi = 256
    target_range = list(range(target_lo, target_hi + 1))
    num_targets = len(target_range)
    
    def calculate_counts(A):
        """
        Calculate representation counts for all sums in [target_lo, target_hi].
        A is a set of integers.
        Returns a list of counts indexed by sum value (offset by 0, but we only care about target range).
        """
        counts = [0] * (target_hi + 1)
        A_list = list(A)
        for i in range(len(A_list)):
            ai = A_list[i]
            for j in range(i, len(A_list)):
                aj = A_list[j]
                s = ai + aj
                if target_lo <= s <= target_hi:
                    counts[s] += 1
        return counts

    def get_peak_and_uncovered(counts):
        peak = 0
        uncovered = 0
        for s in target_range:
            c = counts[s]
            if c == 0:
                uncovered += 1
            elif c > peak:
                peak = c
        return uncovered, peak

    def greedy_build():
        """
        Builds a valid initial set A using a greedy strategy that prioritizes coverage
        and then tries to keep counts balanced.
        """
        A = set()
        counts = [0] * (target_hi + 1)
        
        # We need enough elements to cover 129 sums.
        # Each element x contributes to sums x+y for y in A.
        # If we have k elements, we have ~k^2/2 sums.
        # To cover 129 sums with low multiplicity, we need k such that k^2/2 >= 129 * peak.
        # If peak=1, k >= sqrt(258) ~ 16.
        # If peak=2, k >= sqrt(516) ~ 23.
        # Let's aim for a set of size ~30-40 to allow for balancing.
        
        # Start with a random subset to break symmetry
        # Actually, a structured start might be better.
        # Let's try adding elements that cover the most new sums.
        
        # Initial random seed
        for _ in range(10):
            x = random.randint(0, n)
            A.add(x)
            
        counts = calculate_counts(A)
        
        max_iter = 500
        for _ in range(max_iter):
            if time.time() > deadline - 10:
                break
                
            # Check if all covered
            uncovered, peak = get_peak_and_uncovered(counts)
            if uncovered == 0:
                # If covered, we might stop adding to avoid increasing peak?
                # Or we can continue adding to "spread" the load if we have room?
                # No, adding always increases total counts, so it tends to increase peak or keep it same.
                # So stop adding when coverage is complete.
                break
            
            best_x = -1
            best_gain = -1
            best_risk = float('inf')
            
            # Candidate pool: all x in [0, n] not in A
            # We want x that covers many 0s and adds to sums with low current count.
            
            for x in range(n + 1):
                if x in A:
                    continue
                
                new_zeros = 0
                max_new_count = 0
                total_increase = 0
                
                # Simulate adding x
                # Sums involving x: x+y for y in A, and 2x
                for y in A:
                    s = x + y
                    if target_lo <= s <= target_hi:
                        c = counts[s]
                        if c == 0:
                            new_zeros += 1
                        if c + 1 > max_new_count:
                            max_new_count = c + 1
                        total_increase += 1
                
                s2 = 2 * x
                if target_lo <= s2 <= target_hi:
                    c = counts[s2]
                    if c == 0:
                        new_zeros += 1
                    if c + 1 > max_new_count:
                        max_new_count = c + 1
                    total_increase += 1
                
                # Score: Maximize new_zeros, minimize max_new_count
                # If max_new_count is very high, avoid.
                # Simple heuristic: new_zeros * 100 - max_new_count
                score = new_zeros * 100 - max_new_count
                
                if score > best_gain or (score == best_gain and max_new_count < best_risk):
                    best_gain = score
                    best_x = x
                    best_risk = max_new_count
            
            if best_x == -1:
                break
                
            A.add(best_x)
            # Update counts
            for y in A:
                if y == best_x:
                    continue
                s = best_x + y
                if target_lo <= s <= target_hi:
                    counts[s] += 1
            s2 = 2 * best_x
            if target_lo <= s2 <= target_hi:
                counts[s2] += 1
        
        return A, counts

    def minimax_polish(A, counts, time_limit):
        """
        Iteratively swap elements to reduce the peak representation count.
        Move: Remove x from A, Add y to A.
        Accept if new_peak < old_peak or (new_peak == old_peak and new_uncovered < old_uncovered).
        If equal, accept with some probability to escape local minima? 
        Actually, for minimax, we strictly want to lower the peak.
        If we can't lower the peak, we try to lower the variance or number of elements at peak.
        """
        A = set(A)
        counts = counts[:]
        
        curr_unc, curr_peak = get_peak_and_uncovered(counts)
        best_A = A.copy()
        best_counts = counts[:]
        best_peak = curr_peak
        best_unc = curr_unc
        
        # We will perform a series of swaps.
        # To be efficient, we precompute the "impact" of removing x and adding y.
        # But since A is small (~30-40), O(|A|^2) per step is fine.
        
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < time_limit:
            improved = False
            
            # Iterate through all pairs (x in A, y not in A)
            # To speed up, we can sample. But |A| is small.
            # Let's iterate all x in A.
            # For each x, find best y not in A.
            
            # Optimization: Precompute which y's are "safe" (don't increase peak)
            
            for x in list(A):
                if time.time() > time_limit:
                    break
                
                # Temporarily remove x
                A.remove(x)
                # Update counts: decrement sums involving x
                for y in A:
                    s = x + y
                    if target_lo <= s <= target_hi:
                        counts[s] -= 1
                s2 = 2 * x
                if target_lo <= s2 <= target_hi:
                    counts[s2] -= 1
                    
                # Now try adding each y not in A
                best_y = -1
                best_new_peak = curr_peak # We want to be <= curr_peak
                best_new_unc = curr_unc
                best_new_var = 0 # Not calculating var for speed, just peak/unc
                
                # We only care about y that doesn't break coverage (unc stays same or decreases)
                # and keeps peak low.
                
                # To save time, we can just try a few random y's or all y's.
                # Let's try all y in [0, n] not in A.
                # 129 * |A| ~ 129 * 40 = 5000 ops per x.
                # 40 * 5000 = 200,000 ops per full pass.
                # 100 passes = 20M ops. Doable in 1s? Maybe tight in Python.
                # Let's sample y's if A is large.
                
                # Actually, let's just do a random subset of y's to speed up search.
                # Or, better: identify "critical" sums.
                
                # Let's try all y for now, with a time check.
                
                # We need to check if adding y creates a new peak > curr_peak.
                # And if it covers any uncovered sums.
                
                # Precompute zero_sums and peak_sums?
                # zero_sums = [s for s in target_range if counts[s] == 0]
                # This changes as we modify counts.
                
                # Let's just iterate y.
                for y in range(n + 1):
                    if y in A:
                        continue
                    
                    # Simulate adding y
                    # Check if any sum x+y or 2y exceeds curr_peak
                    # And count how many zeros it covers.
                    
                    valid = True
                    new_unc = curr_unc
                    max_c = 0
                    
                    # We need to know counts AFTER removing x.
                    # counts is currently in state "x removed".
                    
                    for z in A:
                        s = y + z
                        if target_lo <= s <= target_hi:
                            c = counts[s] + 1
                            if c > curr_peak:
                                valid = False
                                break
                            if c > max_c:
                                max_c = c
                            if counts[s] == 0:
                                new_unc -= 1
                    if not valid:
                        continue
                    
                    s2 = 2 * y
                    if target_lo <= s2 <= target_hi:
                        c = counts[s2] + 1
                        if c > curr_peak:
                            valid = False
                        else:
                            if c > max_c:
                                max_c = c
                            if counts[s2] == 0:
                                new_unc -= 1
                            
                    if not valid:
                        continue
                    
                    # If valid, compare with best_y
                    # We prefer lower new_unc, then lower max_c (to flatten)
                    if best_y == -1:
                        best_y = y
                        best_new_unc = new_unc
                        best_new_var = max_c
                    else:
                        if new_unc < best_new_unc:
                            best_y = y
                            best_new_unc = new_unc
                            best_new_var = max_c
                        elif new_unc == best_new_unc and max_c < best_new_var:
                            best_y = y
                            best_new_var = max_c
                
                # Rollback x removal
                A.add(x)
                for y in A:
                    if y == x:
                        continue
                    s = x + y
                    if target_lo <= s <= target_hi:
                        counts[s] += 1
                s2 = 2 * x
                if target_lo <= s2 <= target_hi:
                    counts[s2] += 1
                
                if best_y != -1:
                    # Apply swap: Remove x, Add best_y
                    # We already removed x in the loop above, but we rolled it back.
                    # So we need to do the swap now.
                    
                    # Remove x
                    A.remove(x)
                    for z in A:
                        s = x + z
                        if target_lo <= s <= target_hi:
                            counts[s] -= 1
                    s2 = 2 * x
                    if target_lo <= s2 <= target_hi:
                        counts[s2] -= 1
                        
                    # Add best_y
                    A.add(best_y)
                    for z in A:
                        if z == best_y:
                            continue
                        s = best_y + z
                        if target_lo <= s <= target_hi:
                            counts[s] += 1
                    s2 = 2 * best_y
                    if target_lo <= s2 <= target_hi:
                        counts[s2] += 1
                        
                    new_unc, new_peak = get_peak_and_uncovered(counts)
                    
                    # Update bests
                    if new_peak < best_peak or (new_peak == best_peak and new_unc < best_unc):
                        best_peak = new_peak
                        best_unc = new_unc
                        best_A = A.copy()
                        best_counts = counts[:]
                        improved = True
                    elif new_peak < curr_peak:
                        # Even if not better than global best, it's better than current state
                        # We accept it to move forward
                        pass # curr_peak will be updated below
                    
                    curr_unc, curr_peak = get_peak_and_uncovered(counts)
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            else:
                no_improve_count = 0
                
        return best_A, best_peak, best_unc

    best_A = None
    best_peak = float('inf')
    best_unc = float('inf')
    
    # Run multiple restarts
    num_restarts = 20
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        A_init, counts_init = greedy_build()
        
        # Allocate time for polishing
        remaining = deadline - time.time()
        # Give 20% of remaining time to each restart, min 1s
        polish_time = max(1.0, remaining * 0.2)
        
        A_pol, peak_pol, unc_pol = minimax_polish(A_init, counts_init, time.time() + polish_time)
        
        if peak_pol < best_peak or (peak_pol == best_peak and unc_pol < best_unc):
            best_peak = peak_pol
            best_unc = unc_pol
            best_A = A_pol
            
        # If we reached peak 1, we can stop? 
        # Peak 1 means a Sidon set / B2 sequence covering the range.
        # This is very hard. Peak 2 is more realistic.
        if best_peak <= 2:
            break
            
    if best_A is None:
        # Fallback
        return list(range(64, 129))
        
    return list(best_A)

# Ensure the function name matches the contract
# The contract says: construct() -> a nonempty subset A of [0,n=128]
# The current program has:
# def construct(): ...
# def _construct_impl(): pass
# def construct(): ... (redefined)
#
# My code defines construct() once. This is correct.

if __name__ == "__main__":
    # Simple test
    A = construct()
    counts = [0]*257
    A_list = list(A)
    for i in range(len(A_list)):
        for j in range(i, len(A_list)):
            s = A_list[i] + A_list[j]
            if 128 <= s <= 256:
                counts[s] += 1
    peak = max(counts[128:257])
    unc = sum(1 for c in counts[128:257] if c == 0)
    print(f"Peak: {peak}, Uncovered: {unc}, Size: {len(A)}")
