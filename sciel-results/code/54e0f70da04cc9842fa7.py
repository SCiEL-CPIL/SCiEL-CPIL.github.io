import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = 128
    target_lo = 128
    target_hi = 256
    target_range = list(range(target_lo, target_hi + 1))
    num_targets = len(target_range)
    
    # Precompute valid pairs for each sum? Or just iterate A.
    # A is small (~50). Iterating A for updates is O(|A|).
    
    def get_score(A_set, counts):
        """
        Returns (uncovered, peak, variance)
        We use a weighted score for SA: 
        score = uncovered * 10000 + peak * 100 + variance
        """
        uncovered = 0
        peak = 0
        total_sq = 0
        total = 0
        for s in target_range:
            c = counts[s]
            if c == 0:
                uncovered += 1
            else:
                if c > peak:
                    peak = c
                total_sq += c * c
                total += c
        
        # Variance term helps flatten the distribution
        # If total is 0, variance is 0
        var = 0.0
        if total > 0:
            mean = total / num_targets
            var = (total_sq / num_targets) - (mean * mean)
            
        return uncovered, peak, var

    def calculate_initial_counts(A):
        counts = [0] * 257
        # A is a list
        for i in range(len(A)):
            ai = A[i]
            for j in range(i, len(A)):
                aj = A[j]
                s = ai + aj
                if target_lo <= s <= target_hi:
                    counts[s] += 1
        return counts

    def greedy_build():
        """
        Builds A by greedily adding elements that maximize coverage 
        while keeping counts below a soft cap.
        """
        A = []
        A_set = set()
        counts = [0] * 257
        
        # Start with a few random seeds to break symmetry? 
        # Or start empty.
        # Let's try starting with a random subset of size 10 to get a base.
        initial_size = 10
        for _ in range(initial_size):
            x = random.randint(0, n)
            if x not in A_set:
                A.append(x)
                A_set.add(x)
        
        # Update counts for initial set
        counts = calculate_initial_counts(A)
        
        # Iteratively add elements
        # We want to fill gaps.
        # Score for adding x: 
        # 1. Increases coverage of 0s (high priority)
        # 2. Doesn't create new peaks (medium priority)
        
        max_iter = 1000
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            best_x = -1
            best_score = -float('inf')
            
            # Candidate pool: random subset of [0, n] not in A to save time? 
            # Or just iterate all 129. 129 * |A| ops is ~ 129 * 50 = 6450. Fast.
            for x in range(n + 1):
                if x in A_set:
                    continue
                
                # Simulate adding x
                # New counts would be counts + delta
                # We don't want to copy counts array (size 257) 129 times per step.
                # Instead, calculate the "gain" analytically.
                
                # Gains:
                # For each z in A, sum s = x+z.
                # If counts[s] == 0, we gain coverage.
                # If counts[s] >= current_peak, we might increase peak.
                
                gain_coverage = 0
                risk_peak = 0
                current_peak = 0
                # We need current peak. Calculating it every time is O(129).
                # Let's maintain current_peak and current_uncovered outside loop?
                # No, it changes.
                
                # Optimization: Precompute which sums are 0.
                # zero_sums = {s for s in target_range if counts[s] == 0}
                
                # For x:
                # Sums formed: {x+z for z in A} U {2x}
                
                # Let's just estimate.
                # If we add x, how many new 0s do we cover?
                new_zeros_covered = 0
                max_count_after = 0
                for z in A:
                    s = x + z
                    if target_lo <= s <= target_hi:
                        c = counts[s]
                        if c == 0:
                            new_zeros_covered += 1
                        if c + 1 > max_count_after:
                            max_count_after = c + 1
                
                s2 = 2 * x
                if target_lo <= s2 <= target_hi:
                    c = counts[s2]
                    if c == 0:
                        new_zeros_covered += 1
                    if c + 1 > max_count_after:
                        max_count_after = c + 1
                
                # Score: High for new coverage, penalize if max_count_after is high
                # We want to keep counts low.
                # Let's target a max count of, say, 10.
                penalty = 0
                if max_count_after > 10:
                    penalty = (max_count_after - 10) * 100
                
                score = new_zeros_covered * 1000 - penalty
                
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x == -1:
                break
            
            # Apply best_x
            A.append(best_x)
            A_set.add(best_x)
            for z in A:
                if z == best_x:
                    continue
                s = best_x + z
                if target_lo <= s <= target_hi:
                    counts[s] += 1
            s2 = 2 * best_x
            if target_lo <= s2 <= target_hi:
                counts[s2] += 1
            
            # Check if we covered all
            # Quick check: if all counts > 0, stop adding? 
            # Maybe we can add more to reduce peak? No, adding increases counts.
            # So stop adding when coverage is complete.
            all_covered = True
            for s in target_range:
                if counts[s] == 0:
                    all_covered = False
                    break
            
            if all_covered:
                # We have full coverage. Now we might want to stop adding.
                # Further additions will only increase peak/variance.
                break
                
        return A, counts

    def simulated_annealing(A, counts, time_limit):
        """
        Polish the set using SA.
        Moves: Swap x in A with y not in A.
        """
        A = list(A)
        A_set = set(A)
        counts = counts[:]
        
        curr_unc, curr_peak, curr_var = get_score(A_set, counts)
        curr_score = curr_unc * 10000 + curr_peak * 100 + curr_var
        
        best_A = A[:]
        best_counts = counts[:]
        best_score = curr_score
        
        # SA Parameters
        T = 1000.0
        cooling = 0.999
        min_T = 1.0
        
        while time.time() < time_limit:
            if T < min_T:
                break
                
            # Generate move: Swap
            x = random.choice(A)
            y = random.randint(0, n)
            attempts = 0
            while y in A_set and attempts < 5:
                y = random.randint(0, n)
                attempts += 1
            if y in A_set:
                continue
            
            # Apply Swap
            # Remove x
            A.remove(x)
            A_set.remove(x)
            for z in A:
                s = x + z
                if target_lo <= s <= target_hi:
                    counts[s] -= 1
            s2 = 2 * x
            if target_lo <= s2 <= target_hi:
                counts[s2] -= 1
                
            # Add y
            A.append(y)
            A_set.add(y)
            for z in A:
                if z == y:
                    continue
                s = y + z
                if target_lo <= s <= target_hi:
                    counts[s] += 1
            s2 = 2 * y
            if target_lo <= s2 <= target_hi:
                counts[s2] += 1
                
            new_unc, new_peak, new_var = get_score(A_set, counts)
            new_score = new_unc * 10000 + new_peak * 100 + new_var
            
            delta = new_score - curr_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                # Accept
                curr_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = A[:]
                    best_counts = counts[:]
            else:
                # Reject: Rollback
                # Add x back
                A.remove(y)
                A_set.remove(y)
                for z in A:
                    if z == x: # x is not in A yet
                        pass
                    s = y + z
                    if target_lo <= s <= target_hi:
                        counts[s] -= 1
                s2 = 2 * y
                if target_lo <= s2 <= target_hi:
                    counts[s2] -= 1
                    
                A.append(x)
                A_set.add(x)
                for z in A:
                    if z == x:
                        continue
                    s = x + z
                    if target_lo <= s <= target_hi:
                        counts[s] += 1
                s2 = 2 * x
                if target_lo <= s2 <= target_hi:
                    counts[s2] += 1
                    
            T *= cooling
            
        return best_A, best_score

    best_overall_A = None
    best_overall_score = float('inf')
    
    # Run multiple restarts with different random seeds for the initial greedy part
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() > deadline - 10: # Leave 10s for final polish or return
            break
            
        # 1. Greedy Build
        A_init, counts_init = greedy_build()
        
        # 2. SA Polish
        # Allocate time based on remaining budget
        remaining = deadline - time.time()
        # Give SA a chunk of time, e.g., 50% of remaining, but at least 1s
        sa_time = max(1.0, remaining * 0.5)
        
        A_polished, score_polished = simulated_annealing(A_init, counts_init, time.time() + sa_time)
        
        if score_polished < best_overall_score:
            best_overall_score = score_polished
            best_overall_A = A_polished
            
        # If we get a very good score (low uncovered, low peak), we can stop
        # Score = unc*10000 + peak*100 + var
        # If unc=0 and peak <= 5, score < 500 + var.
        if best_overall_score < 1000: # unc=0, peak<=10, var small
             # We can break early if we are very confident
             if best_overall_score < 500: # unc=0, peak<=5
                 break

    if best_overall_A is None:
        # Fallback
        return list(range(64, 129))
        
    return best_overall_A

# Keep the required interface
def construct():
    return _construct_impl()

# Alias the real logic
def _construct_impl():
    # Re-define the logic inside to avoid global state issues if called multiple times?
    # No, standard Python function scope is fine.
    # But the code above defined construct. Let's rename the inner logic.
    pass

# Wait, I defined construct() twice. Let's fix the structure.

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 128
    target_lo = 128
    target_hi = 256
    target_range = list(range(target_lo, target_hi + 1))
    num_targets = len(target_range)
    
    def get_score(A_set, counts):
        uncovered = 0
        peak = 0
        total_sq = 0
        total = 0
        for s in target_range:
            c = counts[s]
            if c == 0:
                uncovered += 1
            else:
                if c > peak:
                    peak = c
                total_sq += c * c
                total += c
        var = 0.0
        if total > 0:
            mean = total / num_targets
            var = (total_sq / num_targets) - (mean * mean)
        return uncovered, peak, var

    def calculate_initial_counts(A):
        counts = [0] * 257
        for i in range(len(A)):
            ai = A[i]
            for j in range(i, len(A)):
                aj = A[j]
                s = ai + aj
                if target_lo <= s <= target_hi:
                    counts[s] += 1
        return counts

    def greedy_build():
        A = []
        A_set = set()
        counts = [0] * 257
        
        initial_size = 10
        for _ in range(initial_size):
            x = random.randint(0, n)
            if x not in A_set:
                A.append(x)
                A_set.add(x)
        
        counts = calculate_initial_counts(A)
        
        max_iter = 2000
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            
            best_x = -1
            best_score = -float('inf')
            
            for x in range(n + 1):
                if x in A_set:
                    continue
                
                new_zeros_covered = 0
                max_count_after = 0
                for z in A:
                    s = x + z
                    if target_lo <= s <= target_hi:
                        c = counts[s]
                        if c == 0:
                            new_zeros_covered += 1
                        if c + 1 > max_count_after:
                            max_count_after = c + 1
                
                s2 = 2 * x
                if target_lo <= s2 <= target_hi:
                    c = counts[s2]
                    if c == 0:
                        new_zeros_covered += 1
                    if c + 1 > max_count_after:
                        max_count_after = c + 1
                
                penalty = 0
                if max_count_after > 10:
                    penalty = (max_count_after - 10) * 100
                
                score = new_zeros_covered * 1000 - penalty
                
                if score > best_score:
                    best_score = score
                    best_x = x
            
            if best_x == -1:
                break
            
            A.append(best_x)
            A_set.add(best_x)
            for z in A:
                if z == best_x:
                    continue
                s = best_x + z
                if target_lo <= s <= target_hi:
                    counts[s] += 1
            s2 = 2 * best_x
            if target_lo <= s2 <= target_hi:
                counts[s2] += 1
            
            all_covered = True
            for s in target_range:
                if counts[s] == 0:
                    all_covered = False
                    break
            
            if all_covered:
                break
                
        return A, counts

    def simulated_annealing(A, counts, time_limit):
        A = list(A)
        A_set = set(A)
        counts = counts[:]
        
        curr_unc, curr_peak, curr_var = get_score(A_set, counts)
        curr_score = curr_unc * 10000 + curr_peak * 100 + curr_var
        
        best_A = A[:]
        best_counts = counts[:]
        best_score = curr_score
        
        T = 1000.0
        cooling = 0.999
        min_T = 1.0
        
        while time.time() < time_limit:
            if T < min_T:
                break
                
            x = random.choice(A)
            y = random.randint(0, n)
            attempts = 0
            while y in A_set and attempts < 5:
                y = random.randint(0, n)
                attempts += 1
            if y in A_set:
                continue
            
            # Apply Swap
            A.remove(x)
            A_set.remove(x)
            for z in A:
                s = x + z
                if target_lo <= s <= target_hi:
                    counts[s] -= 1
            s2 = 2 * x
            if target_lo <= s2 <= target_hi:
                counts[s2] -= 1
                
            A.append(y)
            A_set.add(y)
            for z in A:
                if z == y:
                    continue
                s = y + z
                if target_lo <= s <= target_hi:
                    counts[s] += 1
            s2 = 2 * y
            if target_lo <= s2 <= target_hi:
                counts[s2] += 1
                
            new_unc, new_peak, new_var = get_score(A_set, counts)
            new_score = new_unc * 10000 + new_peak * 100 + new_var
            
            delta = new_score - curr_score
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                curr_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = A[:]
                    best_counts = counts[:]
            else:
                # Rollback
                A.remove(y)
                A_set.remove(y)
                for z in A:
                    s = y + z
                    if target_lo <= s <= target_hi:
                        counts[s] -= 1
                s2 = 2 * y
                if target_lo <= s2 <= target_hi:
                    counts[s2] -= 1
                    
                A.append(x)
                A_set.add(x)
                for z in A:
                    if z == x:
                        continue
                    s = x + z
                    if target_lo <= s <= target_hi:
                        counts[s] += 1
                s2 = 2 * x
                if target_lo <= s2 <= target_hi:
                    counts[s2] += 1
                    
            T *= cooling
            
        return best_A, best_score

    best_overall_A = None
    best_overall_score = float('inf')
    
    num_restarts = 100
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        A_init, counts_init = greedy_build()
        
        remaining = deadline - time.time()
        sa_time = max(0.5, remaining * 0.4)
        
        A_polished, score_polished = simulated_annealing(A_init, counts_init, time.time() + sa_time)
        
        if score_polished < best_overall_score:
            best_overall_score = score_polished
            best_overall_A = A_polished
            
        if best_overall_score < 500:
            break

    if best_overall_A is None:
        return list(range(64, 129))
        
    return best_overall_A
