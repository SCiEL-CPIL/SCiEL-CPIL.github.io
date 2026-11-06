import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = 128
    target_lo = 128
    target_hi = 256
    target_range = range(target_lo, target_hi + 1)
    
    best_A = None
    best_peak = float('inf')
    best_uncovered = float('inf')
    
    def get_score(counts):
        uncovered = 0
        peak = 0
        for s in target_range:
            c = counts[s]
            if c == 0:
                uncovered += 1
            elif c > peak:
                peak = c
        return uncovered, peak

    def calculate_initial_counts(A):
        counts = [0] * (n + 1)  # Max sum 256, so size 257 is safe, but we only care about 128-256
        # Actually, let's just use a dict or array of size 257
        counts = [0] * 257
        A_sorted = sorted(A)
        for i in range(len(A_sorted)):
            for j in range(i, len(A_sorted)):
                s = A_sorted[i] + A_sorted[j]
                if target_lo <= s <= target_hi:
                    counts[s] += 1
        return counts

    def local_search(initial_A, num_steps):
        A = list(initial_A)
        A_set = set(A)
        counts = calculate_initial_counts(A)
        curr_unc, curr_peak = get_score(counts)
        curr_score = (curr_unc, curr_peak)
        
        best_A_local = A[:]
        best_score_local = curr_score
        
        # Pre-compute contributions to peak?
        # For each x, count how many times it appears in pairs summing to current peak.
        
        for step in range(num_steps):
            if time.time() > deadline:
                break
            
            # Identify current peak sums
            # We might have multiple sums with the same peak count
            # Let's just pick a random x from A, but bias towards x that are in many peak pairs?
            # That's expensive. Let's stick to random x, but maybe try multiple y's for each x?
            
            x = random.choice(A)
            
            # Try a few random y's
            improved = False
            for _ in range(10): # Try 10 candidates for y
                y = random.randint(0, n)
                if y in A_set:
                    continue
                
                # Simulate swap x -> y
                # 1. Remove x
                # 2. Add y
                # 3. Check score
                
                # We need to update counts.
                # Instead of copying, let's apply, check, rollback if bad.
                
                # Save state? No, just modify and revert.
                # But modifying counts is O(k). Reverting is O(k).
                # So 2 * O(k) per candidate. 10 candidates -> 20 * O(k).
                # This might be slow.
                
                # Alternative: Calculate delta counts.
                # delta[s] = (count of pairs with y) - (count of pairs with x)
                
                # Pairs removed: (x, z) for z in A, z != x, and (x, x)
                # Pairs added: (y, z) for z in A, and (y, y)
                
                # Let's just do the copy approach for simplicity and speed in Python (C-level copy is fast).
                # But 10 copies per step might be too slow.
                
                # Let's try 1 candidate per step, but increase steps.
                pass
            
            # Revised strategy: 1 random swap per step, but with "restarts" if stuck.
            # Or, just do 1 swap, if it improves, keep it. If not, revert.
            
            y = random.randint(0, n)
            attempts = 0
            while y in A_set and attempts < 5:
                y = random.randint(0, n)
                attempts += 1
            if y in A_set:
                continue

            # Save state
            old_A = A[:]
            old_counts = counts[:]
            
            # Apply Swap
            A.remove(x)
            A_set.remove(x)
            for z in A:
                s = x + z
                if target_lo <= s <= target_hi:
                    counts[s] -= 1
            s = 2 * x
            if target_lo <= s <= target_hi:
                counts[s] -= 1
            
            A.append(y)
            A_set.add(y)
            for z in A:
                if z == y:
                    continue
                s = y + z
                if target_lo <= s <= target_hi:
                    counts[s] += 1
            s = 2 * y
            if target_lo <= s <= target_hi:
                counts[s] += 1
            
            new_unc, new_peak = get_score(counts)
            new_score = (new_unc, new_peak)
            
            if new_score < curr_score:
                curr_score = new_score
                if curr_score < best_score_local:
                    best_score_local = curr_score
                    best_A_local = A[:]
            else:
                # Rollback
                A = old_A
                A_set = set(A)
                counts = old_counts
        
        return best_A_local, best_score_local

    # Try different initial sizes
    # The optimal size is likely around 40-50.
    sizes = [35, 40, 45, 50]
    
    for k in sizes:
        if time.time() > deadline:
            break
            
        # Multiple restarts
        # Estimate time per restart
        # 2000 steps * (copy + update)
        # Let's do 10 restarts with 5000 steps each.
        num_restarts = 10
        steps_per_restart = 5000
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Random initial set
            init_A = random.sample(range(n + 1), k)
            
            A_res, score_res = local_search(init_A, steps_per_restart)
            
            if score_res < (best_uncovered, best_peak):
                best_uncovered, best_peak = score_res
                best_A = A_res
                
            # Early stopping if we get a very good result
            if best_uncovered == 0 and best_peak <= 5:
                break
                
        if best_uncovered == 0 and best_peak <= 5:
            break

    if best_A is None:
        # Fallback
        return list(range(64, 129))
        
    return best_A
