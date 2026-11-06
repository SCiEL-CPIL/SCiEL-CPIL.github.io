import time
import random

def _precompute_factors(limit):
    """Precompute prime factors for all numbers up to limit."""
    factors = [[] for _ in range(limit + 1)]
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
                factors[j].append(i)
    for i in range(2, limit + 1):
        if is_prime[i]:
            factors[i].append(i)
    return factors

# Precompute for max sum 20000
MAX_SUM = 20000
FACTORS = _precompute_factors(MAX_SUM)

def construct():
    start_time = time.time()
    timeout = 55.0 # Safety margin
    
    best_solution = None
    best_score = float('inf')
    
    # We will try a few restarts
    # Each restart involves local search
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Initialize
        # Strategy: Random or Structured?
        # Let's try random distinct integers
        x = random.sample(range(1, 10001), 12)
        
        # State:
        # counts[p] = number of sums divisible by p
        counts = [0] * (MAX_SUM + 1)
        active_count = 0
        
        def update_counts(s, delta):
            nonlocal active_count
            for p in FACTORS[s]:
                old_c = counts[p]
                new_c = old_c + delta
                counts[p] = new_c
                if old_c == 0 and new_c > 0:
                    active_count += 1
                elif old_c > 0 and new_c == 0:
                    active_count -= 1
        
        # Initialize counts
        for i in range(12):
            for j in range(i + 1, 12):
                s = x[i] + x[j]
                update_counts(s, 1)
        
        current_score = active_count
        if current_score < best_score:
            best_score = current_score
            best_solution = x[:]
            
        # Local Search
        improved = True
        while improved and (time.time() - start_time < timeout):
            improved = False
            for i in range(12):
                old_val = x[i]
                best_v = -1
                best_c = current_score
                
                # To speed up, we might not check all 10000 if we find a very good one early?
                # But we want the best move for this i.
                # We can randomize the order of v to avoid bias?
                # Or just iterate 1..10000.
                
                # Optimization: If we just want *an* improvement, we can stop early.
                # But finding the *best* improvement for i might lead to faster convergence.
                # Given the cost of 10000 checks, let's try to be smart.
                # We can sample candidates if time is tight.
                # For now, full scan.
                
                # We need to check distinctness.
                # x_set = set(x)
                # remove old_val from set for check
                
                for v in range(1, 10001):
                    if v == old_val:
                        continue
                    if v in x: # O(12) check
                        continue
                    
                    # Simulate change
                    # We need to revert counts after simulation
                    # Since we are in a tight loop, modifying global counts and reverting is risky if not careful.
                    # Better: Calculate delta in active_count without modifying global counts?
                    # No, we need to know if a prime count goes 1->0 or 0->1.
                    # We can do a local simulation.
                    
                    # Let's use a temporary structure or just careful revert.
                    # Revert is safer if we use a list of changes.
                    
                    changes = []
                    local_active_delta = 0
                    
                    # Remove old sums
                    for j in range(12):
                        if j == i: continue
                        s_old = old_val + x[j]
                        for p in FACTORS[s_old]:
                            old_c = counts[p]
                            new_c = old_c - 1
                            changes.append((p, new_c))
                            if old_c == 1 and new_c == 0:
                                local_active_delta -= 1
                            # if old_c > 1, no change in active status
                            
                    # Add new sums
                    for j in range(12):
                        if j == i: continue
                        s_new = v + x[j]
                        for p in FACTORS[s_new]:
                            # We need the current count in the *simulated* state.
                            # The simulated count is counts[p] - (decrements for p) + (increments for p so far)
                            # This is getting complicated to track without modifying counts.
                            
                            # Alternative: Just modify counts, calculate, then revert.
                            pass
                    
                    # Let's stick to modify and revert. It's simpler and less error-prone.
                    # We must ensure we revert ALL changes.
                    
                    # --- SIMULATION START ---
                    # Store original counts for affected primes?
                    # Or just apply changes and revert them.
                    
                    affected_primes = set()
                    
                    for j in range(12):
                        if j == i: continue
                        s_old = old_val + x[j]
                        for p in FACTORS[s_old]:
                            affected_primes.add(p)
                            counts[p] -= 1
                            if counts[p] == 0:
                                local_active_delta -= 1
                            
                    for j in range(12):
                        if j == i: continue
                        s_new = v + x[j]
                        for p in FACTORS[s_new]:
                            if p not in affected_primes:
                                affected_primes.add(p)
                            # Check if count was 0 before increment?
                            # We need to know if it was 0 in the *current simulated state*.
                            # If we decremented it, it might be 0.
                            # If we didn't touch it, it's original counts[p].
                            
                            # This logic is tricky because of order of operations.
                            # Let's use a helper to get current simulated count.
                            # Actually, just tracking delta is hard if multiple sums affect same prime.
                            
                            # Let's revert the "modify and revert" idea to a "calculate delta" approach using a temporary dict for counts?
                            # No, that's slow.
                            
                            # Let's go back to:
                            # 1. Apply decrement.
                            # 2. Apply increment.
                            # 3. Check active status.
                            # 4. Revert.
                            
                            # To do this correctly, we need to know the state *after* all decrements but *before* increments?
                            # No, the order doesn't matter for the final count.
                            # But for active status change, we need to know if it crossed 0.
                            
                            # Let's just compute the final count for each affected prime in the simulated state.
                            # final_count[p] = counts[p] - dec_count[p] + inc_count[p]
                            # active_before = 1 if counts[p] > 0 else 0
                            # active_after = 1 if final_count[p] > 0 else 0
                            # delta += active_after - active_before
                            
                            # This is much cleaner!
                            
                            pass # This block is replaced by the cleaner logic below

                    # --- CLEANER SIMULATION LOGIC ---
                    # Reset local_active_delta
                    local_active_delta = 0
                    # We need to collect all affected primes and their net change in count.
                    # net_change[p] = inc_count[p] - dec_count[p]
                    
                    # To avoid allocating dicts in the inner loop, we can use arrays?
                    # Or just use a list of (p, delta) and aggregate?
                    # Given small number of sums (11), number of affected primes is small.
                    
                    # Let's use a temporary dictionary for this specific v evaluation.
                    # It might be slow but correct.
                    # If too slow, we optimize later.
                    
                    temp_counts_delta = {}
                    
                    for j in range(12):
                        if j == i: continue
                        s_old = old_val + x[j]
                        for p in FACTORS[s_old]:
                            temp_counts_delta[p] = temp_counts_delta.get(p, 0) - 1
                        
                    for j in range(12):
                        if j == i: continue
                        s_new = v + x[j]
                        for p in FACTORS[s_new]:
                            temp_counts_delta[p] = temp_counts_delta.get(p, 0) + 1
                            
                    # Now calculate active delta
                    for p, delta_c in temp_counts_delta.items():
                        old_c = counts[p]
                        new_c = old_c + delta_c
                        if old_c == 0 and new_c > 0:
                            local_active_delta += 1
                        elif old_c > 0 and new_c == 0:
                            local_active_delta -= 1
                            
                    new_score = current_score + local_active_delta
                    if new_score < best_c:
                        best_c = new_score
                        best_v = v
                        
                # End v loop
                
                if best_v != -1:
                    # Apply the change
                    old_val = x[i]
                    x[i] = best_v
                    
                    # Update global counts and active_count
                    # We can reuse the temp_counts_delta logic or just re-run update_counts
                    # Let's just re-run update_counts for simplicity and safety, 
                    # though we already calculated it.
                    # Actually, we can just apply the deltas we calculated?
                    # No, we didn't store them per prime for the *best* v in a reusable way easily.
                    # Let's just do a standard update.
                    
                    # Remove old
                    for j in range(12):
                        if j == i: continue
                        s_old = old_val + x[j]
                        update_counts(s_old, -1)
                        
                    # Add new
                    for j in range(12):
                        if j == i: continue
                        s_new = x[i] + x[j]
                        update_counts(s_new, 1)
                        
                    current_score = best_c
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_solution = x[:]

    return best_solution
