import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        if n == 3:
            return True
        if n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    primes = [p for p in range(2, 1001) if is_prime(p)]
    
    # Select a set of primes that fits the budget and is likely good for coverage
    # We want small primes for better local coverage.
    # Try to maximize the number of primes? Or minimize the largest gap?
    # Let's start with primes from 11 upwards, and add more if budget allows?
    # Actually, smaller primes are "denser". 
    # Let's try to include as many small primes as possible under budget 1.
    
    # Sort primes by value
    # Greedy: Start with smallest, add if budget allows?
    # Sum 1/2 + 1/3 + 1/5 + 1/7 = 1.17 > 1.
    # So we can't have 2,3,5,7.
    # What about 3,5,7,11? 1/3+1/5+1/7+1/11 = 0.33+0.2+0.14+0.09 = 0.76.
    # Add 13? 0.76 + 0.07 = 0.83.
    # Add 17? 0.83 + 0.05 = 0.88.
    # Add 19? 0.88 + 0.05 = 0.93.
    # Add 23? 0.93 + 0.04 = 0.97.
    # Add 29? 0.97 + 0.03 = 1.00.
    # So {3,5,7,11,13,17,19,23,29} is a candidate set.
    
    # Let's try this set.
    candidate_primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    # Check budget
    cost = sum(Fraction(1, p) for p in candidate_primes)
    
    # We can swap out larger primes for smaller ones if we remove some.
    # But let's just use a robust heuristic:
    # Start with primes p >= 13 (like seed) but optimize residues.
    # Then try to swap a few large primes for smaller ones if budget allows.
    
    # Let's initialize with p >= 13
    initial_primes = [p for p in primes if p >= 13]
    cost = sum(Fraction(1, p) for p in initial_primes)
    
    # Try to add 11
    if cost + Fraction(1, 11) <= Fraction(1):
        initial_primes.append(11)
        cost += Fraction(1, 11)
        
    # Try to add 7?
    # If we add 7, cost increases by 1/7 ~ 0.14.
    # Current cost with 11 added is ~ 0.84 + 0.09 = 0.93.
    # 0.93 + 0.14 = 1.07 > 1.
    # So we can't add 7 directly.
    # We could swap two large primes for 7?
    # e.g. remove 997 and 991? Cost saved ~ 0.002. Not enough.
    # We need to save 0.07.
    # Remove primes summing to ~ 0.07.
    # 1/13 ~ 0.07.
    # So swap {13} for {7}? 
    # If we remove 13 and add 7:
    # Cost change: -1/13 + 1/7 = -0.077 + 0.142 = +0.065.
    # Current cost 0.93 + 0.065 = 0.995 <= 1.
    # This looks promising! 7 is much better for coverage than 13.
    
    # Let's implement a simple swap heuristic to improve the prime set.
    # Iterate through potential additions (small primes not in set)
    # and try to remove enough large primes to make it fit.
    
    current_primes = initial_primes
    current_cost = cost
    prime_set = set(current_primes)
    
    # Try adding primes 2, 3, 5, 7, 11 (if not present) in order of preference (smallest first)
    # 2 is too expensive (0.5).
    # 3 (0.33), 5 (0.2), 7 (0.14), 11 (0.09)
    
    for p_add in [7, 5, 3]:
        if p_add in prime_set:
            continue
        need = Fraction(1, p_add) - (Fraction(1) - current_cost)
        if need <= 0:
            # Fits directly
            current_primes.append(p_add)
            prime_set.add(p_add)
            current_cost += Fraction(1, p_add)
        else:
            # Need to remove primes summing to at least 'need'
            # Sort primes in current set by value descending (largest first) to remove least valuable?
            # Actually, we want to keep small primes. So remove largest primes first.
            removable = [p for p in current_primes if p > p_add]
            removable.sort(reverse=True)
            removed_sum = Fraction(0)
            to_remove = []
            for p_rem in removable:
                to_remove.append(p_rem)
                removed_sum += Fraction(1, p_rem)
                if removed_sum >= need:
                    break
            
            if removed_sum >= need:
                for p_rem in to_remove:
                    current_primes.remove(p_rem)
                    prime_set.remove(p_rem)
                    current_cost -= Fraction(1, p_rem)
                
                current_primes.append(p_add)
                prime_set.add(p_add)
                current_cost += Fraction(1, p_add)
                # If we still have budget, we might have removed too much?
                # If removed_sum > need, we have extra budget.
                # We can try to add back some of the removed ones? 
                # Or add other small primes?
                # For now, just add the target.
                
    # Final check
    assert current_cost <= Fraction(1)
    
    N = len(current_primes)
    max_L = 50000
    
    # Initialize residues randomly
    residues = [random.randint(0, p - 1) for p in current_primes]
    
    # Coverage array
    count = [0] * (max_L + 1)
    
    def update_coverage(idx, old_r, new_r):
        p = current_primes[idx]
        for n in range(old_r, max_L + 1, p):
            if n >= 1:
                count[n] -= 1
        for n in range(new_r, max_L + 1, p):
            if n >= 1:
                count[n] += 1
                
    # Initial coverage
    for i, (p, r) in enumerate(zip(current_primes, residues)):
        for n in range(r, max_L + 1, p):
            if n >= 1:
                count[n] += 1
                
    def find_L():
        for n in range(1, max_L + 1):
            if count[n] == 0:
                return n - 1
        return max_L
        
    best_L = find_L()
    best_residues = residues[:]
    
    # Local Search (Hill Climbing)
    def local_search(residues):
        # Copy state to local variables? 
        # We operate on the global count array, so we must revert changes carefully.
        improved = True
        local_L = best_L
        # We need to pass current residues? No, we modify the global 'residues' list in place?
        # Better: Pass a list of residues and return new list and L.
        # But modifying global count is faster if we manage reverts.
        
        # Let's do it in-place on the global 'residues' and 'count'
        # and save/restore if we want to backtrace? 
        # Standard HC: Try change, if better, keep. If not, revert.
        
        while improved:
            improved = False
            for i in range(N):
                p = current_primes[i]
                old_r = residues[i]
                best_r = old_r
                best_L_local = local_L
                
                for new_r in range(p):
                    if new_r == old_r:
                        continue
                    
                    # Apply
                    update_coverage(i, old_r, new_r)
                    
                    # Check L
                    # Optimization: We only care if L increases.
                    # If L increases, it must be that count[local_L+1] became > 0?
                    # Or count at some point before local_L+1?
                    # No, L is the first 0.
                    # If we change residue, holes might appear before current L?
                    # Yes. So we must scan from 1.
                    # But if we scan from 1, it's O(L).
                    # L is ~1000-5000. N is ~100. P is ~100.
                    # 100 * 100 * 5000 = 50 million ops per pass? 
                    # 50M ops is ~ 1-2 seconds in Python?
                    # We have 110s. So ~50-100 passes possible.
                    
                    L_local = find_L()
                    
                    if L_local > best_L_local:
                        best_L_local = L_local
                        best_r = new_r
                        local_L = L_local # Update global best for this pass?
                        # No, we should compare against the current best in this iteration
                        # But if we find a better one, we update best_L_local.
                    else:
                        # Revert if not better?
                        # Wait, we need to revert before trying next new_r
                        # UNLESS new_r is the best so far.
                        # Logic:
                        # 1. Save old state
                        # 2. Try new_r
                        # 3. If better than current best, update best
                        # 4. Revert to old state (unless new_r is the one we want to keep)
                        # This is expensive.
                        
                        # Better:
                        # Keep a copy of the count array? No, too big.
                        # Apply, check, revert.
                        pass 
                        
                # End new_r loop
                # Apply the best_r found
                if best_r != old_r:
                    update_coverage(i, old_r, best_r)
                    residues[i] = best_r
                    local_L = best_L_local
                    improved = True
                    
                # End for i loop
            # End while improved
            
        return local_L

    # The logic above for reverting inside the inner loop is tricky.
    # Let's rewrite the inner loop to be safe:
    
    def local_search_optimized():
        improved = True
        cur_L = best_L
        # We use the global 'residues' and 'count'
        
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(N):
                p = current_primes[i]
                old_r = residues[i]
                best_r = old_r
                best_L_i = cur_L
                
                # We will try all residues
                # We need to restore state to (old_r) after each try
                # So:
                # 1. Save current count? No.
                # 2. Apply change from old_r to new_r.
                # 3. Calculate L.
                # 4. If L > best_L_i, update best_L_i and best_r.
                # 5. Revert change (new_r back to old_r).
                
                for new_r in range(p):
                    if new_r == old_r:
                        continue
                    
                    # Apply
                    update_coverage(i, old_r, new_r)
                    
                    L_new = find_L()
                    
                    if L_new > best_L_i:
                        best_L_i = L_new
                        best_r = new_r
                        
                    # Revert
                    update_coverage(i, new_r, old_r)
                    
                # End for new_r
                
                # Apply best_r if different from old_r
                if best_r != old_r:
                    update_coverage(i, old_r, best_r)
                    residues[i] = best_r
                    cur_L = best_L_i
                    improved = True
                    
                if time.time() > deadline - 10:
                    break
                    
        return cur_L

    # ILS Loop
    # Start with random residues
    # Run Local Search
    # If stuck, kick
    
    iterations = 0
    while time.time() < deadline - 10:
        iterations += 1
        
        # If it's the first iteration, just run local search on initial random
        # Otherwise, kick first?
        
        if iterations > 1:
            # Kick: change random residues
            num_kicks = random.randint(1, 5)
            for _ in range(num_kicks):
                idx = random.randint(0, N - 1)
                old_r = residues[idx]
                p = current_primes[idx]
                new_r = random.randint(0, p - 1)
                if new_r != old_r:
                    update_coverage(idx, old_r, new_r)
                    residues[idx] = new_r
        
        # Run Local Search
        L_new = local_search_optimized()
        
        if L_new > best_L:
            best_L = L_new
            best_residues = residues[:]
        
    # Restore best state
    # We need to update count array to reflect best_residues?
    # The return value is just the list of classes.
    # The evaluator will likely check validity and score.
    # We don't need to return the count array.
    
    return [[current_primes[i], best_residues[i]] for i in range(N)]
