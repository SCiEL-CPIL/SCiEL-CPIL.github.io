import time
import math
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate Primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(1000)
    n_primes = len(primes)
    
    # Helper: Check if a specific integer x is covered by at least 2 residues
    def is_double_covered(x, residues):
        count = 0
        for i in range(n_primes):
            if x % primes[i] == residues[i]:
                count += 1
                if count >= 2:
                    return True
        return False

    # Helper: Find the first gap in the prefix 1..L
    def find_gap(residues, L):
        for x in range(1, L + 1):
            if not is_double_covered(x, residues):
                return x
        return -1 # No gap found

    # Helper: Try to fix a gap at x by changing one residue
    # Returns (success, new_residues)
    def fix_gap(x, residues):
        # We need to add coverage to x.
        # Currently, x has count < 2.
        # We need to find a prime p_i such that changing r_i to x % p_i 
        # increases the count for x without breaking the "double coverage" 
        # of any other number in 1..L (or at least the critical ones).
        
        # First, identify which primes are "free" to change for x
        # A prime i is a candidate if residues[i] != x % primes[i]
        
        candidates = []
        for i in range(n_primes):
            req_r = x % primes[i]
            if residues[i] != req_r:
                candidates.append(i)
        
        if not candidates:
            return False, None
        
        # Sort candidates by prime size? 
        # Smaller primes affect more numbers in the prefix, so they are "risky".
        # Larger primes affect fewer numbers, so they are "safer" to change.
        # Let's try larger primes first (less likely to break other coverages).
        candidates.sort(key=lambda i: primes[i], reverse=True)
        
        for i in candidates:
            old_r = residues[i]
            new_r = x % primes[i]
            residues[i] = new_r
            
            # Check if this broke anything else in 1..L?
            # We don't know L exactly here, but we can check the current frontier.
            # Actually, we just want to see if x is now covered.
            # And we hope we didn't break x-1, x-2, etc.
            # To be safe, we check if the gap at x is resolved.
            # If it is, we accept. The caller will re-evaluate the whole prefix.
            
            if is_double_covered(x, residues):
                return True, residues
            else:
                # Revert
                residues[i] = old_r
                
        return False, None

    # Main Strategy: Greedy Frontier Extension with Backtracking
    # We build the residues incrementally.
    # Start with L=0.
    # Try to extend to L+1.
    # If we fail to extend, we might need to backtrack to a previous decision.
    
    # Since full backtracking is exponential, we use a "limited backtracking" or "restart" strategy.
    # However, the directive suggests a different structural change.
    # Let's try a "Local Repair" loop that is deterministic and prioritized.
    
    # Initialization: 
    # A good heuristic for initial residues is to spread them out.
    # r_p = 0 is often bad because 0 is a common residue.
    # Let's try r_p = 1 for all? Or random?
    # The current seed uses ((i+2)//2) % p.
    # Let's start with a "balanced" random initialization, then repair.
    
    random.seed(42)
    
    # State
    best_residues = [0] * n_primes
    best_score = 0
    
    # We will run multiple "restarts" with different initializations if time permits.
    # But first, let's try a strong deterministic repair on a standard init.
    
    def run_repair_loop(init_residues, max_L_target=1000):
        """
        Attempts to extend the coverage prefix as far as possible.
        """
        residues = init_residues[:]
        
        # Calculate initial L
        L = 0
        while L < max_L_target:
            L += 1
            if not is_double_covered(L, residues):
                L -= 1
                break
        current_L = L
        
        # Now try to extend
        steps = 0
        max_steps = 5000 # Limit steps per repair run to ensure time safety
        
        while current_L < max_L_target and steps < max_steps:
            steps += 1
            target_x = current_L + 1
            
            # Try to fix target_x
            success, new_residues = fix_gap(target_x, residues)
            
            if success:
                residues = new_residues
                # Check if we broke any previous coverage?
                # We need to verify the whole prefix 1..target_x is still valid.
                # If not, we might have introduced a new gap earlier.
                new_L = 0
                for x in range(1, target_x + 1):
                    if is_double_covered(x, residues):
                        new_L = x
                    else:
                        break
                
                if new_L > current_L:
                    current_L = new_L
                else:
                    # We broke something earlier. 
                    # This is a conflict. 
                    # For now, revert? Or try to fix the new gap?
                    # Let's try to fix the new gap if it's smaller than target_x.
                    # This can lead to a loop. 
                    # Simple heuristic: If we didn't advance, revert and mark this prime as "bad" for this x?
                    # Actually, if new_L < current_L, we regressed.
                    # Revert the change.
                    # residues = old_residues (we don't have it saved here easily in the loop structure above)
                    # Let's restructure: keep a copy before change.
                    
                    # Since we modified residues in place in fix_gap, we need to be careful.
                    # Let's assume fix_gap returns the modified list.
                    # If it failed to improve L, we should revert.
                    # But fix_gap already reverted if is_double_covered(x) was false.
                    # The issue is if it covered x but broke x-1.
                    # In that case, is_double_covered(target_x) is True, so fix_gap returned True.
                    # But the prefix check failed.
                    # We need to revert.
                    # We don't have the old residues here. 
                    # Let's change strategy: don't modify in place in fix_gap, or pass a copy.
                    
                    # To keep it simple and fast, let's assume that if we fix the frontier, we rarely break the back.
                    # If we do break the back, we just accept the regression and try to fix the new gap?
                    # No, that's bad.
                    
                    # Let's try a different move: Random perturbation if stuck.
                    # Pick a random prime, change residue, check if L improves.
                    i = random.randint(0, n_primes - 1)
                    old_r = residues[i]
                    new_r = random.randint(0, primes[i] - 1)
                    if new_r != old_r:
                        residues[i] = new_r
                        # Check new L
                        new_L = 0
                        for x in range(1, max(current_L, 100) + 10): # Check a bit ahead
                            if is_double_covered(x, residues):
                                new_L = x
                            else:
                                break
                        if new_L > current_L:
                            current_L = new_L
                        else:
                            residues[i] = old_r
                    continue # Skip the "success" block logic since we handled it here? 
                    # No, the structure is:
                    # if success: ... else: ...
                    # The "else" is not triggered if success is True.
                    # So if success is True but L didn't increase, we are stuck in the "success" branch.
                    # We need to handle the case where success=True but L didn't increase.
                    
                    # Let's refactor the loop to be cleaner.
                    pass 
            else:
                # Could not fix target_x directly.
                # Try random perturbation to escape local minimum.
                i = random.randint(0, n_primes - 1)
                old_r = residues[i]
                new_r = random.randint(0, primes[i] - 1)
                if new_r != old_r:
                    residues[i] = new_r
                    new_L = 0
                    # Check up to current_L + 10
                    limit = current_L + 10
                    if limit > max_L_target: limit = max_L_target
                    for x in range(1, limit + 1):
                        if is_double_covered(x, residues):
                            new_L = x
                        else:
                            break
                    if new_L > current_L:
                        current_L = new_L
                    else:
                        residues[i] = old_r
        
        return current_L, residues

    # The previous logic was a bit messy. Let's write a cleaner version.
    
    def optimize(init_residues):
        residues = init_residues[:]
        
        # Calculate initial L
        def get_L(res):
            L = 0
            # We can speed this up by checking from the end if we have a previous L, 
            # but for simplicity, let's just scan. 
            # Optimization: We know L is likely small.
            # Max L is 1000. 
            # 1000 * 168 checks = 168,000 ops. Very fast.
            for x in range(1, 1001):
                if is_double_covered(x, res):
                    L = x
                else:
                    break
            return L
        
        current_L = get_L(residues)
        best_L = current_L
        best_res = residues[:]
        
        # Greedy Repair Loop
        # We iterate, trying to fix the first gap.
        # If we can't fix it, we do a random kick.
        
        stagnation_counter = 0
        
        while time.time() < deadline and current_L < 1000:
            gap_x = current_L + 1
            
            # Try to fix gap_x
            fixed = False
            # Try candidates sorted by prime size (larger first)
            candidates = []
            for i in range(n_primes):
                req_r = gap_x % primes[i]
                if residues[i] != req_r:
                    candidates.append(i)
            
            candidates.sort(key=lambda i: primes[i], reverse=True)
            
            for i in candidates:
                old_r = residues[i]
                new_r = gap_x % primes[i]
                residues[i] = new_r
                
                new_L = get_L(residues)
                
                if new_L > current_L:
                    current_L = new_L
                    best_L = current_L
                    best_res = residues[:]
                    stagnation_counter = 0
                    fixed = True
                    break
                else:
                    residues[i] = old_r
            
            if not fixed:
                # Random kick
                i = random.randint(0, n_primes - 1)
                old_r = residues[i]
                new_r = random.randint(0, primes[i] - 1)
                if new_r != old_r:
                    residues[i] = new_r
                    new_L = get_L(residues)
                    if new_L > current_L:
                        current_L = new_L
                        best_L = current_L
                        best_res = residues[:]
                        stagnation_counter = 0
                    else:
                        residues[i] = old_r
                        stagnation_counter += 1
                
                # If we are very stuck, restart with a different random seed?
                # Or just keep kicking.
                
        return best_L, best_res

    # Run the optimizer with a few different initializations
    # 1. Seed initialization
    init1 = [((i + 2) // 2) % p for i, p in enumerate(primes)]
    score1, res1 = optimize(init1)
    
    # 2. Random initialization
    random.seed(123)
    init2 = [random.randint(0, p - 1) for p in primes]
    score2, res2 = optimize(init2)
    
    # 3. "Zero" initialization (all 0)
    init3 = [0] * n_primes
    score3, res3 = optimize(init3)
    
    # Pick the best
    best_score = max(score1, score2, score3)
    if best_score == score1:
        return res1
    elif best_score == score2:
        return res2
    else:
        return res3
