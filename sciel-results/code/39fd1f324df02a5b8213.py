import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    # Time limit guard
    start_time = time.time()
    time_limit = 110.0  # Seconds
    
    # 1. Generate primes up to 1000
    primes = []
    for p in range(2, 1001):
        is_prime = True
        for q in range(2, int(p ** 0.5) + 1):
            if p % q == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(p)
            
    n_primes = len(primes)
    
    # 2. Setup for evaluation
    # We want to find the largest L. 
    # Based on the problem, L >= 1000 is the target. 
    # Let's set a reasonable upper bound for the coverage array to allow search space.
    # If we go too high, initialization and updates might be slow.
    # 50,000 is a good balance. 
    LIMIT = 50000
    
    # Coverage array: cov[i] is the number of primes p such that x=i is in the residue class.
    # We use 1-based indexing for x in 1..LIMIT.
    # cov[0] is unused or for x=0.
    cov = [0] * (LIMIT + 1)
    
    # Initial residues
    # Start with a random assignment to explore the space
    residues = [random.randint(0, p - 1) for p in primes]
    
    # Initialize coverage
    def update_coverage(residues, cov, primes):
        cov = [0] * (LIMIT + 1)
        for i, p in enumerate(primes):
            r = residues[i]
            # x = r, r+p, r+2p ...
            # Note: x must be >= 1.
            # If r == 0, the first positive x is p.
            # If r > 0, the first positive x is r.
            start = r
            if start == 0:
                start = p
            for x in range(start, LIMIT + 1, p):
                cov[x] += 1
        return cov

    cov = update_coverage(residues, cov, primes)
    
    # Calculate initial score
    def get_score(cov, limit):
        L = 0
        for x in range(1, limit + 1):
            if cov[x] >= 2:
                L = x
            else:
                break
        return L

    current_score = get_score(cov, LIMIT)
    best_score = current_score
    best_residues = residues[:]
    
    # Local Search
    # We try to improve the score by changing one residue at a time.
    # We use a greedy approach with some randomness to escape local minima.
    
    # Helper to apply a change and update score incrementally
    def try_change(idx, new_r):
        nonlocal cov, current_score, best_score, best_residues
        
        old_r = residues[idx]
        if old_r == new_r:
            return
        
        p = primes[idx]
        
        # Identify affected positions
        # Decrement old
        old_start = old_r
        if old_start == 0:
            old_start = p
        old_positions = list(range(old_start, LIMIT + 1, p))
        
        # Increment new
        new_start = new_r
        if new_start == 0:
            new_start = p
        new_positions = list(range(new_start, LIMIT + 1, p))
        
        # Apply changes
        # We need to track if the "validity" of the prefix 1..current_score is broken.
        # And if the prefix can be extended.
        
        # 1. Decrement old positions
        broken = False
        break_point = current_score + 1 # Assume no break initially
        
        for x in old_positions:
            cov[x] -= 1
            if x <= current_score and cov[x] < 2:
                broken = True
                break_point = x
                
        # 2. Increment new positions
        for x in new_positions:
            cov[x] += 1
            
        # 3. Update Score
        if broken:
            # The score has decreased. Find the new largest L such that 1..L are all >= 2.
            # We know that at break_point, cov < 2.
            # So new L < break_point.
            new_L = break_point - 1
            # Scan down from new_L to find the first x where cov[x] < 2
            # Actually, we need to find the largest L' such that 1..L' are valid.
            # Since cov[break_point] < 2, L' <= break_point - 1.
            # But maybe cov[break_point - 1] is also < 2?
            # We scan down from break_point - 1.
            while new_L > 0 and cov[new_L] < 2:
                new_L -= 1
            current_score = new_L
        else:
            # The prefix 1..current_score is still valid.
            # Check if we can extend.
            # Start checking from current_score + 1.
            new_L = current_score
            # We need to check if cov[new_L + 1] >= 2, etc.
            # Note: The change might have increased cov at positions > current_score.
            # It might have also increased cov at positions <= current_score, but that doesn't extend L.
            # It might have decreased cov at positions <= current_score, but we already checked 'broken' is False.
            # So we just scan up.
            
            # Optimization: Only scan up if we actually incremented something near current_score?
            # Simpler: Just scan up from current_score + 1.
            # But we must be careful: did we decrement a position > current_score that was previously >= 2?
            # If we decremented a position > current_score, it doesn't affect the validity of 1..current_score.
            # But it might affect the extension.
            # Example: L=10. cov[11]=2. We change a residue such that cov[11] becomes 1.
            # Then we cannot extend to 11.
            # In our logic:
            # - We checked `broken` for x <= current_score.
            # - If not broken, 1..current_score are valid.
            # - We then try to extend. We scan from current_score + 1.
            # - If cov[current_score + 1] is now 1 (because we decremented it), the scan stops immediately.
            # - This is correct.
            
            while new_L < LIMIT and cov[new_L + 1] >= 2:
                new_L += 1
            current_score = new_L
            
        # Accept if improved
        if current_score > best_score:
            best_score = current_score
            best_residues = residues[:]
            residues[idx] = new_r
            return True
        else:
            # Reject: Revert changes
            for x in old_positions:
                cov[x] += 1
            for x in new_positions:
                cov[x] -= 1
            # current_score is restored implicitly by the logic above?
            # No, we modified current_score. We need to restore it or just not update best.
            # Since we reverted cov, the state is back to before the try.
            # But `current_score` variable was updated. We should restore it too.
            # Actually, it's safer to just not modify `residues` and `current_score` if we reject.
            # In this implementation, I modified cov and current_score locally.
            # I need to restore current_score.
            # Let's refactor: calculate new_score without modifying global state until accepted.
            return False

    # The above try_change modifies global state and reverts on failure.
    # It's a bit messy. Let's write a cleaner version that calculates the potential new score
    # or simply performs the swap, checks, and reverts if worse.
    
    # Simpler Local Search Loop
    steps = 0
    while time.time() - start_time < time_limit:
        # Pick a random prime index
        idx = random.randint(0, n_primes - 1)
        p = primes[idx]
        old_r = residues[idx]
        
        # Pick a new random residue
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue
            
        # Apply change
        residues[idx] = new_r
        
        # Update coverage
        # Decrement old
        old_start = old_r
        if old_start == 0:
            old_start = p
        for x in range(old_start, LIMIT + 1, p):
            cov[x] -= 1
            
        # Increment new
        new_start = new_r
        if new_start == 0:
            new_start = p
        for x in range(new_start, LIMIT + 1, p):
            cov[x] += 1
            
        # Recalculate score
        # Optimization: We know the previous score.
        # If the change only affected x > current_score, the score can only increase or stay same.
        # If the change affected x <= current_score, the score can decrease, increase, or stay same.
        
        # Let's just recalculate from the current_score downwards if needed, or upwards.
        
        # Check if 1..current_score are still valid
        valid = True
        # We only need to check the positions that were decremented and are <= current_score
        # But we don't track which ones were decremented easily without storing them.
        # Given the small size of primes (168) and LIMIT (50k), a full scan is 50k ops.
        # If we do 1000 steps/sec, that's 50M ops/sec. Might be tight in Python.
        
        # Let's use the incremental logic properly.
        
        # 1. Check if any x in 1..current_score has cov[x] < 2.
        # We can just scan 1..current_score.
        # If current_score is small (e.g., 1000), this is fast.
        # If current_score is large (e.g., 10000), this is 10k ops.
        
        new_score = current_score
        
        # Check validity of 1..new_score
        # We only need to check if it broke.
        # If it broke, we find the new L.
        broke = False
        break_idx = new_score + 1
        for x in range(1, new_score + 1):
            if cov[x] < 2:
                broke = True
                break_idx = x
                break
        
        if broke:
            # Find new L
            new_score = break_idx - 1
            while new_score > 0 and cov[new_score] < 2:
                new_score -= 1
        else:
            # Try to extend
            # Start from new_score + 1
            while new_score < LIMIT and cov[new_score + 1] >= 2:
                new_score += 1
                
        # Accept or Reject
        if new_score >= current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_residues = residues[:]
        else:
            # Revert
            residues[idx] = old_r
            # Revert coverage
            # Decrement new
            new_start = new_r
            if new_start == 0:
                new_start = p
            for x in range(new_start, LIMIT + 1, p):
                cov[x] -= 1
            # Increment old
            old_start = old_r
            if old_start == 0:
                old_start = p
            for x in range(old_start, LIMIT + 1, p):
                cov[x] += 1
            # current_score remains unchanged (it was the old one)
            
        steps += 1
        if steps % 1000 == 0:
            # print(f"Steps: {steps}, Score: {current_score}, Best: {best_score}")
            pass

    return best_residues
# EVOLVE-BLOCK-END
