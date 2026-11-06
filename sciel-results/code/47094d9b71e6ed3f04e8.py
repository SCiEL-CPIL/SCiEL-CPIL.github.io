"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    # Deadline guard: 150s limit, use 145s for safety
    deadline = time.time() + 145.0
    
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    # We need to find residues r_p for each prime p.
    # Score L: largest L such that every integer 1..L has coverage >= 2.
    # Coverage(x) = count of primes p such that x % p == r_p.
    
    # Initial heuristic: 
    # We want small numbers to be covered by many primes.
    # If we set r_p = x mod p for many p, x gets high coverage.
    # Let's start with r_p = 1 for all primes, then hill climb.
    residues = [1] * n_primes
    
    # Helper function to compute coverage for a given residues array
    def compute_coverage(res, max_val=10000):
        cov = [0] * (max_val + 1)
        for i, p in enumerate(primes):
            r = res[i]
            if r == 0:
                start = 0
            else:
                start = r
            # x = r mod p => x in {r, r+p, r+2p, ...}
            for x in range(start, max_val + 1, p):
                if x > 0:  # We care about 1..max_val
                    cov[x] += 1
        return cov
    
    def get_L(cov):
        L = 0
        for x in range(1, len(cov)):
            if cov[x] >= 2:
                L = x
            else:
                break
        return L
    
    # Initial coverage and L
    cov = compute_coverage(residues)
    L = get_L(cov)
    best_L = L
    best_residues = residues[:]
    
    # Hill climbing
    # We'll try to improve by changing one residue at a time.
    # Focus on the "bottleneck": the smallest x in 1..L+1 with cov[x] < 2.
    # If L is maximal, then cov[L+1] < 2 (or L is max_val).
    # Actually, if L is the answer, then cov[x] >= 2 for all x <= L, and cov[L+1] < 2.
    # So the bottleneck is x = L+1.
    
    max_iter = 10000
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        
        # Find bottleneck
        if L >= len(cov) - 1:
            # We reached the end of our array, might need to extend? 
            # But max_val is 10000, and we likely won't reach L=10000.
            # If we do, we can return.
            break
            
        x_target = L + 1
        
        best_move_gain = 0
        best_move_i = -1
        best_move_r = -1
        
        # Try changing residue for each prime
        for i in range(n_primes):
            p = primes[i]
            old_r = residues[i]
            
            # To cover x_target, we need x_target % p == new_r
            new_r = x_target % p
            
            if new_r == old_r:
                continue
            
            # Calculate change in coverage
            # Old covered points: old_r, old_r+p, ...
            # New covered points: new_r, new_r+p, ...
            
            # Instead of full update, let's just estimate or do incremental.
            # For speed, let's do a full incremental update of the cov array.
            # It's O(max_val) which is 10000. 168 * 10000 = 1.68M ops per iter.
            # 10000 iters -> 1.68B ops. Too slow in Python.
            
            # We need a smarter update or smaller search space.
            # Maybe only try changing a few primes?
            # Or update cov array incrementally:
            # subtract old pattern, add new pattern.
            
            # Let's try this, but optimize the loop.
            # We can use numpy if available, but let's stick to pure Python first
            # and see if we can prune candidates.
            
            # Only consider if this change might help x_target.
            # It definitely increases cov[x_target] by 1.
            # It decreases cov for other x.
            
            # Let's just do the incremental update for a few random primes or the best ones?
            # No, greedy: try all, but update efficiently.
            
            # Actually, 168 primes * (10000/2) average iterations in inner loop?
            # No, the loop is `for x in range(r, max_val+1, p)`.
            # Number of terms is roughly max_val/p.
            # Sum over p of 10000/p for p in primes.
            # Harmonic series of primes ~ log(log(N)).
            # Sum_{p<=1000} 1/p ~ 2.2.
            # So total ops per prime change ~ 10000 * 2.2 / 168? No.
            # For each prime i, we iterate max_val/p_i times.
            # Total work for evaluating all 168 moves: Sum_{i=1}^{168} (2 * 10000/p_i)
            # Sum 1/p_i for p_i <= 1000 is approx 2.19.
            # So 2 * 10000 * 2.19 = 43800 operations per iteration.
            # This is very fast!
            
            # Let's implement the incremental update carefully.
            
            # Save old cov? No, we can revert.
            
            # 1. Subtract old coverage
            for x in range(old_r, len(cov), p):
                if x > 0:
                    cov[x] -= 1
            
            # 2. Add new coverage
            for x in range(new_r, len(cov), p):
                if x > 0:
                    cov[x] += 1
            
            # 3. Evaluate new L
            # We only care if L increases.
            # L can only increase if cov[x_target] became >= 2 AND all previous were ok (they were).
            # Actually, L could decrease if we broke some x <= L.
            # So we need to check if L increased.
            # Since cov[x] for x < x_target was >= 2, and we might have reduced some,
            # it's possible L decreases.
            # But we only accept if new_L > old_L.
            
            # Let's find the new L.
            new_L = L
            # Check if L can be extended
            if cov[x_target] >= 2:
                # It might go further
                x_check = x_target + 1
                while x_check < len(cov) and cov[x_check] >= 2:
                    new_L = x_check
                    x_check += 1
            # Check if L was reduced (bottleneck moved earlier?)
            # This is hard to check quickly without scanning from 1.
            # But if we reduced coverage at some x <= L, L might drop.
            # However, for greedy "push L forward", we usually accept moves that increase L.
            # If a move decreases L, it's bad.
            # Let's just compute new_L by scanning from 1 if cov[x_target] >= 2?
            # No, if cov[x_target] < 2, new_L <= L.
            # If cov[x_target] >= 2, new_L >= x_target = L+1.
            # So we only need to scan if cov[x_target] >= 2.
            
            if new_L > best_move_gain:
                best_move_gain = new_L
                best_move_i = i
                best_move_r = new_r
            
            # Revert
            # 1. Add back old
            for x in range(old_r, len(cov), p):
                if x > 0:
                    cov[x] += 1
            # 2. Remove new
            for x in range(new_r, len(cov), p):
                if x > 0:
                    cov[x] -= 1
        
        if best_move_i != -1 and best_move_gain > L:
            # Apply best move
            residues[best_move_i] = best_move_r
            # Update cov array permanently
            p = primes[best_move_i]
            old_r = residues[best_move_i] # Wait, we just set it. We need the old one.
            # We didn't store old_r in the loop variable scope outside.
            # Let's re-fetch or store.
            # In the loop, `old_r = residues[i]`.
            # So `old_r` is the value before change.
            # Let's assume `old_r` is still valid from the last iteration of the inner loop that won?
            # No, `old_r` is local to the `for i` loop.
            # We need to store `old_r_best`.
            
            # Let's refactor to store old_r_best.
            # But for now, let's just re-compute the update for the applied move?
            # Or just update cov again.
            
            # To be safe and simple, let's just re-compute coverage from scratch?
            # No, that's O(N * max_val) ~ 1.6M ops.
            # Better to just apply the delta.
            # But I need `old_r`.
            # Let's modify the loop to store `old_r_best`.
            
            # Actually, I can just re-run the "subtract old, add new" logic with the stored values.
            # I need to store old_r_best alongside best_move_i.
            
            # Let's restart the logic with proper storage.
            
            # Wait, the code above doesn't store old_r_best.
            # Let's fix the loop.
            
            pass # Placeholder, will rewrite loop below
            
        else:
            # No improvement found
            break
    
    # Let's rewrite the hill climbing loop properly
    
    # Reset to best found so far (which is initial for now)
    residues = best_residues[:]
    cov = compute_coverage(residues)
    L = get_L(cov)
    
    while time.time() < deadline:
        # Find bottleneck
        if L >= len(cov) - 1:
            break
        
        x_target = L + 1
        
        best_move_gain = 0
        best_move_i = -1
        best_move_r = -1
        best_move_old_r = -1
        
        # Try changing residue for each prime
        for i in range(n_primes):
            p = primes[i]
            old_r = residues[i]
            
            # To cover x_target, we need x_target % p == new_r
            new_r = x_target % p
            
            if new_r == old_r:
                continue
            
            # Incremental update
            # Subtract old
            for x in range(old_r, len(cov), p):
                if x > 0:
                    cov[x] -= 1
            
            # Add new
            for x in range(new_r, len(cov), p):
                if x > 0:
                    cov[x] += 1
            
            # Evaluate new L
            new_L = L
            if cov[x_target] >= 2:
                x_check = x_target + 1
                while x_check < len(cov) and cov[x_check] >= 2:
                    new_L = x_check
                    x_check += 1
            
            if new_L > best_move_gain:
                best_move_gain = new_L
                best_move_i = i
                best_move_r = new_r
                best_move_old_r = old_r
            
            # Revert
            # Add back old
            for x in range(old_r, len(cov), p):
                if x > 0:
                    cov[x] += 1
            # Remove new
            for x in range(new_r, len(cov), p):
                if x > 0:
                    cov[x] -= 1
        
        if best_move_i != -1 and best_move_gain > L:
            # Apply best move
            residues[best_move_i] = best_move_r
            # Update cov array permanently
            p = primes[best_move_i]
            old_r = best_move_old_r
            new_r = best_move_r
            
            # Subtract old
            for x in range(old_r, len(cov), p):
                if x > 0:
                    cov[x] -= 1
            # Add new
            for x in range(new_r, len(cov), p):
                if x > 0:
                    cov[x] += 1
            
            L = best_move_gain
            if L > best_L:
                best_L = L
                best_residues = residues[:]
        else:
            # No improvement, try some random moves to escape local optima?
            # Or break.
            # Let's try a few random restarts or random moves if stuck.
            # For simplicity, break.
            break
    
    return best_residues
# EVOLVE-BLOCK-END
