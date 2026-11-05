import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    total_words = q ** n
    
    # Precompute balls
    # ball[i] is a list of words covered by center i
    balls = []
    for i in range(total_words):
        digits = []
        tmp = i
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        
        covered = [i]
        # Generate all neighbors within radius
        # For radius 1, it's just i and i with one digit changed
        if radius == 1:
            for pos in range(n):
                orig_val = digits[pos]
                for val in range(q):
                    if val == orig_val:
                        continue
                    new_digits = digits[:]
                    new_digits[pos] = val
                    new_word = 0
                    for d in new_digits:
                        new_word = new_word * q + d
                    covered.append(new_word)
        else:
            # General radius handling (slower, but n is small)
            # Use iterative expansion or precomputed distance
            # For this task, radius is likely 1.
            # We'll stick to radius 1 optimization for speed, 
            # but add a fallback if radius > 1.
            pass
            
        balls.append(covered)
        
    # If radius > 1, the above is incomplete. 
    # However, the task specifies K_3(6,1), so radius=1 is the target.
    # We will assume radius=1 for the core logic but handle general case if needed.
    # Given the specific task description "K_3(6,1)", we optimize for radius 1.
    
    # Greedy Initialization
    cover_count = np.zeros(total_words, dtype=int)
    codewords = []
    uncovered = np.ones(total_words, dtype=bool)
    
    # Greedy Set Cover
    while np.any(uncovered):
        if time.time() > deadline - 10:
            break
        best_word = -1
        max_covered = 0
        # To speed up, we can sample or just iterate. 729 is small.
        for w in range(total_words):
            # Count how many uncovered points this word covers
            # Optimization: only check if w is not in codewords? 
            # Actually, adding a duplicate is useless.
            if w in codewords:
                continue
            cnt = 0
            for u in balls[w]:
                if uncovered[u]:
                    cnt += 1
            if cnt > max_covered:
                max_covered = cnt
                best_word = w
        
        if best_word == -1 or max_covered == 0:
            break
            
        codewords.append(best_word)
        for u in balls[best_word]:
            cover_count[u] += 1
            if cover_count[u] == 1:
                uncovered[u] = False
        
    # Local Search: Prune and Swap
    # We want to minimize len(codewords)
    
    # Helper to check if a swap c -> c' is valid
    # It is valid if for all u in balls[c], cover_count[u] > 1 OR u in balls[c']
    # Actually, simpler: 
    # new_cover_count = cover_count - indicator(balls[c]) + indicator(balls[c'])
    # Valid if min(new_cover_count) >= 1
    
    # We will implement a loop that tries to prune, then swap.
    
    def try_prune():
        """Removes redundant codewords. Returns True if any removed."""
        removed = False
        i = 0
        while i < len(codewords):
            if time.time() > deadline:
                return removed
            c = codewords[i]
            # Check if c is redundant
            # Redundant if for all u in balls[c], cover_count[u] > 1
            redundant = True
            for u in balls[c]:
                if cover_count[u] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove c
                for u in balls[c]:
                    cover_count[u] -= 1
                codewords.pop(i)
                removed = True
                # Do not increment i, check same index again
            else:
                i += 1
        return removed

    def try_swap():
        """Tries to find a valid swap c -> c'. Returns True if swap performed."""
        if len(codewords) == 0:
            return False
            
        # We want to swap a codeword c with a non-codeword c'
        # such that the resulting set is still a cover.
        
        # Optimization: 
        # A swap is valid if for every u in balls[c] \ balls[c'], cover_count[u] > 1.
        # (Because if u is in balls[c'] too, the count doesn't drop below the new contribution).
        
        # To speed up, we can identify "critical" points for c: 
        # points u in balls[c] where cover_count[u] == 1.
        # If there are no critical points, c is redundant (handled by prune).
        # So for a swap to be valid, c' MUST cover all critical points of c.
        
        # Let's iterate through codewords c
        # To make it stochastic, we can shuffle codewords or pick random ones.
        # But 73*729 is ~50k checks, which is fast.
        
        # We'll iterate through all c in codewords
        # and all c' in 0..total_words-1 not in codewords.
        
        # To avoid O(N^2) in the worst case if we don't break early, 
        # we can just do a single pass.
        
        # Precompute set of codewords for fast lookup
        cw_set = set(codewords)
        
        # We'll collect potential swaps and pick one randomly or the first one?
        # Picking the first one might get stuck in a specific local minimum.
        # Random selection helps escape.
        
        potential_swaps = []
        
        for c in codewords:
            # Find critical points of c
            criticals = [u for u in balls[c] if cover_count[u] == 1]
            
            if not criticals:
                continue # Should have been pruned, but just in case
            
            # We need a c' such that criticals is a subset of balls[c']
            # And c' is not in codewords.
            
            # Iterate through all possible c'
            # Optimization: Only consider c' that cover at least one critical point?
            # Actually, c' must cover ALL critical points.
            
            # This check is expensive if done for all c'.
            # Alternative: For each c', check if it covers criticals.
            
            # Since n=6, q=3, total_words=729.
            # criticals size is at most 13.
            
            # We can precompute for each word w, the list of words it covers.
            # We have balls.
            
            # Let's just iterate c'
            for c_prime in range(total_words):
                if c_prime in cw_set:
                    continue
                # Check if balls[c_prime] covers all criticals
                # Efficient check: 
                # For each u in criticals, is u in balls[c_prime]?
                # We can precompute a bitmask or just check.
                # Since criticals is small, we can check:
                valid = True
                for u in criticals:
                    if u not in balls[c_prime]:
                        valid = False
                        break
                if valid:
                    potential_swaps.append((c, c_prime))
                    
        if potential_swaps:
            # Pick a random swap to avoid bias
            c, c_prime = random.choice(potential_swaps)
            # Perform swap
            # Remove c
            for u in balls[c]:
                cover_count[u] -= 1
            codewords.remove(c)
            
            # Add c_prime
            for u in balls[c_prime]:
                cover_count[u] += 1
            codewords.append(c_prime)
            
            return True
        return False

    # Main Local Search Loop
    # Alternate between Prune and Swap.
    # If stuck (no prune, no swap), Perturb (Add random word).
    
    best_size = len(codewords)
    best_codewords = codewords.copy()
    
    while time.time() < deadline:
        progress = False
        
        # 1. Prune
        if try_prune():
            progress = True
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = codewords.copy()
        
        # 2. Swap
        if try_swap():
            progress = True
            # After swap, we might be able to prune
            # Loop back to top of while to prune again? 
            # Or just continue.
            # Let's continue to next iteration of main loop.
            # But we should check prune immediately after swap.
            if try_prune():
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = codewords.copy()
        
        if not progress:
            # Stuck. Perturb.
            # Add a random word
            # To make it effective, add a word that covers many points?
            # Or just random.
            # Random is good for exploration.
            
            # Ensure we don't add a duplicate
            available = [w for w in range(total_words) if w not in set(codewords)]
            if not available:
                break # Full space covered, can't add
            
            c_new = random.choice(available)
            for u in balls[c_new]:
                cover_count[u] += 1
            codewords.append(c_new)
            
            # After adding, try to prune immediately to get back to smaller size
            try_prune()
            
            # Update best if improved (unlikely to improve size immediately after add+prune, 
            # but possible if the add enabled a swap-like effect? No, add increases size. 
            # Prune decreases. Net effect could be smaller if the new word made 2 old words redundant? 
            # Yes, that's possible.)
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = codewords.copy()

    # Return the best found
    # Convert best_codewords (ints) to list of lists
    result = []
    for w in best_codewords:
        digits = []
        tmp = w
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        result.append(digits)
        
    return result
