import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    # Parameters specific to the task
    # The task specifies q=3, n=8, radius=1
    # We handle the general case but optimize for the known instance.
    
    total_words = q ** n
    
    # 1. Precomputation
    # Map integer index to tuple of digits
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute Hamming distance matrix? No, too big (6561^2).
    # Instead, precompute the "ball" of each word as a list of indices.
    # For q=3, n=8, radius=1, ball size = 1 + 8*(3-1) = 17.
    # Total size 6561 * 17 ~ 111k entries. Very small.
    
    # We will use a set of indices for the current code to check membership quickly.
    # And a boolean array (or bitset) for covered words.
    
    # Precompute neighbors for fast ball expansion
    # neighbors[i] = list of j such that dist(i, j) <= radius
    neighbors = [None] * total_words
    q_powers = [1] * n
    for i in range(n - 2, -1, -1):
        q_powers[i] = q_powers[i + 1] * q
        
    for i in range(total_words):
        d = digits_of[i]
        ball = set([i])
        if radius == 1:
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * q_powers[pos]
                    idx = i + delta
                    ball.add(idx)
        else:
            # General radius (slower, but n=8 is small)
            # For radius > 1, we'd need combinations. 
            # Given the task is K_3(8,1), we assume radius=1 optimization is key.
            # But let's keep it general just in case.
            for r in range(1, radius + 1):
                import itertools
                for pos_comb in itertools.combinations(range(n), r):
                    for variation in itertools.product(range(q), repeat=r):
                        is_diff = False
                        delta = 0
                        for k, pos in enumerate(pos_comb):
                            new_val = variation[k]
                            if new_val != d[pos]:
                                is_diff = True
                                delta += (new_val - d[pos]) * q_powers[pos]
                        if not is_diff:
                            continue
                        idx = i + delta
                        ball.add(idx)
        neighbors[i] = list(ball)

    # 2. Initial Construction: Algebraic / Tetracode Based
    # The Tetracode is a ternary linear code of length 4, dimension 2, min dist 4.
    # Actually, the "Tetracode" often refers to the [4,2,4] ternary code.
    # The ternary Golay code is [12,6,6].
    # A known construction for K_3(8,1) uses the direct sum or product of smaller codes.
    # Specifically, K_3(8,1) can be bounded/constructed using the "doubling" or "product" method.
    # However, the best known 486 is not a simple direct sum of tetracodes (which would give larger codes).
    # The 486 code is likely a shortened or punctured Golay code or similar.
    
    # Strategy: Start with a random greedy code, but initialize with a structured subset
    # to provide a good starting point for local search.
    # Or, simply start with a good greedy and run ILS.
    
    def greedy_initial():
        covered = [False] * total_words
        code = []
        remaining = list(range(total_words))
        random.shuffle(remaining) # Randomize start to break symmetry? 
        # Actually, deterministic greedy with tie-breaking by index is faster and reproducible.
        # Let's use a "max marginal gain" greedy.
        
        # To speed up, we don't recompute all gains from scratch.
        # But for n=8, 6561 words, and ~400 codewords, 
        # Greedy is O(|Code| * |Space| * BallSize).
        # 400 * 6561 * 17 ~ 44M ops. Might be too slow in Python if not optimized.
        
        # Optimization: Use numpy for vectorized coverage checks?
        # Or just efficient Python loops.
        
        # Let's try a simpler approach: 
        # 1. Pick a random word, add to code.
        # 2. Find word with max new coverage.
        
        covered_count = 0
        code_set = set()
        
        while covered_count < total_words and time.time() < deadline - 10.0:
            best_idx = -1
            max_gain = 0
            
            # Sample a subset of candidates if time is tight? 
            # No, let's try full scan first. 
            # If too slow, we'll see.
            
            for i in range(total_words):
                if i in code_set:
                    continue
                gain = 0
                for j in neighbors[i]:
                    if not covered[j]:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            
            if best_idx == -1 or max_gain == 0:
                break
                
            code.append(best_idx)
            code_set.add(best_idx)
            for j in neighbors[best_idx]:
                if not covered[j]:
                    covered[j] = True
                    covered_count += 1
                    
        return code, covered

    # Run initial greedy
    code, covered = greedy_initial()
    
    # 3. Iterated Local Search (ILS)
    # Objective: Minimize len(code)
    # Moves: 
    #   - Remove a codeword if it doesn't break coverage.
    #   - Swap a codeword with another if it allows removal of another codeword (2-opt).
    
    def is_valid_code(code):
        # Check if code covers all words
        # This is O(|Code| * BallSize)
        # We can maintain a count array for speed during search.
        pass

    # For efficiency, maintain `cover_count` array where cover_count[w] = number of codewords covering w.
    # A code is valid if min(cover_count) >= 1.
    # A codeword c can be removed if for all w in ball(c), cover_count[w] > 1.
    
    # Re-initialize cover_count from the greedy result
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code:
        for j in neighbors[c]:
            cover_count[j] += 1
            
    # Check validity
    if np.min(cover_count) < 1:
        # Greedy failed? Unlikely. But fix it.
        # Add uncovered words
        for w in range(total_words):
            if cover_count[w] == 0:
                code.append(w)
                for j in neighbors[w]:
                    cover_count[j] += 1
                # Note: This might make code larger, but valid.
        
    best_code = code[:]
    best_size = len(code)
    
    # ILS Loop
    current_code = code[:]
    current_cover_count = cover_count.copy()
    
    # Helper: Try to remove redundant codewords
    def prune(code, cover_count):
        removed_any = True
        while removed_any:
            removed_any = False
            for i in range(len(code)):
                c = code[i]
                # Check if c is redundant
                # c is redundant if all w in ball(c) have cover_count[w] > 1
                # Note: cover_count includes c itself.
                # So we need to check if cover_count[w] - 1 >= 1 for all w in ball(c)?
                # No, if we remove c, the new count is cover_count[w] - 1.
                # We need new count >= 1.
                # So cover_count[w] > 1 for all w in ball(c).
                
                is_redundant = True
                for w in neighbors[c]:
                    if current_cover_count[w] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove c
                    for w in neighbors[c]:
                        current_cover_count[w] -= 1
                    code.pop(i)
                    removed_any = True
                    break # Restart loop
        return code

    # Helper: Try a swap (remove c, add c_new)
    # This is expensive to check all pairs.
    # Heuristic: Pick a random c from code.
    # Pick a random c_new from space (or neighbors of c?).
    # If swap maintains validity (or improves "slack"), accept.
    
    # Better Move: "2-swap"
    # Remove c1, add c_new. Then prune.
    
    # Let's implement a simple ILS with random restarts and local search.
    
    rng = random.Random(42)
    
    for iteration in range(1000):
        if time.time() > deadline - 5.0:
            break
            
        # Local Search: Prune
        current_code = prune(current_code, current_cover_count)
        
        if len(current_code) < best_size:
            best_size = len(current_code)
            best_code = current_code[:]
            
        # If we reached a local minimum (no pruning possible), perturb
        # Perturbation: Remove a random codeword (making code invalid), then repair?
        # Or: Add a random codeword, then prune?
        # Or: Swap a random codeword with a random other word, then prune.
        
        # Let's try: Pick a random codeword c_old.
        # Pick a random word c_new (not in code).
        # Remove c_old, add c_new.
        # This might make the code invalid.
        # Then, we must repair by adding words until valid, then prune.
        
        # This "remove-add-repair" is expensive.
        
        # Alternative: "Kick" - Remove a random codeword. 
        # If valid, prune.
        # If invalid, we have a hole. We must fill it.
        
        # Let's try a different perturbation: 
        # Select a random codeword c.
        # Find a word c_new that is far from c but close to some uncovered area?
        # Too complex.
        
        # Simple Perturbation:
        # 1. Pick random c_old in current_code.
        # 2. Pick random c_new in [0, total_words).
        # 3. If c_new is in code, skip.
        # 4. Tentatively remove c_old, add c_new.
        # 5. Check validity. If valid, prune and accept.
        # 6. If invalid, reject.
        
        # This is slow because validity check is O(|Code| * BallSize).
        
        # Optimization: Use the cover_count array.
        # If we remove c_old, cover_count drops by 1 for ball(c_old).
        # If any of those becomes 0, it's invalid.
        # If valid, we add c_new, cover_count rises by 1 for ball(c_new).
        # Then we prune.
        
        if len(current_code) > 0:
            c_old = rng.choice(current_code)
            c_new = rng.randint(0, total_words - 1)
            
            if c_new in current_code:
                continue
                
            # Check if removing c_old keeps cover >= 1
            valid_remove = True
            for w in neighbors[c_old]:
                if current_cover_count[w] <= 1:
                    valid_remove = False
                    break
            
            if not valid_remove:
                # Can't remove c_old directly. 
                # Maybe swap with a word that covers the same "weak" points?
                # This is complex. Let's just try a different c_old or skip.
                # To avoid getting stuck, we can force a "repair" step.
                # But for now, let's just try another iteration.
                continue
                
            # Perform swap
            # Remove c_old
            for w in neighbors[c_old]:
                current_cover_count[w] -= 1
            current_code.remove(c_old)
            
            # Add c_new
            for w in neighbors[c_new]:
                current_cover_count[w] += 1
            current_code.append(c_new)
            
            # Prune
            current_code = prune(current_code, current_cover_count)
            
            if len(current_code) < best_size:
                best_size = len(current_code)
                best_code = current_code[:]
                
        # Occasionally, do a "random restart" of the local search state
        # by taking the best code found so far and re-initializing cover_count?
        # No, that's expensive.
        
        # If we are stuck (no improvement in last 100 iterations), 
        # we could try a different perturbation strategy.
        # But given time limits, simple ILS is best.

    # Final Verification and Return
    # Ensure best_code is valid
    # Re-verify cover_count for best_code just in case
    final_cover_count = np.zeros(total_words, dtype=np.int32)
    for c in best_code:
        for j in neighbors[c]:
            final_cover_count[j] += 1
            
    if np.min(final_cover_count) < 1:
        # Repair
        for w in range(total_words):
            if final_cover_count[w] == 0:
                best_code.append(w)
                for j in neighbors[w]:
                    final_cover_count[j] += 1
                # This might make it larger, but valid.
                # We should prune again if time permits.
                if time.time() < deadline - 2.0:
                    # Convert to list for prune
                    temp_code = best_code[:]
                    # Re-init cover count for temp
                    tc = np.zeros(total_words, dtype=np.int32)
                    for c in temp_code:
                        for j in neighbors[c]:
                            tc[j] += 1
                    temp_code = prune(temp_code, tc)
                    best_code = temp_code

    return [list(digits_of[i]) for i in best_code]

# To allow testing
if __name__ == "__main__":
    code = construct(3, 8, 1)
    print(f"Code size: {len(code)}")
