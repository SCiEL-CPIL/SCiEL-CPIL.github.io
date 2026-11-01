"""
Ternary covering code via Simulated Annealing with bitset coverage tracking.
Strategy:
1. Generate a diverse initial solution using randomized greedy set cover.
2. Apply Simulated Annealing:
   - Moves: Insert, Delete, Swap (random pair), 2-opt (swap two codewords).
   - Cost function: Number of codewords + penalty for uncovered words.
   - Acceptance: Metropolis criterion based on temperature.
3. Rapidly cool down to find a deep local minimum.
4. Final local search (strict) to polish the result.
"""

import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    N = q ** n
    if N <= 0:
        return []
    
    # Precompute all words
    all_words = np.zeros((N, n), dtype=np.int32)
    for idx in range(N):
        rem = idx
        for j in range(n - 1, -1, -1):
            all_words[idx, j] = rem % q
            rem //= q
            
    # Precompute Hamming balls as bitsets
    balls = [0] * N
    if radius == 1:
        # Fast generation for radius 1
        for c in range(N):
            mask = 1 << c
            digits = [0] * n
            rem = c
            for j in range(n - 1, -1, -1):
                digits[j] = rem % q
                rem //= q
            for pos in range(n):
                orig = digits[pos]
                for new_val in range(q):
                    if new_val == orig:
                        continue
                    delta = (new_val - orig) * (q ** (n - 1 - pos))
                    new_idx = c + delta
                    if 0 <= new_idx < N:
                        mask |= (1 << new_idx)
            balls[c] = mask
    else:
        # Fallback for general radius
        def hamming_dist(i, j):
            d = 0
            a = i
            b = j
            for _ in range(n):
                if (a % q) != (b % q):
                    d += 1
                a //= q
                b //= q
            return d
        for c in range(N):
            mask = 0
            for w in range(N):
                if hamming_dist(c, w) <= radius:
                    mask |= (1 << w)
            balls[c] = mask

    full_mask = (1 << N) - 1
    
    # Helper to calculate cost
    # Cost = len(code) + lambda * uncovered_count
    # We want to minimize cost.
    # Uncovered count can be tracked via bitset.
    
    def get_initial_code():
        # Randomized Greedy Set Cover
        uncovered = full_mask
        code = []
        # Shuffle candidates to diversify
        candidates = list(range(N))
        random.shuffle(candidates)
        
        while uncovered and time.time() < deadline - 10:
            best_idx = -1
            max_new = 0
            # To speed up, we don't check all N every time if we have many candidates
            # But N is small (2187), so we check all or a subset.
            # Let's check all for robustness in initial phase.
            for c in candidates:
                if c in code:
                    continue
                new_cov = balls[c] & uncovered
                cnt = new_cov.bit_count()
                if cnt > max_new:
                    max_new = cnt
                    best_idx = c
            
            if best_idx == -1 or max_new == 0:
                break
                
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
            
            # Early exit if time is tight
            if time.time() > deadline - 20:
                break
                
        # If not fully covered, force add random codewords to ensure validity for SA start
        # Actually, SA can handle invalid states if we penalize them.
        # But it's better to start valid or near-valid.
        if uncovered:
             # Add random codewords to cover remaining
             missing = uncovered
             while missing and time.time() < deadline - 10:
                 # Pick a random uncovered word
                 lsb = missing & -missing
                 w = lsb.bit_length() - 1
                 # Find a codeword covering w
                 # Simple scan
                 found = False
                 for c in range(N):
                     if (balls[c] & (1 << w)) and c not in code:
                         code.append(c)
                         uncovered &= ~balls[c]
                         missing &= ~balls[c]
                         found = True
                         break
                 if not found:
                      # Should not happen if balls are correct, but safety break
                      break
                 
        return code

    # Initial Solution
    best_code = get_initial_code()
    if not best_code:
        # Fallback: single word? No, must cover all.
        # If time ran out in init, just return empty or best effort?
        # Better to return a valid cover if possible.
        # Let's try a quick greedy again if we are here.
        uncovered = full_mask
        while uncovered and time.time() < deadline - 5:
             best_idx = -1
             max_new = 0
             for c in range(N):
                 if c in best_code: continue
                 new_cov = balls[c] & uncovered
                 cnt = new_cov.bit_count()
                 if cnt > max_new:
                     max_new = cnt
                     best_idx = c
             if best_idx == -1: break
             best_code.append(best_idx)
             uncovered &= ~balls[best_idx]
             
        if uncovered:
             # Force validity
             missing = uncovered
             while missing:
                 lsb = missing & -missing
                 w = lsb.bit_length() - 1
                 for c in range(N):
                     if (balls[c] & (1 << w)) and c not in best_code:
                         best_code.append(c)
                         missing &= ~balls[c]
                         break
                 if not (missing & (1<<w)):
                     missing ^= lsb
                 else:
                     break # Safety

    # State for SA
    current_code = best_code.copy()
    current_cov = 0
    for c in current_code:
        current_cov |= balls[c]
        
    # Cost function
    # Primary: Size. Secondary: Uncovered.
    # We want to minimize Size.
    # If Uncovered > 0, add large penalty.
    PENALTY = 1000
    
    def calc_cost(code, cov):
        uncovered_count = (full_mask ^ cov).bit_count()
        return len(code) + PENALTY * uncovered_count

    current_cost = calc_cost(current_code, current_cov)
    best_cost = current_cost
    best_code_sa = current_code.copy()
    
    # Simulated Annealing Parameters
    T_start = 10.0
    T_end = 0.01
    cooling_rate = 0.9995  # Tune this to fit time budget
    max_iter_per_T = 100
    
    # Estimate total iterations
    # Time budget remaining
    time_remaining = deadline - time.time()
    # Assume ~1ms per iteration (conservative for Python with bit ops on 2187 bits)
    # Actually, bit ops on 2187 bits are fast. Let's aim for ~50k-100k iterations.
    
    T = T_start
    iters = 0
    total_iters = 0
    
    while T > T_end and time.time() < deadline - 5:
        # Do some iterations at this temperature
        for _ in range(max_iter_per_T):
            if time.time() > deadline - 5:
                break
            
            # Generate Move
            move_type = random.random()
            
            new_code = current_code.copy()
            new_cov = current_cov
            
            # 1. Delete (10%)
            if move_type < 0.10 and len(new_code) > 1:
                i = random.randint(0, len(new_code) - 1)
                c = new_code.pop(i)
                # Check if c is unique for any word
                # If we remove c, do we lose coverage?
                # We need to know if any word in balls[c] has count 1.
                # We don't track counts efficiently in SA loop for speed.
                # Instead, we just compute new_cov and check validity via bitset?
                # new_cov = (current_cov & ~balls[c]) is WRONG.
                # Coverage is OR. Removing c means:
                # new_cov = OR of balls[x] for x in new_code.
                # This is expensive to recompute from scratch.
                # Optimization: We can check if balls[c] contains any word NOT covered by others.
                # This requires counts.
                # Alternative: Just compute new_cov by OR-ing all? O(N * code_size).
                # For N=2187, code_size~200, this is 400k ops. Too slow for 100k iters.
                
                # Let's stick to Moves that don't require full recompute or use precomputed counts.
                # Actually, let's switch strategy:
                # Use "Swap" and "Insert" primarily, which are cheaper to evaluate if we track counts?
                # Or, use a simpler heuristic:
                # Try to remove c. If it breaks coverage, reject immediately?
                # To check if it breaks coverage, we need to know if any w in balls[c] is ONLY covered by c.
                # This is equivalent to: (balls[c] & ~OR(balls[others])) != 0.
                # OR(balls[others]) = current_cov ^ (bits only in balls[c])? No.
                
                # Let's use a different move set for SA that is cheaper:
                # 1. Swap two codewords.
                # 2. Replace one codeword with another.
                # 3. Add one codeword.
                # 4. Remove one codeword (only if we can verify it's redundant quickly).
                
                # To verify redundancy quickly, we need counts.
                # Let's maintain counts array.
                # But updating counts on every move is O(ball_size) ~ 1+7*2 = 15 ops. Very fast.
                
                # Let's refactor to maintain counts.
                pass # Skip this iteration if we didn't implement counts. 
                # Actually, let's just implement the counts maintenance below.
                
            # Reset for correct logic implementation below
            # We will implement the move logic properly in the main loop below.
            # The 'pass' above was a placeholder. Let's do it right.
            
            # --- CORRECTED MOVE LOGIC WITH COUNTS ---
            # We need counts array.
            # Let's initialize counts outside the loop.
            
            break # Break inner loop to restart with proper structure? No, let's just write the code correctly.
            
    # --- RE-IMPLEMENTATION OF SA LOOP WITH COUNTS ---
    
    # Initialize counts for current_code
    counts = np.zeros(N, dtype=np.int32)
    for c in current_code:
        mask = balls[c]
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            counts[idx] += 1
            m ^= lsb
            
    # Helper to apply move and return new state or None if invalid structure
    def apply_move(code, counts, cov, move):
        """
        move: ('add', c) or ('remove', c) or ('swap', c_old, c_new)
        Returns: (new_code, new_counts, new_cov, valid)
        """
        new_code = code.copy()
        # We will update counts and cov in place on copies for simplicity, 
        # but for speed we should mutate and rollback.
        # Given Python overhead, copying counts (2187 ints) is ~microseconds.
        # Let's copy.
        new_counts = counts.copy()
        new_cov = cov
        
        if move[0] == 'add':
            c = move[1]
            if c in new_code:
                return None
            new_code.append(c)
            mask = balls[c]
            # Update counts
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                new_counts[idx] += 1
                m ^= lsb
            new_cov |= mask
            
        elif move[0] == 'remove':
            c = move[1]
            if c not in new_code:
                return None
            # Check if removal is safe (no unique words)
            mask = balls[c]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if new_counts[idx] == 1:
                    return None # Would break coverage
                m ^= lsb
            
            # Safe to remove
            new_code.remove(c)
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                new_counts[idx] -= 1
                m ^= lsb
            # Update cov: if any word in mask now has count 0, it becomes uncovered
            # But we checked count > 1, so count >= 1. So cov doesn't change?
            # Wait, if count was 2, it becomes 1. Still covered.
            # So new_cov remains same?
            # Yes, if no word drops to 0, cov is unchanged.
            
        elif move[0] == 'swap':
            c_old, c_new = move[1], move[2]
            if c_old not in new_code or c_new in new_code or c_old == c_new:
                return None
            
            idx_old = new_code.index(c_old)
            
            mask_old = balls[c_old]
            mask_new = balls[c_new]
            
            # Check validity of swap:
            # Words in mask_old \ mask_new must have count >= 2 (so they don't drop to 0)
            words_to_check = mask_old & ~mask_new
            m = words_to_check
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if new_counts[idx] < 2:
                    return None
                m ^= lsb
            
            # Apply swap
            new_code[idx_old] = c_new
            
            # Update counts
            # Remove c_old
            m = mask_old
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                new_counts[idx] -= 1
                m ^= lsb
                
            # Add c_new
            m = mask_new
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                new_counts[idx] += 1
                m ^= lsb
                
            # Update cov
            # New words covered by c_new that were not covered by c_old?
            # If a word was uncovered (count 0) and is in mask_new, it becomes covered.
            # Can a word be uncovered? Only if it was not in current_cov.
            # current_cov is OR of all balls.
            # If w is in mask_new, and w was not in current_cov, then w was not covered by any other codeword.
            # So count[w] was 0.
            # After adding c_new, count[w] becomes 1.
            # So new_cov |= mask_new is correct.
            # But wait, did we uncover anything?
            # We checked mask_old \ mask_new has count >= 2.
            # So no word in mask_old drops to 0.
            # Words in intersection: count unchanged (or -1+1).
            # So no word drops to 0.
            # So new_cov = current_cov | mask_new.
            new_cov |= mask_new
            
        return new_code, new_counts, new_cov, True

    T = T_start
    while T > T_end and time.time() < deadline - 5:
        for _ in range(max_iter_per_T):
            if time.time() > deadline - 5:
                break
            
            move_type = random.random()
            move = None
            
            if move_type < 0.1 and len(current_code) > 1:
                # Remove
                c = random.choice(current_code)
                move = ('remove', c)
            elif move_type < 0.2:
                # Add
                # Pick a random word not in code? Or random word?
                # Picking a word that is "hard to cover" is better.
                # For simplicity, pick random c.
                c = random.randint(0, N - 1)
                if c not in current_code:
                    move = ('add', c)
            else:
                # Swap
                if len(current_code) > 1:
                    c_old = random.choice(current_code)
                    c_new = random.randint(0, N - 1)
                    if c_new not in current_code and c_new != c_old:
                        move = ('swap', c_old, c_new)
            
            if move is None:
                continue
                
            result = apply_move(current_code, counts, current_cov, move)
            if result is None:
                continue
                
            new_code, new_counts, new_cov, _ = result
            new_cost = calc_cost(new_code, new_cov)
            
            delta_cost = new_cost - current_cost
            
            # Metropolis Acceptance
            if delta_cost <= 0:
                accept = True
            else:
                # P(accept) = exp(-delta / T)
                # If T is small, this is very small.
                try:
                    accept_prob = np.exp(-delta_cost / T)
                except OverflowError:
                    accept_prob = 0.0
                accept = random.random() < accept_prob
                
            if accept:
                current_code = new_code
                counts = new_counts
                current_cov = new_cov
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_code_sa = current_code.copy()
                    
        T *= cooling_rate
        
    # Final Local Search (Strict) to polish
    # Try to remove any redundant codeword
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Rebuild counts for best_code_sa? 
        # We should run this on the current best state.
        # Let's use a local search on best_code_sa.
        
        # Build counts for best_code_sa
        b_counts = np.zeros(N, dtype=np.int32)
        b_cov = 0
        for c in best_code_sa:
            b_cov |= balls[c]
            mask = balls[c]
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                b_counts[idx] += 1
                m ^= lsb
                
        # Try removals
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(len(best_code_sa) - 1, -1, -1):
                c = best_code_sa[i]
                mask = balls[c]
                # Check if redundant
                m = mask
                redundant = True
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if b_counts[idx] == 1:
                        redundant = False
                        break
                    m ^= lsb
                    
                if redundant:
                    # Remove
                    best_code_sa.pop(i)
                    m = mask
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        b_counts[idx] -= 1
                        m ^= lsb
                    # b_cov doesn't change
                    changed = True
                    improved = True
                    break # Restart scan
        
        # Try swaps (1-for-1) to enable removals?
        # This is complex. Let's just do removals.
        
    # Final validity check
    final_cov = 0
    for c in best_code_sa:
        final_cov |= balls[c]
        
    if final_cov != full_mask:
        # Repair
        missing = full_mask & ~final_cov
        while missing and time.time() < deadline - 1:
            lsb = missing & -missing
            w = lsb.bit_length() - 1
            for c in range(N):
                if (balls[c] & (1 << w)) and c not in best_code_sa:
                    best_code_sa.append(c)
                    final_cov |= balls[c]
                    missing &= ~balls[c]
                    break
            if not (missing & (1<<w)):
                missing ^= lsb
            else:
                break

    return [list(all_words[i]) for i in best_code_sa]
