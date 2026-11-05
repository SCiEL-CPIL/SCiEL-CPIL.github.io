import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit (or 150s)
    
    if q != 3 or n != 8 or radius != 1:
        return _greedy_construct(q, n, radius, deadline)
    
    total_words = 3 ** 8  # 6561
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % 3)
            x //= 3
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute balls as bitmasks
    balls = [0] * total_words
    q_powers = [1] * n
    for i in range(n - 2, -1, -1):
        q_powers[i] = q_powers[i + 1] * 3
        
    for i in range(total_words):
        d = digits_of[i]
        mask = 1 << i
        for pos in range(n):
            orig_digit = d[pos]
            for new_digit in range(3):
                if new_digit == orig_digit:
                    continue
                delta = (new_digit - orig_digit) * q_powers[pos]
                idx = i + delta
                mask |= (1 << idx)
        balls[i] = mask
        
    all_mask = (1 << total_words) - 1
    
    # --- STEP 1: Construct Tetracode and its product ---
    # Tetracode is a (36, 4, 1) covering code.
    # We construct it explicitly.
    # A known Tetracode consists of all words (a,b,c,d) such that:
    # 1. a+b+c+d = 0 mod 3
    # 2. a+2b+c+2d = 0 mod 3
    # Wait, that's a linear code of size 3^(4-2) = 9. That's the simplex code or similar?
    # No, the Tetracode is a nonlinear code.
    # Reference: "Tetracode" in coding theory often refers to the (36, 4, 1) code.
    # Actually, let's just use a greedy set cover to find a Tetracode-like structure,
    # or better, use the known construction for K_3(8,1)=486 directly if possible.
    
    # Alternative known construction for K_3(8,1) = 486:
    # It is often constructed as a union of two (243, 8, 1) codes? No.
    # 486 = 2 * 3^5.
    # A perfect code of length 4 doesn't exist (Hamming bound 9, but actual K=9? No, K_3(4,1)=9 is a linear code?
    # Let's check K_3(4,1). Hamming bound: 3^4 / (1+4*2) = 9.
    # A linear code of length 4, dimension 2, parity check matrix H = [1 1 1 1; 0 1 2 0]?
    # If a perfect code exists, K_3(4,1) = 9.
    # Let's verify if a (9, 4, 1) code exists.
    # Yes, the simplex code [4, 2, 2] has minimum distance 2, so it covers radius 0.
    # We need covering radius 1.
    # The code C = { (a,b,a,b) }? No.
    # Let's try to construct a (9, 4, 1) code.
    # If C is a code of size 9, and covers 81 words with radius 1 (ball size 9), it must be perfect.
    # A perfect ternary code of length 4?
    # Hamming bound is tight. Existence is known for certain lengths.
    # Actually, for q=3, perfect codes exist for n = (3^k - 1)/(3 - 1) - 1? No.
    # Ternary perfect codes exist for n = (3^k - 1)/2?
    # k=2: (9-1)/2 = 4. Yes! There is a perfect ternary code of length 4.
    # It is equivalent to the simplex code? No, simplex is linear distance 2.
    # The perfect code is nonlinear.
    
    # Construction of the perfect ternary code of length 4 (size 9):
    # One simple construction:
    # Let S = {0, 1, 2}.
    # Code C:
    # 0000
    # 1110, 1101, 1011, 0111
    # 2220, 2201, 2021, 0222
    # Let's check distance.
    # 0000 to 1110: dist 3.
    # 1110 to 1101: dist 2.
    # Does it cover everything?
    # Ball(0000, 1) = {0000, 1000, 2000, 0100, 0200, 0010, 0020, 0001, 0002}
    # Ball(1110, 1) = {1110, 0110, 2110, 1010, 1210, 1100, 1120, 1111, 1112}
    # Overlap?
    # 0000 vs 1110: 0000 in Ball(0000). 1110 in Ball(1110).
    # Is there overlap between Ball(0000) and Ball(1110)?
    # 1000 is in Ball(0000). Is 1000 in Ball(1110)?
    # dist(1000, 1110) = 1 (pos 1). Yes!
    # So they overlap. This is not a perfect code.
    
    # Actually, for q=3, n=4, K_3(4,1) is NOT 9.
    # Lower bound is 9. Upper bound?
    # It turns out K_3(4,1) = 9 is NOT possible because the balls would have to be disjoint to cover 81 with 9 balls of size 9.
    # If they are not disjoint, we need more.
    # It is known that K_3(4,1) = 9? No, likely larger.
    # Actually, looking up "Ternary covering code K(4,1) 3":
    # The value is 9? Or 10?
    # If K_3(4,1)=9, then K_3(8,1) <= 9*9 = 81. But we know K_3(8,1)=486.
    # So K_3(4,1) is definitely larger.
    
    # Okay, scratch the Tetracode perfect code idea.
    
    # Let's stick to a robust heuristic that is structurally different: **Iterated Local Search (ILS)** with a **Greedy Randomized Adaptive Search Procedure (GRASP)** style initialization.
    # The key improvement over the current SA is better initialization and more aggressive diversification.
    
    # 1. Initialize with multiple randomized greedy codes.
    # 2. Keep the best one.
    # 3. Apply ILS: Perturb, Local Search, Accept/Reject.
    
    # But we have a hard time limit.
    # The current program's SA is slow because it does random swaps.
    # Let's use a **Greedy Pruning with Lookahead** approach.
    
    # Strategy:
    # 1. Fast Greedy to get a valid code C0.
    # 2. Prune C0 using a priority queue:
    #    - For each word in C0, calculate "removability" (number of words ONLY covered by it, i.e., unique coverage).
    #    - If unique coverage is 0, remove it.
    #    - This is exact for the current set.
    # 3. To escape local minima, perform "Sweeps":
    #    - Remove a set of k words, then re-cover the uncovered parts with new words.
    
    # Let's implement a "Greedy + Greedy Prune" loop.
    
    # Step 1: Initial Greedy
    # We'll do 3 different random greedy runs and pick the smallest.
    
    best_code = None
    best_size = float('inf')
    
    # Precompute reverse balls (who covers me) to speed up pruning?
    # reverse_balls[i] = set of j such that i in balls[j].
    # This is too much memory/time for 6561 * 9.
    # 6561 * 9 is ~60k. Storing as lists is fine.
    
    # Actually, for pruning, we need to know if a word is "essential".
    # A word c is essential if there exists a word x in balls[c] such that no other c' in Code (c' != c) covers x.
    # i.e., x is in balls[c] and x is NOT in union of balls[c'] for c' in Code \ {c}.
    
    # Efficient way:
    # Count for each word x, how many codewords cover it.
    # If count[x] == 1, then x is uniquely covered by its covering codeword.
    # If a codeword c covers any x with count[x] == 1, then c is essential.
    # If a codeword c covers NO x with count[x] == 1, then c is removable.
    
    def _greedy_randomized(balls, all_mask, deadline, random_factor=0.1):
        uncovered = all_mask
        code = set()
        total_words = len(balls)
        
        # Pre-compute ball sizes? They are all 9.
        
        while uncovered and time.time() < deadline:
            candidates = []
            # Find candidates that cover at least one uncovered word.
            # Optimization: Only check words that have high overlap with uncovered?
            # Just scan all.
            for i in range(total_words):
                if i in code:
                    continue
                gain = (balls[i] & uncovered).bit_count()
                if gain > 0:
                    # Add some randomness
                    # If gain is high, likely to be selected.
                    # Use "adaptive" probability.
                    # Simple: if gain == max_gain, add to candidates.
                    candidates.append((gain, i))
            
            if not candidates:
                break
                
            # Find max gain
            max_gain = max(g for g, _ in candidates)
            
            # Select from top performers
            # Filter to those with gain >= max_gain * (1 - random_factor)
            threshold = max_gain * (1 - random_factor)
            top_candidates = [i for g, i in candidates if g >= threshold]
            
            choice = random.choice(top_candidates)
            code.add(choice)
            uncovered &= ~balls[choice]
            
        return code

    def _prune_code(code, balls, all_mask, deadline):
        # code is a set
        if not code:
            return code
            
        # Count coverage
        # cover_count[i] = number of codewords in code that cover i
        cover_count = [0] * len(balls)
        
        for c in code:
            mask = balls[c]
            # Iterate set bits in mask
            # This is faster than iterating all words if code is small?
            # Code size ~500. Ball size 9.
            # 500 * 9 = 4500 operations.
            # Iterating set bits:
            m = mask
            while m:
                # Isolate lowest set bit
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
                
        # Identify removable words
        # A word c is removable if for all x in balls[c], cover_count[x] > 1.
        # i.e., c has no unique coverage.
        
        removable = set()
        for c in code:
            m = balls[c]
            is_removable = True
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if cover_count[idx] == 1:
                    is_removable = False
                    break
                m ^= lsb
            if is_removable:
                removable.add(c)
                
        # Remove all removable words
        if removable:
            for c in removable:
                code.remove(c)
            # Recalculate cover_count?
            # Or just re-run the whole prune loop until stable.
            return _prune_code(code, balls, all_mask, deadline)
            
        return code

    # Run multiple greedy initializations
    num_starts = 3
    start_budget = deadline - 60.0 # Leave 60s for pruning/search
    
    for _ in range(num_starts):
        if time.time() > start_budget:
            break
        code_set = _greedy_randomized(balls, all_mask, time.time() + 2.0, random_factor=0.2)
        if not code_set:
            continue
        # Prune
        code_set = _prune_code(code_set, balls, all_mask, time.time() + 5.0)
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = set(code_set)
            
    if best_code is None:
        # Fallback
        best_code = _fast_greedy(balls, all_mask, deadline - 10)
        best_code = set(best_code)
        best_size = len(best_code)
        
    # Now, use the remaining time for "Local Search" or "Perturbation"
    # We will perform "Swap" moves:
    # 1. Try to remove a word. If invalid, identify the "deficit" words.
    # 2. Try to add a word that fixes the deficit and allows removal of another word?
    # Too complex.
    
    # Let's use **Iterated Local Search (ILS)** on the set of codewords.
    # State: Code Set C.
    # Move:
    #   a) Remove c from C.
    #   b) If invalid, let U be uncovered.
    #   c) Find c' not in C that covers the most of U.
    #   d) If C - {c} + {c'} is valid, accept.
    #   e) Else, reject (or try another c').
    
    # This is a "2-opt" style move.
    
    # To speed up, we can restrict search.
    
    # But we already have a strong pruning step.
    # The gap between Greedy+Prune and Optimal (486) is likely small.
    # Greedy usually gets ~500.
    
    # Let's try to improve the pruning by allowing "temporary invalidity" in a controlled way?
    # No, strict validity is easier to maintain with bitmasks.
    
    # Let's just run a tight loop of:
    # 1. Pick random c in Code.
    # 2. Remove c.
    # 3. If valid, keep removed.
    # 4. If invalid, find best c' to add.
    #    - Calculate U (uncovered).
    #    - Find c' maximizing overlap with U.
    #    - If C - {c} + {c'} is valid, swap.
    #    - Else, revert.
    
    # We need to be careful with time.
    
    current_code = set(best_code)
    current_size = len(current_code)
    
    # Precompute cover_mask for current code
    cover_mask = 0
    for c in current_code:
        cover_mask |= balls[c]
        
    # Helper to get uncovered
    def get_uncovered(cm):
        return all_mask & ~cm
        
    # Helper to find best cover for a set of uncovered words
    def find_best_cover(uncovered, exclude_set, deadline):
        best_idx = -1
        best_gain = 0
        # We only care about words in `uncovered`
        # Iterate candidates?
        # To be fast, we can sample or iterate all.
        # Iterating all 6561 is fast enough if done few times.
        for i in range(len(balls)):
            if i in exclude_set:
                continue
            gain = (balls[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        return best_idx, best_gain

    # Main Loop
    no_improve_counter = 0
    max_no_improve = 500 # Tune this?
    
    while time.time() < deadline - 1.0:
        # Try a removal
        c = random.choice(list(current_code))
        
        # Tentative removal
        new_cover = cover_mask & ~balls[c]
        
        if new_cover == all_mask:
            # Valid removal
            current_code.remove(c)
            cover_mask = new_cover
            current_size -= 1
            no_improve_counter = 0
            if current_size < best_size:
                best_size = current_size
                best_code = set(current_code)
        else:
            # Invalid removal
            uncovered = new_cover ^ all_mask # Wait, ~new_cover & all_mask
            
            # Find best replacement
            c_prime, gain = find_best_cover(uncovered, current_code - {c}, time.time() + 0.01)
            
            if c_prime != -1:
                # Check if swap is valid
                # new_cover already excludes c.
                # Add c_prime
                swap_cover = new_cover | balls[c_prime]
                if swap_cover == all_mask:
                    # Valid swap
                    current_code.remove(c)
                    current_code.add(c_prime)
                    cover_mask = swap_cover
                    # Size unchanged, but we explored a new state
                    # Does this help future removals? Yes.
                    no_improve_counter += 1
                else:
                    no_improve_counter += 1
            else:
                no_improve_counter += 1
        
        if no_improve_counter > max_no_improve:
            # Perturbation: Add 2-3 random words, then Prune?
            # Or just reset to best_code and try again with different random seed?
            # Let's try a "Big Jump":
            # Remove 5% of code randomly, then re-cover?
            
            # Simpler: Just reset to best and change strategy?
            # For now, just keep going. The random nature should eventually find something.
            
            # Actually, let's implement a "Restart" if stuck.
            # If we haven't improved best_size in a long time, reset to best_code.
            if time.time() % 1.0 < 0.01: # Check every second roughly
                if current_size > best_size + 10:
                    # We are much worse than best. Reset.
                    current_code = set(best_code)
                    cover_mask = 0
                    for c in current_code:
                        cover_mask |= balls[c]
                    current_size = len(current_code)
                    no_improve_counter = 0

    # Final Prune
    if time.time() < deadline - 0.5:
        best_code = _prune_code(best_code, balls, all_mask, deadline)
        
    # Ensure validity of best_code
    final_cover = 0
    for c in best_code:
        final_cover |= balls[c]
    if final_cover != all_mask:
        uncovered = all_mask & ~final_cover
        while uncovered:
            i = random.randint(0, total_words - 1)
            if (uncovered >> i) & 1:
                best_code.add(i)
                final_cover |= balls[i]
                uncovered &= ~balls[i]
                break

    return [list(digits_of[i]) for i in best_code]

def _fast_greedy(balls, all_mask, deadline):
    uncovered = all_mask
    code = []
    total_words = len(balls)
    
    while uncovered and time.time() < deadline:
        best_idx = -1
        max_cover = 0
        
        for i in range(total_words):
            if i in code:
                continue
            gain = (balls[i] & uncovered).bit_count()
            if gain > max_cover:
                max_cover = gain
                best_idx = i
                
        if max_cover == 0:
            break
            
        code.append(best_idx)
        uncovered &= ~balls[best_idx]
        
    return code

def _greedy_construct(q, n, radius, deadline):
    # Fallback for non-standard cases
    total_words = q ** n
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    balls = [0] * total_words
    q_powers = [1] * n
    for i in range(n - 2, -1, -1):
        q_powers[i] = q_powers[i + 1] * q
        
    for i in range(total_words):
        d = digits_of[i]
        mask = 1 << i
        if radius == 1:
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * q_powers[pos]
                    idx = i + delta
                    mask |= (1 << idx)
        else:
            # General radius
            for r in range(1, radius + 1):
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
                        mask |= (1 << idx)
        balls[i] = mask
        
    all_mask = (1 << total_words) - 1
    code = _fast_greedy(balls, all_mask, deadline)
    
    if not code:
        return [list(range(q)) for _ in range(1)]
        
    # Ensure validity
    cover = 0
    for c in code:
        cover |= balls[c]
        
    if cover != all_mask:
        uncovered = all_mask & ~cover
        for i in range(total_words):
            if (uncovered >> i) & 1:
                code.append(i)
                cover |= balls[i]
                
    return [list(digits_of[i]) for i in code]

