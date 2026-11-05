import time
import itertools
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin for 120s limit
    
    # The problem is specific: q=3, n=8, radius=1
    # K_3(8,1) is known to be 486.
    # Strategy:
    # 1. Construct the Tetracode C4 (size 36, length 4, radius 1).
    # 2. Construct the product code C4 x C4 (size 36*36 = 1296, length 8, radius 1).
    #    Wait, product of covering codes: If C1 covers (n1, q, r1) and C2 covers (n2, q, r2),
    #    then C1 x C2 covers (n1+n2, q, r1+r2).
    #    Tetracode covers (4, 3, 1).
    #    Product covers (8, 3, 2). This is too large a radius for our requirement (radius 1).
    #    However, the product code has size 1296. We need size ~486.
    #    Actually, there is a known construction for K_3(8,1) = 486.
    #    The Tetracode is a perfect code? No, K_3(4,1) = 36. 3^4 = 81. Ball size 1+4*2=9. 81/9 = 9.
    #    Wait, K_3(4,1) is actually 36? Let's check.
    #    Perfect code exists for n=4? Hamming bound: 3^4 / (1 + 4*2) = 81/9 = 9.
    #    So a perfect code of size 9 would exist if K_3(4,1)=9.
    #    But K_3(4,1) is actually 36? No, that's too big.
    #    Let's re-evaluate K_3(4,1).
    #    Lower bound: 81/9 = 9.
    #    Upper bound: ?
    #    Actually, the Tetracode is a code of length 4, size 36, radius 1?
    #    Let's check the literature. "Tetracode" usually refers to a specific code.
    #    In van Laarhoven et al., K_3(8,1) = 486.
    #    There is a construction: C = { (u, v) | u in C4, v in C4 }? No.
    #    Actually, 486 = 2 * 3^5.
    #    
    #    Let's use a different approach:
    #    Since we know the answer is 486, and greedy gets ~486-500,
    #    we can start with a large valid code and prune.
    #    
    #    A simpler valid code:
    #    We can use the "greedy" result from the current program as a starting point
    #    but optimize the pruning significantly.
    #    
    #    However, the directive asks for a MATERIAL DIFFERENCE.
    #    The current program does Greedy + Local Search.
    #    
    #    New Strategy: **Simulated Annealing on the Code Set**
    #    Instead of adding/removing one by one greedily, we define an energy function:
    #    E = |Code| + Penalty * (Number of Uncovered Words)
    #    
    #    Moves:
    #    1. Remove a random codeword.
    #    2. Add a random codeword.
    #    3. Swap a random codeword with a random other word.
    #    
    #    We start with a valid code (e.g., from a fast greedy run or a known construction).
    #    We cool down the temperature to minimize size while maintaining validity (Penalty -> Inf).
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: use standard greedy
        return _greedy_construct(q, n, radius, deadline)
        
    # Precomputation for q=3, n=8
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
    
    # Step 1: Get a valid initial code using a fast greedy
    # We use a slightly randomized greedy to get a good starting point, 
    # or just the standard greedy.
    # To be fast, we'll do a standard greedy first.
    code = _fast_greedy(balls, all_mask, deadline - 30)
    
    if code is None or len(code) == 0:
        # Emergency fallback
        return [[0]*8]
        
    # Convert to set for fast lookup
    code_set = set(code)
    cover_mask = 0
    for c in code_set:
        cover_mask |= balls[c]
        
    # Ensure it's valid
    if cover_mask != all_mask:
        # Fix validity
        uncovered = all_mask & ~cover_mask
        while uncovered:
            # Pick lex smallest uncovered
            # This is O(N^2) worst case, but should be small
            for i in range(total_words):
                if (uncovered >> i) & 1:
                    code_set.add(i)
                    cover_mask |= balls[i]
                    uncovered &= ~balls[i]
                    break
            if time.time() > deadline - 10:
                break
        if cover_mask != all_mask:
             # If we still can't cover, add all uncovered
             for i in range(total_words):
                 if (uncovered >> i) & 1:
                     code_set.add(i)
                     cover_mask |= balls[i]
                     
    current_size = len(code_set)
    best_code = list(code_set)
    best_size = current_size
    
    # Step 2: Simulated Annealing
    # We want to minimize size.
    # Energy = size + M * (uncovered_count)
    # Since we maintain validity strictly, uncovered_count is 0.
    # So we just minimize size.
    # But we allow temporary invalidity to escape local minima?
    # If we allow invalidity, we need a large penalty.
    # Let's try a strict validity approach first: only remove if valid, or swap if valid.
    # Actually, "Swap" can be done as Remove + Add.
    # If Remove makes it invalid, we must Add something to fix it.
    
    # Better approach: 
    # 1. Try to remove a random codeword. If valid, accept.
    # 2. Try to swap a random codeword A with a random word B (not in code).
    #    Remove A. If invalid, add B. Check if valid.
    #    If valid, accept.
    
    # To speed up "is valid" checks, we maintain cover_mask.
    
    # SA Parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.99
    T = T_start
    steps = 0
    max_steps = 500000 # Limit steps
    
    while time.time() < deadline - 2.0 and steps < max_steps and T > T_end:
        steps += 1
        
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = list(code_set)
            
        # Choose move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try Remove
            if len(code_set) <= 1:
                continue
            c = random.choice(list(code_set))
            new_cover = cover_mask & ~balls[c]
            if new_cover == all_mask:
                # Valid removal
                code_set.remove(c)
                cover_mask = new_cover
            else:
                # Invalid, reject (or try to fix with add?)
                # For now, just reject.
                pass
                
        elif move_type < 0.8:
            # Try Add (only if we are currently invalid? No, we are always valid)
            # Adding increases size, so we only do this if we want to enable a future removal.
            # But simple SA usually explores neighborhood.
            # Let's try Swap.
            pass
        else:
            # Try Swap: Remove A, Add B
            if len(code_set) <= 1:
                continue
            a = random.choice(list(code_set))
            # Pick random B not in code
            # To pick random B not in code efficiently:
            # Generate random index, check if in code.
            b = random.randint(0, total_words - 1)
            if b in code_set:
                continue
                
            # Tentative
            new_cover = (cover_mask & ~balls[a]) | balls[b]
            if new_cover == all_mask:
                # Valid swap
                code_set.remove(a)
                code_set.add(b)
                cover_mask = new_cover
            else:
                # Invalid, reject
                pass
                
        T *= alpha
        
    # Final cleanup: Greedy removal of redundant words
    # This is fast and deterministic
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        # Shuffle to avoid bias? No, deterministic is fine for final polish.
        # Iterate through code
        for c in list(code_set):
            new_cover = cover_mask & ~balls[c]
            if new_cover == all_mask:
                code_set.remove(c)
                cover_mask = new_cover
                changed = True
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = list(code_set)
                # Restart loop as set changed
                break
                
    # Ensure final validity
    final_cover = 0
    for c in best_code:
        final_cover |= balls[c]
        
    if final_cover != all_mask:
        # Fallback to the code from SA if it was valid, or fix it
        # Actually, best_code was only updated if valid.
        # But let's double check.
        # If best_code is invalid, use the last valid state from SA loop?
        # We stored best_code only when len < best_size, which implies it was the current code_set.
        # And current code_set was only modified if valid.
        # So best_code should be valid.
        
        # However, if the very first greedy was invalid and we failed to fix it?
        # We fixed it before SA.
        
        # Just in case, add missing words
        uncovered = all_mask & ~final_cover
        for i in range(total_words):
            if (uncovered >> i) & 1:
                best_code.append(i)
                final_cover |= balls[i]
                
    # Convert to list of lists
    return [list(digits_of[i]) for i in best_code]

def _fast_greedy(balls, all_mask, deadline):
    uncovered = all_mask
    code = []
    total_words = len(balls)
    
    while uncovered and time.time() < deadline:
        best_idx = -1
        max_cover = 0
        
        # Optimize: if uncovered is small, we can iterate only over candidates that cover uncovered bits?
        # Hard to do efficiently with bitmasks without precomputed reverse index.
        # Just scan all.
        
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
