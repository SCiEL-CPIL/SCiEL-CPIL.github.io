import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    # Evaluator kills at 150s. We use a 10s safety margin.
    deadline = start_time + 140.0
    
    total_words = q ** n
    
    # Precompute digits for all words
    # For q=3, n=8, total_words=6561.
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute balls (neighborhoods) as bitmasks
    # A ball of radius r around i contains all words within Hamming distance r.
    balls = [0] * total_words
    
    # Precompute powers of q for fast index calculation
    powers = [1] * n
    for i in range(n-2, -1, -1):
        powers[i] = powers[i+1] * q

    for i in range(total_words):
        d = digits_of[i]
        mask = 1 << i  # self
        
        if radius == 1:
            for pos in range(n):
                orig_digit = d[pos]
                for new_digit in range(q):
                    if new_digit == orig_digit:
                        continue
                    delta = (new_digit - orig_digit) * powers[pos]
                    idx = i + delta
                    mask |= (1 << idx)
        else:
            # General radius fallback
            for r in range(1, radius + 1):
                for pos_comb in itertools.combinations(range(n), r):
                    for variation in itertools.product(range(q), repeat=r):
                        delta = 0
                        is_diff = False
                        for k, pos in enumerate(pos_comb):
                            new_val = variation[k]
                            if new_val != d[pos]:
                                is_diff = True
                                delta += (new_val - d[pos]) * powers[pos]
                        if not is_diff:
                            continue
                        idx = i + delta
                        mask |= (1 << idx)
                        
        balls[i] = mask

    all_mask = (1 << total_words) - 1
    
    # Helper to get the cover mask
    def get_cover_mask(indices):
        mask = 0
        for i in indices:
            mask |= balls[i]
        return mask

    # Strategy:
    # 1. Greedy Set Cover to get a valid initial solution.
    # 2. Aggressive Redundancy Pruning:
    #    - Iterate through the code.
    #    - If removing a codeword keeps the cover valid, remove it.
    #    - Repeat until no more removals are possible.
    # 3. To escape local minima of "minimal but not minimum" sets, we use a 
    #    "Kick" strategy:
    #    - Occasionally, we might get stuck in a minimal cover that is larger 
    #      than the optimum. 
    #    - However, for this specific problem (K_3(8,1)), the greedy + prune 
    #    often gets very close. 
    #    - A more powerful move is "Swap": Remove a word, and if it breaks 
    #      coverage, find a single word that fixes the most uncovered spots 
    #      and adds it? No, that increases size.
    #    - The key to beating greedy is often that greedy picks "locally best" 
    #      which leads to "globally suboptimal" due to overlap.
    #    
    #    Let's try a different approach for the second phase: 
    #    **Local Search with Swap Moves**.
    #    State: Set of codewords.
    #    Move: 
    #      1. Remove a random codeword c1.
    #      2. If valid, accept (size decreases).
    #      3. If invalid, find a codeword c2 NOT in the set such that 
    #         (Set - {c1} + {c2}) is valid. 
    #         If found, accept (size same, but structure changed, might allow 
    #         further removals).
    #      4. If no single swap fixes it, try a "double swap" or revert.
    #
    #    Actually, a simpler and very effective method for covering codes is 
    #    **Iterative Improvement via "Minimal" Check**.
    #
    #    Let's implement a robust Greedy + Prune, and if time permits, 
    #    run a simple SA that ONLY accepts moves that keep the solution valid 
    #    (i.e., only removals that don't break coverage, or swaps that 
    #    maintain coverage). This is essentially a random walk on the space 
    #    of minimal covers. If we can find a path from one minimal cover to 
    #    another with smaller size, we win. But minimal covers have the same 
    #    size? No, "minimal" means no element can be removed. "Minimum" means 
    #    smallest possible. A minimal cover can be larger than the minimum.
    #
    #    So, we need moves that change the set while keeping it valid, 
    #    potentially allowing future removals.
    #
    #    Let's stick to the proven effective strategy for this specific 
    #    benchmark: 
    #    1. Greedy.
    #    2. Prune to minimal.
    #    3. Randomized Local Search:
    #       - While time remains:
    #         - Try to remove a random element. If valid, keep it.
    #         - If not valid, try to swap a random element with a random 
    #           other element. If the new set is valid, keep it.
    #         - This explores the space of valid covers.
    
    def greedy_construct():
        uncovered = all_mask
        code = []
        while uncovered:
            if time.time() > deadline - 20.0:
                return code
            
            best_idx = -1
            max_cover = 0
            candidates = []
            
            for i in range(total_words):
                gain = (balls[i] & uncovered).bit_count()
                if gain > max_cover:
                    max_cover = gain
                    best_idx = i
                    candidates = [i]
                elif gain == max_cover and gain > 0:
                    candidates.append(i)
            
            if max_cover == 0:
                break
                
            best_idx = random.choice(candidates)
            code.append(best_idx)
            uncovered &= ~balls[best_idx]
            
        return code

    def prune_to_minimal(code):
        # Remove any redundant elements
        changed = True
        while changed:
            changed = False
            cover = get_cover_mask(code)
            for i in range(len(code)):
                c = code[i]
                # If removing c keeps cover valid
                if (cover & ~balls[c]) == all_mask:
                    code.pop(i)
                    changed = True
                    break # Restart loop to recompute cover efficiently? 
                    # Actually, we can update cover incrementally if we track it,
                    # but recomputing is O(k * ball_size) which is fine for k~500.
        
        return code

    # Initial Greedy
    code = greedy_construct()
    code = prune_to_minimal(code)
    
    best_code = code[:]
    best_size = len(code)
    
    # Local Search Phase
    # We will run a loop that tries to improve the solution.
    # Move 1: Random Removal. If valid, accept.
    # Move 2: Random Swap. If valid, accept.
    # We track the current cover mask to speed up validity checks.
    
    current_cover = get_cover_mask(code)
    current_uncovered = total_words - current_cover.bit_count()
    
    # If greedy+prune didn't cover everything (shouldn't happen), fix it
    if current_uncovered > 0:
        # Greedy fallback again
        code = greedy_construct()
        code = prune_to_minimal(code)
        current_cover = get_cover_mask(code)
        best_code = code[:]
        best_size = len(code)

    all_indices = list(range(total_words))
    
    # Time budget for local search
    ls_deadline = deadline - 5.0
    
    # We use a simple randomized hill climbing / local search.
    # To avoid getting stuck, we don't use temperature, just random restarts 
    # or random walks.
    # Actually, a "Randomized Minimal Cover Search" is good.
    
    while time.time() < ls_deadline:
        # Try a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Try to remove a random element
            if not code:
                continue
            idx = random.randint(0, len(code) - 1)
            c = code[idx]
            
            # Check if removing c keeps it valid
            new_cover = current_cover & ~balls[c]
            if new_cover == all_mask:
                # Accept removal
                code.pop(idx)
                current_cover = new_cover
                if len(code) < best_size:
                    best_code = code[:]
                    best_size = len(code)
            else:
                # Rejection: do nothing
                pass
                
        else:
            # Try a random swap
            if not code:
                continue
            idx = random.randint(0, len(code) - 1)
            old_c = code[idx]
            new_c = random.choice(all_indices)
            
            if new_c == old_c:
                continue
            if new_c in code:
                continue
            
            # Calculate new cover
            # new_cover = (current_cover & ~balls[old_c]) | balls[new_c]
            # This is fast with bitmasks
            
            # Optimization: Only compute if it might be valid?
            # We can check if new_c covers any of the words that old_c uniquely covered?
            # Too complex. Just compute.
            
            temp_cover = (current_cover & ~balls[old_c]) | balls[new_c]
            
            if temp_cover == all_mask:
                # Accept swap
                code[idx] = new_c
                current_cover = temp_cover
                # Size is same, so no immediate best update, 
                # but this new configuration might allow future removals.
                
                # Optional: Try to prune immediately after a successful swap?
                # It might be expensive to prune every swap. 
                # Let's just let the removal moves handle it.
                pass

    # Final Prune on the best code found
    if best_code:
        best_code = prune_to_minimal(best_code)
    
    # Fallback
    if not best_code:
        best_code = code
        best_code = prune_to_minimal(best_code)

    # Convert indices back to words
    return [list(digits_of[i]) for i in best_code]

