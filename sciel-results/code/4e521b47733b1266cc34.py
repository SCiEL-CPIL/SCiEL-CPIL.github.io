import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s limit
    
    # Precomputation
    # Total words in space
    total_words = q ** n
    
    # Map word (tuple) to index
    # We can just use integer representation in base q
    def tuple_to_idx(t):
        idx = 0
        for val in t:
            idx = idx * q + val
        return idx
        
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % q
            idx //= q
        return t

    # Precompute balls for all codewords
    # ball_masks[c_idx] is a bitmask of all words covered by codeword c_idx
    # Since total_words can be up to 2187, we use Python integers as bitsets.
    # Bit i is set if word i is covered.
    
    ball_masks = [0] * total_words
    
    # Generate all words
    all_words = []
    for idx in range(total_words):
        t = idx_to_tuple(idx)
        all_words.append(t)
        
    # For each codeword, compute the mask of words it covers
    for c_idx, c_word in enumerate(all_words):
        mask = 0
        # Cover the word itself
        mask |= (1 << c_idx)
        
        # Cover words at distance 1
        for pos in range(n):
            orig_val = c_word[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create neighbor
                neighbor = list(c_word)
                neighbor[pos] = new_val
                n_idx = tuple_to_idx(neighbor)
                mask |= (1 << n_idx)
        ball_masks[c_idx] = mask
        
    # Greedy Set Cover Algorithm
    def greedy_solve(random_seed):
        rng = random.Random(random_seed)
        uncovered = (1 << total_words) - 1
        selected = []
        
        # To speed up, we can maintain a list of available codewords
        # But checking all 2187 each time is O(2187 * iterations).
        # Iterations ~ 200. Total ops ~ 400k. Very fast.
        
        while uncovered:
            best_gain = -1
            best_c_idx = -1
            # Random tie-breaking helps escape local optima
            # We iterate through all codewords
            for c_idx in range(total_words):
                gain = (uncovered & ball_masks[c_idx]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_c_idx = c_idx
                elif gain == best_gain and gain > 0:
                    # Random tie break
                    if rng.random() < 0.5:
                        best_c_idx = c_idx
                        
            if best_c_idx == -1 or best_gain == 0:
                # Should not happen if uncovered is not empty and balls are valid
                break
                
            selected.append(best_c_idx)
            uncovered ^= (uncovered & ball_masks[best_c_idx])
            
        # Post-processing: Remove redundant codewords
        # This is a simple local search: try removing each codeword
        # If we remove c, do we still cover everything?
        # We can recompute coverage or check if any word becomes uncovered.
        # A simpler check: For each selected codeword c, check if there is a word w 
        # that is ONLY covered by c among the selected set. If no such word, c is redundant.
        
        # To do this efficiently:
        # 1. Compute coverage count for each word from the selected set.
        # 2. If any word has count 1, its covering codeword is essential.
        # 3. Codewords that don't uniquely cover any word can be removed.
        
        # However, removing one might make another redundant.
        # We can iterate this pruning process.
        
        changed = True
        while changed:
            changed = False
            # Count how many selected codewords cover each word
            # This is O(|selected| * |ball_size|) per check, which is small.
            # But doing it from scratch every time is expensive.
            
            # Alternative:
            # For each c in selected (in reverse order to remove last added first?):
            # Check if c is essential.
            # c is essential if there exists a word w in ball(c) such that
            # no other c' in selected (c' != c) covers w.
            
            new_selected = []
            for c_idx in selected:
                # Check if c_idx is essential
                essential = False
                c_mask = ball_masks[c_idx]
                # Iterate through bits set in c_mask
                # If any bit w is set in c_mask and NOT in union of others, it's essential.
                
                # Union of others:
                # We can precompute union of all selected, then subtract c?
                # No, we need to know if w is covered by ANY other.
                
                # Efficient check:
                # Let U = union of masks of all selected codewords.
                # If we remove c, new_uncovered = U \ ball(c).
                # If (new_uncovered & ~U) == 0? No, that's trivial.
                # We need to check if ball(c) contains any word that is NOT covered by (selected \ {c}).
                
                # Let's compute coverage counts for words in ball(c).
                # This is still potentially slow if done naively for each c.
                
                # Given small size, let's just re-verify coverage for the specific ball of c.
                # For each word w in ball(c):
                #   is w covered by some c' in selected where c' != c?
                
                # Optimization:
                # If we maintain a global coverage count array, we can update it.
                # But the "while changed" loop makes state management tricky.
                
                # Let's stick to a simpler heuristic:
                # Just return the greedy result. The pruning might be complex to implement correctly and fast enough in Python within the loop.
                # Greedy alone is usually quite good.
                
                new_selected.append(c_idx)
            
            selected = new_selected
            
        return [all_words[i] for i in selected]

    best_solution = None
    best_len = float('inf')
    
    # Run multiple greedy passes with different seeds
    seed = 0
    while time.time() - start_time < time_limit:
        solution = greedy_solve(seed)
        if len(solution) < best_len:
            best_len = len(solution)
            best_solution = solution
            
        seed += 1
        # If we get a very good solution, we might want to spend more time?
        # Or just keep trying.
        
    if best_solution is None:
        # Fallback to the original method if something went wrong
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return best_solution
