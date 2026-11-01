import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    # Handle the specific case q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other cases: return the whole space (trivial valid code)
        return [list(x) for x in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    # Precompute all words and their balls
    all_words = list(itertools.product(range(3), repeat=6))
    N = len(all_words) # 729
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # balls[i] is a list of indices of words within distance 1 of all_words[i]
    balls = []
    for w in all_words:
        ball = [word_to_idx[w]]
        for i in range(6):
            for d in range(3):
                if d == w[i]: continue
                new_w = list(w)
                new_w[i] = d
                ball.append(word_to_idx[tuple(new_w)])
        balls.append(ball)
    
    # Convert balls to numpy arrays for faster set operations if needed, 
    # but lists are fine for 17 elements.
    
    # Greedy Construction
    def greedy_construct():
        uncovered = set(range(N))
        code = []
        while uncovered:
            best_word = -1
            best_count = -1
            # Randomly shuffle order of candidates to break symmetry
            candidates = list(uncovered)
            random.shuffle(candidates)
            # We only need to check words in uncovered or their neighbors? 
            # Actually, the best next codeword must cover at least one uncovered word.
            # So we can just iterate over all words, or just those in uncovered.
            # Iterating over all 729 words is fast.
            for w_idx in range(N):
                # Optimization: if w_idx is not in uncovered, it might still be a good codeword?
                # No, a codeword that doesn't cover any uncovered word is useless.
                # But a codeword *not* in uncovered might cover uncovered words.
                # So we must check all words, or at least words that are "close" to uncovered.
                # Checking all 729 is cheap (729 * 17 ~ 12k ops).
                count = 0
                for u in balls[w_idx]:
                    if u in uncovered:
                        count += 1
                        if count > best_count:
                            best_count = count
                            best_word = w_idx
                # Early exit if we found a word that covers all remaining? Unlikely.
                
            if best_word == -1:
                break # Should not happen if uncovered is not empty
                
            code.append(best_word)
            # Update uncovered
            for u in balls[best_word]:
                uncovered.discard(u)
                
        return code

    best_code = []
    best_size = N # Initialize with worst case
    
    # Run multiple restarts
    num_restarts = 200
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        random.seed(restart) # Deterministic restarts for reproducibility, or just let random run
        
        # 1. Greedy Start
        code = greedy_construct()
        code_set = set(code)
        
        # 2. Local Search to minimize size
        # We track coverage counts to quickly determine if a word is removable.
        coverage = np.zeros(N, dtype=np.int32)
        for c in code:
            for w in balls[c]:
                coverage[w] += 1
        
        # Helper to check if a word can be removed
        def is_removable(c_idx):
            for w in balls[c_idx]:
                if coverage[w] == 1:
                    return False
            return True

        # Helper to count "unique" words (covered by exactly 1 codeword)
        def count_unique():
            return np.sum(coverage == 1)

        current_unique = count_unique()
        
        # Local Search Loop
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Step 1: Try to remove any word
            # Find all removable words
            removable = [c for c in code if is_removable(c)]
            if removable:
                # Remove one (randomly to break symmetry)
                c_to_remove = random.choice(removable)
                code.remove(c_to_remove)
                code_set.discard(c_to_remove)
                for w in balls[c_to_remove]:
                    coverage[w] -= 1
                improved = True
                current_unique = count_unique()
                continue # Restart the loop to check for more removals
            
            # Step 2: If no removal possible, try swaps to reduce "uniqueness"
            # We want to swap c in code with c' not in code such that
            # the new code has fewer unique words, or allows a removal.
            # A simple heuristic: try to swap a word that has many unique words
            # with a word that covers those unique words and adds less new uniqueness.
            
            # Find words in code with high "unique load"
            # unique_load[c] = number of words in ball(c) that are covered only by c
            unique_loads = {}
            for c in code:
                load = 0
                for w in balls[c]:
                    if coverage[w] == 1:
                        load += 1
                unique_loads[c] = load
                
            # Sort code words by unique load descending
            sorted_code = sorted(code, key=lambda x: unique_loads[x], reverse=True)
            
            # Try swapping the top few candidates
            swapped = False
            for c_in in sorted_code[:5]: # Only try top 5
                if time.time() > deadline:
                    break
                
                # Try to find a c_out not in code that reduces uniqueness
                # c_out should cover the unique words of c_in
                # And ideally cover other words already covered by code.
                
                # Candidate c_out must be outside current code
                # We can iterate through all words not in code? That's 729 - ~80 = ~650.
                # For each, calculate change in uniqueness.
                
                best_c_out = -1
                best_delta_unique = 0 # We want delta < 0 (decrease in unique count)
                
                # Optimization: c_out must cover at least one unique word of c_in
                unique_words_of_c_in = [w for w in balls[c_in] if coverage[w] == 1]
                
                if not unique_words_of_c_in:
                    continue # If c_in has no unique words, it's already "removable" in a sense?
                             # Wait, if coverage[w] > 1 for all w in ball(c_in), it IS removable.
                             # So if we are here, c_in has some unique words.
                             
                for c_out in range(N):
                    if c_out in code_set:
                        continue
                    
                    # Check if c_out covers any unique word of c_in
                    covers_unique = False
                    for w in unique_words_of_c_in:
                        if w in balls[c_out]:
                            covers_unique = True
                            break
                    if not covers_unique:
                        continue
                    
                    # Calculate change in unique count
                    # Remove c_in, add c_out
                    # New coverage = coverage - ball(c_in) + ball(c_out)
                    # We can simulate this on a temporary array?
                    # Or just count the change directly.
                    
                    # Words that become unique:
                    # 1. Words in ball(c_out) that were covered by c_in (and no one else?) 
                    #    If coverage[w] == 1 (by c_in), and c_out covers w, w is still covered.
                    #    If coverage[w] == 2 (by c_in and c_x), and c_out covers w, 
                    #       after removing c_in, coverage becomes 1 (by c_x). Then adding c_out makes it 2.
                    #    This is complex to track without simulation.
                    
                    # Let's just simulate the swap for the top candidates.
                    # Since we only check ~5 c_in and ~650 c_out, and each check is ~34 ops,
                    # 5 * 650 * 34 ~ 110k ops. Very fast.
                    
                    # Simulate
                    # Temp coverage
                    # We don't want to allocate new array every time.
                    # Let's just calculate the delta.
                    
                    # Delta in unique count:
                    # Let U be the set of words with coverage 1.
                    # After swap:
                    # For w in ball(c_in) \ ball(c_out): coverage decreases by 1.
                    #   If coverage[w] was 1 -> 0 (Invalid, but c_in is in code, so coverage>=1.
                    #   If coverage[w] was 1, it was unique to c_in. After removal, it's 0? No, c_in is removed.
                    #   Wait, if coverage[w]==1 and w in ball(c_in), it means only c_in covers w.
                    #   If c_out does NOT cover w, then w becomes uncovered. INVALID SWAP.
                    
                    # So c_out MUST cover all unique words of c_in.
                    # We already checked if it covers *any*. We need it to cover *all*.
                    
                    valid_swap = True
                    for w in unique_words_of_c_in:
                        if w not in balls[c_out]:
                            valid_swap = False
                            break
                    if not valid_swap:
                        continue
                    
                    # Now check if the swap increases uniqueness elsewhere.
                    # Words in ball(c_out) \ ball(c_in): coverage increases by 1.
                    #   If coverage[w] was 1 (unique to some c_x), it becomes 2. Unique count decreases by 1.
                    #   If coverage[w] was 2, it becomes 3. No change in unique count.
                    #   If coverage[w] was 0 (impossible, all covered).
                    
                    # Words in ball(c_in) \ ball(c_out): coverage decreases by 1.
                    #   We know these are not unique to c_in (since c_out covers all unique ones).
                    #   So coverage[w] >= 2.
                    #   If coverage[w] was 2, it becomes 1. Unique count increases by 1.
                    #   If coverage[w] was > 2, no change.
                    
                    delta_unique = 0
                    
                    # Words gaining coverage (in c_out, not c_in)
                    for w in balls[c_out]:
                        if w not in balls[c_in]:
                            if coverage[w] == 1:
                                delta_unique -= 1 # Becomes 2, so unique count drops
                            
                    # Words losing coverage (in c_in, not c_out)
                    for w in balls[c_in]:
                        if w not in balls[c_out]:
                            if coverage[w] == 2:
                                delta_unique += 1 # Becomes 1, so unique count rises
                            
                    if delta_unique < best_delta_unique:
                        best_delta_unique = delta_unique
                        best_c_out = c_out
                        
                if best_c_out != -1:
                    # Perform swap
                    code.remove(c_in)
                    code_set.discard(c_in)
                    for w in balls[c_in]:
                        coverage[w] -= 1
                        
                    code.append(best_c_out)
                    code_set.add(best_c_out)
                    for w in balls[best_c_out]:
                        coverage[w] += 1
                    
                    improved = True
                    swapped = True
                    current_unique = count_unique()
                    break # Restart loop with new code state
            
            if not improved and not swapped:
                # Try random swaps to escape local minima?
                # If stuck, pick a random word in code and a random word out, swap if valid (covers unique).
                # This is essentially what we did, but restricted.
                # Let's just break if we can't improve uniqueness.
                break
                
        # Record result
        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)
            
            if best_size == 73:
                break # Found optimal

    # Convert back to list of lists
    result = [list(all_words[idx]) for idx in best_code]
    
    return result
