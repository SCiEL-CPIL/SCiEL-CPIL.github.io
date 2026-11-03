"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    """
    Construct a covering code for K_q(n, radius).
    Strategy:
    1. Start with the direct product of the ternary tetracode (if n>=4) and the full space on the rest,
       or a simple product code if n<4.
    2. Greedily remove redundant codewords. A codeword is redundant if it is within distance <= radius
       of another codeword. We check this by looking at all other codewords.
       To speed this up, we can use the fact that for radius 1, a word is covered by its neighbors.
       Actually, a simpler greedy removal:
       - Maintain a set of active codewords.
       - Iterate through codewords. For each, check if it is covered by any OTHER active codeword.
       - If yes, mark it for removal.
       - Repeat until no more can be removed.
    
    A more efficient way for radius 1:
    A codeword c is redundant if there exists another codeword c' such that d(c, c') <= 1.
    Since the code is a covering code, every word is within distance 1 of some codeword.
    We can try to remove words that are "close" to others.
    
    Let's use a randomized greedy removal:
    1. Start with the seed code.
    2. Repeat:
       a. Pick a random codeword c from the current set.
       b. Check if c is within distance <= radius of any other codeword in the set.
       c. If yes, remove c.
       d. If no, keep it.
    3. Continue until time limit.
    
    To make it faster, we can use a smarter check.
    For radius 1, c is covered by another codeword c' if c' is in the neighborhood of c (including c itself, but we exclude c).
    So we just need to check if any other codeword is in the neighborhood of c.
    
    Even better: We can build the code and then try to "tighten" it.
    """
    
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin for 120s limit
    
    # Step 1: Generate the initial seed code
    # The seed code is the direct product of the tetracode and the full space on the remaining coordinates.
    # Tetracode: 9 words of length 4.
    # If n < 4, we just use the full space (but that's not efficient). Let's handle n=8 specifically or generally.
    
    # For n=8, q=3, radius=1:
    # Tetracode C_t is a perfect 1-error-correcting code of length 4, size 9.
    # Seed code: C_t x {0,1,2}^(n-4). Size: 9 * 3^(n-4).
    
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        seed_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n-4)]
    else:
        # Fallback for smaller n: just use all words (brute force is small)
        # But n is likely 8 in the task.
        seed_code = [list(tail) for tail in itertools.product(range(q), repeat=n)]
    
    # Convert to a set of tuples for fast membership check
    # But we need to check distance, not just membership.
    # We will maintain a list of codewords and a set of tuples for the current code.
    
    code = [tuple(c) for c in seed_code]
    code_set = set(code)
    
    # Helper function to check if a word is covered by another word in the set
    # We want to know if there exists w' in code_set such that w' != w and d(w, w') <= radius.
    # For radius 1, this means checking all neighbors of w.
    
    def is_redundant(w, current_set):
        # Check if any neighbor of w is in current_set (excluding w itself)
        # Generate all neighbors of w within distance 1
        for i in range(n):
            orig_val = w[i]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                # Create a new word with new_val at position i
                # To be efficient, we can use a list and then convert to tuple
                # But for small n and q, this is fine.
                # Optimization: precompute neighbors? No, w changes.
                neighbor = list(w)
                neighbor[i] = new_val
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple in current_set:
                    return True
        return False
    
    # Greedy removal
    # We can't just iterate once because removing one word might make another redundant.
    # We iterate until no more removals are possible or time runs out.
    
    # To speed up, we can shuffle the order of checking.
    # We will maintain a list of words to check.
    # Actually, a better strategy:
    # Iterate through the list of codewords. For each, check if it's redundant.
    # If it is, remove it from the set and the list.
    # If not, keep it.
    # Then repeat the process on the remaining list.
    # This is O(K^2 * n * q) in the worst case, which might be too slow for K=729.
    # 729^2 * 8 * 3 = ~12 million operations per pass. That's fine in Python if optimized.
    # But we might need many passes.
    
    # Let's try a different approach:
    # We know that in a covering code, every word is covered.
    # We can try to remove a word w if there exists another word w' such that d(w, w') <= 1.
    # We can precompute for each word in the code, the list of other words in the code within distance 1.
    # But the set changes.
    
    # Let's stick to the simple iterative removal but optimize the check.
    # We can use a heuristic: always remove the word with the "least" unique coverage?
    # But that's expensive to compute.
    
    # Simple greedy:
    # 1. Make a copy of the code list.
    # 2. Shuffle it.
    # 3. For each word w in the shuffled list:
    #    a. If w is not in current_set, skip (it was already removed).
    #    b. Check if w is redundant w.r.t current_set \ {w}.
    #    c. If redundant, remove w from current_set.
    # 4. Repeat steps 1-3 until no removals happen in a full pass or time limit.
    
    # We will use a flag to track if any removal happened.
    
    max_passes = 100  # Safety cap
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        removed_any = False
        # Shuffle for randomness
        code_list = list(current_set)
        random.shuffle(code_list)
        
        for w in code_list:
            if time.time() > deadline:
                break
            
            if w not in current_set:
                continue
            
            # Temporarily remove w to check if it's still covered? No, that's not the criterion.
            # The criterion for redundancy in a covering code is:
            # w can be removed if every word that was ONLY covered by w is still covered by others.
            # But we don't track who covers whom.
            
            # A simpler, valid criterion for removal in a covering code:
            # w is redundant if w itself is within distance <= radius of some OTHER codeword w' in the set.
            # Why is this valid?
            # If we remove w, is the code still a covering code?
            # Let x be any word in the space.
            # Originally, d(x, w) <= 1 for some w in code.
            # If d(x, w) <= 1 and w is removed, we need d(x, w'') <= 1 for some w'' in remaining code.
            # We know d(w, w') <= 1 (since w is redundant w.r.t w').
            # By triangle inequality, d(x, w') <= d(x, w) + d(w, w') <= 1 + 1 = 2.
            # This does NOT guarantee d(x, w') <= 1.
            # So this criterion is INVALID.
            
            # Correct criterion:
            # w is redundant if for every x such that d(x, w) <= 1, there exists w'' in code\{w} such that d(x, w'') <= 1.
            # This is expensive to check.
            
            # However, there is a known property for perfect codes or certain structures.
            # For the direct product code C_t x {0..q-1}^(n-4), is it true that if d(w, w') <= 1, then w is redundant?
            # Let's analyze.
            # Let w = (t, x), w' = (t', x').
            # d(w, w') = d(t, t') + d(x, x') <= 1.
            # Since d(t, t') >= 0 and d(x, x') >= 0, either:
            # 1. d(t, t') = 0, d(x, x') = 1.
            # 2. d(t, t') = 1, d(x, x') = 0.
            
            # Case 1: t = t', d(x, x') = 1.
            # Let y be a word such that d(y, w) <= 1.
            # y = (u, v). d(u, t) + d(v, x) <= 1.
            # We need to show d(y, w'') <= 1 for some w'' in code\{w}.
            # w' = (t, x') is in code.
            # d(y, w') = d(u, t) + d(v, x').
            # We know d(v, x) = 0 or 1.
            # If d(v, x) = 0, then d(v, x') = d(x, x') = 1.
            # Then d(y, w') = d(u, t) + 1.
            # Since d(u, t) + d(v, x) <= 1 and d(v, x)=0, we have d(u, t) <= 1.
            # So d(y, w') <= 1 + 1 = 2. Still not enough.
            
            # Wait, if d(u, t) = 0, then d(y, w') = 1. OK.
            # If d(u, t) = 1, then d(v, x) = 0, so d(y, w') = 1 + 1 = 2. Not covered by w'.
            # But maybe y is covered by another word?
            # y = (u, v) with d(u, t) = 1, v = x.
            # Since C_t is a perfect code, for any u, there exists t'' in C_t such that d(u, t'') = 1 (if u != t) or 0 (if u = t).
            # Here d(u, t) = 1, so u is at distance 1 from t.
            # Is there a t'' in C_t such that d(u, t'') = 0? No, u is not in C_t (distance 1).
            # Is there a t'' in C_t such that d(u, t'') = 1? Yes, t is one.
            # Are there others?
            # The tetracode is perfect. The sphere of radius 1 around t covers all words.
            # The words at distance 1 from t are not in the code.
            # So u is not in the code.
            # We need a codeword w'' = (t'', x'') such that d(y, w'') <= 1.
            # d(y, w'') = d(u, t'') + d(v, x'') = d(u, t'') + d(x, x'').
            # We need this to be <= 1.
            # Since d(u, t) = 1, d(u, t'') can be 1 (if t''=t) or more.
            # If t'' = t, we are back to w'.
            # If t'' != t, d(u, t'') >= 2?
            # In the tetracode, the minimum distance is 2? No, it's a perfect code of radius 1, so min distance is 3.
            # If min distance is 3, then for any t, t'' in C_t, t != t'', d(t, t'') >= 3.
            # Then d(u, t'') >= d(t, t'') - d(t, u) >= 3 - 1 = 2.
            # So d(u, t'') >= 2.
            # Then d(y, w'') = d(u, t'') + d(x, x'') >= 2 + 0 = 2.
            # So y is NOT covered by any other codeword if w is removed.
            # Thus, w is NOT redundant.
            
            # Conclusion: For the direct product of a perfect code and a full space, NO word is redundant if we use the simple "covered by another" criterion.
            # The code is "irredundant" in this sense.
            
            # So this approach will not remove any words.
            
            # We need a different strategy.
            
            # Strategy 2: Start with a different code.
            # The known best code has 486 words. The seed has 729.
            # 486 = 2 * 3^5 = 2 * 243.
            # 729 = 9 * 3^4 = 9 * 81.
            # 486 = 6 * 3^3 * 3? No. 486 = 2 * 243.
            # Is there a code of size 486?
            # The van Laarhoven code is size 486.
            # How is it constructed?
            # It is not a simple product.
            
            # Given the time limit and the difficulty of constructing the optimal code from scratch,
            # and the fact that the simple removal strategy fails, we should try to find a better seed.
            
            # Alternative Seed:
            # Maybe we can use a random code and then optimize?
            # Or use a different algebraic construction.
            
            # Let's try a random greedy construction.
            # 1. Start with an empty set.
            # 2. Repeat:
            #    a. Pick a random word w from the space.
            #    b. If w is not covered by the current set, add w to the set.
            #    c. Remove from the "uncovered" set all words within distance 1 of w.
            # 3. This is a standard greedy covering code algorithm.
            # 4. The size depends on the random order.
            # 5. We can run this multiple times with different seeds and keep the best result.
            
            # This might produce a code of size ~500-600.
            # 6561 words. Each word covers 1 + 8*2 = 17 words (for n=8, q=3, radius=1).
            # Lower bound: 6561 / 17 = 386.
            # Greedy might give 500-600.
            # The seed is 729.
            # So greedy might be better than the seed.
            
            # Let's implement the greedy algorithm with a time limit.
            
            # We will use a numpy array to track covered words.
            # Map words to indices: 0 to 3^8 - 1.
            # 3^8 = 6561.
            
            # Precompute neighbors for each index?
            # 6561 * 17 = 111537 neighbors. This is small.
            
            # Let's try this.
            
            break  # Break out of the removal loop, switch to greedy construction
        
        if not removed_any:
            break
    
    # --- Greedy Construction ---
    
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx
    
    def idx_to_word(idx):
        w = [0] * n
        for i in range(n-1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)
    
    # Precompute neighbors for each index
    # neighbors[i] is a list of indices within distance 1 of i
    neighbors = [[] for _ in range(q**n)]
    for idx in range(q**n):
        w = idx_to_word(idx)
        neighbors[idx].append(idx)
        for i in range(n):
            orig = w[i]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[i] = new_val
                neighbors[idx].append(word_to_idx(w))
        w = idx_to_word(idx) # Restore
    
    # Greedy algorithm
    covered = np.zeros(q**n, dtype=bool)
    code_indices = []
    
    # We want to add words that cover the most uncovered words.
    # This is a weighted set cover problem.
    # Greedy: at each step, pick the word that covers the most uncovered words.
    # This is O(K * Space * n * q) which is too slow.
    # Space = 6561. K ~ 500. 500 * 6561 * 24 = 78 million. Might be okay in 100s?
    # But we need to update "uncovered" counts.
    
    # Optimization:
    # Maintain an array `uncovered_count` for each potential codeword? No.
    # Maintain a set of uncovered words.
    # For each candidate word c, count how many uncovered words are in its neighborhood.
    
    # Let's try a simpler random greedy first:
    # Pick a random uncovered word, add a random word that covers it?
    # Or: Pick a random word, if it covers any uncovered word, add it.
    
    # Better:
    # 1. List of uncovered words.
    # 2. While uncovered words exist:
    #    a. Pick a random word c from the space.
    #    b. Check if c covers any uncovered word.
    #    c. If yes, add c to code, mark all words in neighbors[c] as covered.
    #    d. If no, discard c.
    # 3. This is slow.
    
    # Even better:
    # 1. List of uncovered words.
    # 2. Pick a random uncovered word u.
    # 3. Pick a random word c from neighbors[u].
    # 4. Add c to code.
    # 5. Mark all words in neighbors[c] as covered.
    # This guarantees progress.
    
    # Let's implement this.
    
    uncovered = set(range(q**n))
    code_indices = []
    
    while uncovered:
        if time.time() > deadline:
            break
        
        # Pick a random uncovered word
        u = random.choice(list(uncovered))
        
        # Pick a random word c from neighbors[u]
        c = random.choice(neighbors[u])
        
        # Add c to code
        code_indices.append(c)
        
        # Mark all words in neighbors[c] as covered
        for v in neighbors[c]:
            if v in uncovered:
                uncovered.remove(v)
    
    # Convert indices back to words
    final_code = [idx_to_word(idx) for idx in code_indices]
    
    return final_code

# EVOLVE-BLOCK-END
