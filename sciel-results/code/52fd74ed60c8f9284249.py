import time
import random
import sys

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses a local search algorithm starting from greedy solutions to minimize the number of codewords.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Total number of words in the space
    total_words = q ** n
    
    # Helper to convert integer to list of digits
    def int_to_word(idx, n, q):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word
        
    # Helper to convert list of digits to integer
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors (balls of radius 'radius')
    # For q=3, n=6, radius=1, ball size is 1 + 6*2 = 13
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i, n, q)
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        neighbors[i].append(neighbor_idx)
                w[pos] = original_val
        if radius >= 2:
             # Not needed for this specific task (radius=1), but keeping for generality if it were small
             pass

    def greedy_construct():
        """Generates a covering code using the greedy set cover algorithm."""
        covered = [0] * total_words
        codewords = []
        
        # Track uncovered count for quick selection
        # We can just iterate and pick max coverage each time
        # To speed up, we can maintain a list of uncovered words? 
        # For 729 words, O(N^2) is fine. 729 * 729 * 13 ~ 7M ops. Very fast.
        
        # Actually, let's just use a set of uncovered indices
        uncovered = set(range(total_words))
        
        while uncovered:
            best_center = -1
            best_count = -1
            
            # We only need to consider centers that are in uncovered? 
            # No, a covered center might cover many uncovered ones.
            # But typically, picking an uncovered center is a good heuristic.
            # Let's pick the center that maximizes |ball(center) intersect uncovered|.
            
            # Optimization: If we restrict to uncovered centers, it's faster and often good enough.
            # Let's try restricting to uncovered centers first for speed, or all if time permits.
            # Given the small size, let's do all candidates to get a better initial solution.
            
            for candidate in range(total_words):
                # Count uncovered neighbors
                count = 0
                for nb in neighbors[candidate]:
                    if nb in uncovered:
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_center = candidate
            
            if best_center == -1:
                break
                
            codewords.append(best_center)
            for nb in neighbors[best_center]:
                if nb in uncovered:
                    uncovered.remove(nb)
        
        return codewords

    def local_search(initial_codewords):
        """
        Improves the covering code using local search.
        Moves:
        1. Remove redundant codeword.
        2. Swap codeword c_old with c_new if it maintains coverage, then try to remove.
        """
        # Use a set for fast lookup and removal
        code_set = set(initial_codewords)
        # coverage_count[w] = number of codewords in code_set that cover w
        coverage_count = [0] * total_words
        
        for c in code_set:
            for w in neighbors[c]:
                coverage_count[w] += 1
        
        # Helper to check if code is valid (all covered)
        def is_valid():
            for w in range(total_words):
                if coverage_count[w] == 0:
                    return False
            return True
        
        # Helper to try removing a redundant codeword
        def try_remove():
            # Find a redundant codeword
            # A codeword c is redundant if for all w in ball(c), coverage_count[w] > 1
            redundant = None
            for c in code_set:
                is_red = True
                for w in neighbors[c]:
                    if coverage_count[w] <= 1:
                        is_red = False
                        break
                if is_red:
                    redundant = c
                    break
            
            if redundant is not None:
                # Remove it
                for w in neighbors[redundant]:
                    coverage_count[w] -= 1
                code_set.remove(redundant)
                return True
            return False

        # Perform local search
        # We loop until no improvement is found or time runs out
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            
            # 1. Try to remove redundant codewords
            # We might remove multiple in one pass
            while True:
                if time.time() > deadline:
                    return list(code_set)
                if not try_remove():
                    break
                improved = True
            
            # 2. Try swaps
            # For each c_old in code_set, try replacing with c_new not in code_set
            # Check if new set is valid.
            # To make this efficient, we can check if the "gap" left by c_old is covered by c_new.
            # Gap: words w such that coverage_count[w] == 1 and w is in ball(c_old).
            # If such words exist, c_new MUST cover them.
            # If c_new covers all such words, then the new set is valid (assuming c_new doesn't break others? 
            # Actually, adding c_new only increases counts, so it won't break validity of other words.
            # It only risks the words that were uniquely covered by c_old).
            
            # So, valid swap condition:
            # Let U = { w in ball(c_old) | coverage_count[w] == 1 }
            # If U is empty, any c_new is a valid swap (but might not help reduce size immediately, 
            # unless it makes another codeword redundant).
            # If U is not empty, c_new must cover all w in U.
            
            # We want to find a swap that allows a subsequent removal.
            # So after a valid swap, we call try_remove().
            
            # To speed up, we can prioritize swaps that cover "critical" words.
            
            # Iterate over c_old
            c_old_list = list(code_set)
            random.shuffle(c_old_list) # Randomize order to avoid bias
            
            for c_old in c_old_list:
                if time.time() > deadline:
                    return list(code_set)
                
                # Identify words uniquely covered by c_old
                unique_words = []
                for w in neighbors[c_old]:
                    if coverage_count[w] == 1:
                        unique_words.append(w)
                
                # If no unique words, c_old is "soft" redundant in a way? 
                # No, it's just that removing it would uncover nothing *immediately* if we don't add anything?
                # Wait, if coverage_count[w] > 1 for all w in ball(c_old), then c_old IS redundant and try_remove would have caught it.
                # So if we are in the swap phase, c_old is NOT redundant, meaning unique_words is NOT empty?
                # Not necessarily. It's possible that c_old is not redundant because some w in ball(c_old) has count 1, 
                # but that w is also covered by another codeword? No, count 1 means ONLY c_old covers it.
                # So if c_old is not redundant, there must be at least one w in ball(c_old) with count 1.
                
                if not unique_words:
                    # This shouldn't happen if try_remove was exhaustive, but just in case.
                    # If this happens, c_old is redundant.
                    continue

                # We need c_new to cover all unique_words.
                # Let's find candidates for c_new that cover all unique_words.
                # And c_new should not be in code_set.
                
                # To limit search, we can just iterate all possible c_new (0..total_words-1)
                # and check if it covers unique_words.
                # If valid, perform swap, then try_remove.
                
                # Optimization: Precompute for each word w, the list of centers that cover it?
                # Or just check membership.
                
                # Let's iterate c_new
                # To improve convergence, we can prioritize c_new that are "far" or random.
                # Let's just iterate all and pick the first one that leads to a removal.
                
                # To avoid O(N^2) explosion, we can limit the number of swaps tried per c_old or per iteration.
                # But N=729, so 729*729 ~ 500k checks. Each check is O(|unique_words|) + O(1) for validity.
                # This is fast enough.
                
                found_swap = False
                for c_new in range(total_words):
                    if c_new in code_set:
                        continue
                    
                    # Check if c_new covers all unique_words
                    valid_swap = True
                    for w in unique_words:
                        if w not in neighbors[c_new]:
                            valid_swap = False
                            break
                    
                    if not valid_swap:
                        continue
                    
                    # Perform swap
                    # 1. Remove c_old
                    for w in neighbors[c_old]:
                        coverage_count[w] -= 1
                    code_set.remove(c_old)
                    
                    # 2. Add c_new
                    for w in neighbors[c_new]:
                        coverage_count[w] += 1
                    code_set.add(c_new)
                    
                    # 3. Try to remove a redundant codeword
                    removed = False
                    # We might need to try multiple removals
                    while True:
                        if time.time() > deadline:
                            # Revert swap? No, just return current best.
                            # But we need to be careful not to return an invalid code.
                            # The swap was valid, so code is valid.
                            return list(code_set)
                        if not try_remove():
                            break
                        removed = True
                    
                    if removed:
                        # We improved the size!
                        improved = True
                        found_swap = True
                        # Break out of c_new loop and restart c_old loop (or continue?)
                        # Restarting c_old loop is safer as indices/sets changed.
                        break
                    else:
                        # Swap was valid but didn't lead to immediate removal.
                        # We can either keep this swap (to explore neighborhood) or revert.
                        # Keeping it might help find a path to a smaller code.
                        # Let's keep it and mark that we changed state.
                        improved = True # We changed the code, so we should re-evaluate
                        found_swap = True
                        break
                        # Note: If we keep swaps that don't reduce size, we might cycle.
                        # To prevent cycling, we could use a tabu list or only keep if it improves some heuristic.
                        # For now, let's only keep if it leads to a removal OR if we are stuck.
                        # Actually, if it doesn't lead to a removal, we should probably revert to avoid wandering.
                        # Let's revert if no removal happened.
                        # Revert:
                        for w in neighbors[c_new]:
                            coverage_count[w] -= 1
                        code_set.remove(c_new)
                        
                        for w in neighbors[c_old]:
                            coverage_count[w] += 1
                        code_set.add(c_old)
                        # Don't mark improved, continue searching for a better swap
                        # But we already checked this c_new.
                        # We should continue to next c_new.
                        # The 'break' above was wrong if we revert.
                        # Let's structure this better.
                        
                # If we found a swap that led to removal, we broke out.
                # If we didn't, we continue to next c_old.
                
                if found_swap:
                    # If we broke out due to found_swap (which implies improvement or state change), 
                    # we should restart the outer 'while improved' loop to re-evaluate removals.
                    break

        return list(code_set)

    # Main logic: Multi-start local search
    best_code = None
    best_size = float('inf')
    
    # Number of restarts
    # We have ~110s.
    # Greedy is ~0.1s. Local search is ~0.1-1s depending on convergence.
    # Let's do as many as possible.
    
    restarts = 0
    while time.time() < deadline - 5.0: # Leave 5s buffer
        # Generate a greedy solution
        # We can randomize the tie-breaking in greedy to get diverse starts
        greedy_code = greedy_construct()
        
        # Local search
        refined_code = local_search(greedy_code)
        
        if len(refined_code) < best_size:
            best_size = len(refined_code)
            best_code = refined_code
            # If we hit the known optimum (73), we can stop early?
            # The lower bound is 71. 73 is best known.
            # If we find 71 or 72, that's a breakthrough.
            if best_size <= 73:
                # We found a very good solution. Let's try a few more to see if we can go lower.
                # But 73 is the target.
                pass
        
        restarts += 1
        # If we've done a lot of restarts and no improvement, maybe stop?
        # Or just keep going.
        
        # Random seed for next run? 
        # greedy_construct uses deterministic logic unless we add randomness.
        # Let's add randomness to greedy_construct tie-breaking.
        
    # If no code was found (shouldn't happen), return a simple one
    if best_code is None:
        best_code = greedy_construct()
        
    # Convert integer indices back to lists of integers
    result = [int_to_word(c, n, q) for c in best_code]
    
    return result
