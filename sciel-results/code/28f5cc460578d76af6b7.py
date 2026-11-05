"""Improved solver for ternary covering code K_3(6,1) using Greedy + Swap-Remove Local Search."""
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Specifics for q=3, n=6, radius=1
    # Generalize slightly but optimize for this case
    num_words = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage lists for all possible codewords
    # cover_list[w] = list of indices of words covered by codeword w
    cover_list = []
    for w in all_words:
        covered = [word_to_idx[w]]
        for i in range(n):
            for v in range(q):
                if v != w[i]:
                    new_w = list(w)
                    new_w[i] = v
                    covered.append(word_to_idx[tuple(new_w)])
        cover_list.append(covered)
        
    # Greedy Construction
    def greedy_construct():
        uncovered = np.ones(num_words, dtype=bool)
        code = []
        # To speed up, we can precompute for each word which codewords cover it?
        # No, greedy picks best codeword. 
        # We need to evaluate 729 codewords each step.
        # Steps ~ 73. 73 * 729 * (cost to check overlap).
        # Cost to check overlap: sum of uncovered[cover_list[c]].
        # This is fast with numpy.
        
        while np.any(uncovered):
            if time.time() > deadline:
                return code # Return partial if timeout, but greedy is fast
            
            best_c = -1
            best_gain = -1
            # Optimization: Only consider codewords that cover at least one uncovered word
            # We can just iterate all.
            for c_idx in range(num_words):
                if c_idx in code: # O(N) check, slow. Use a set?
                    continue
                # Calculate gain
                # gain = number of uncovered words covered by c_idx
                # Using numpy sum is fast
                gain = np.sum(uncovered[cover_list[c_idx]])
                if gain > best_gain:
                    best_gain = gain
                    best_c = c_idx
            
            if best_c == -1:
                break # Should not happen if uncovered is not empty
            
            code.append(best_c)
            # Update uncovered
            uncovered[cover_list[best_c]] = False
            
        return code

    # Local Search with Swaps
    def local_search(code):
        code_set = set(code)
        cover_count = np.zeros(num_words, dtype=np.int32)
        
        # Initialize counts
        for c in code:
            cover_count[cover_list[c]] += 1
            
        best_code = code[:]
        best_len = len(code)
        
        # Helper: Check if c is redundant
        def is_redundant(c):
            # Redundant if all words covered by c have count >= 2
            # i.e. min(cover_count[cover_list[c]]) >= 2
            return np.min(cover_count[cover_list[c]]) >= 2

        # Helper: Find valid swap for c_remove
        def find_swap(c_remove):
            # Temporarily remove c_remove to find S_bad
            # S_bad = words where cover_count == 1 after removal
            # We can simulate:
            # words_to_check = cover_list[c_remove]
            # For each w in words_to_check:
            #   if cover_count[w] == 1: w is in S_bad
            # We need c_add such that S_bad is subset of cover_list[c_add]
            
            s_bad = []
            for w in cover_list[c_remove]:
                if cover_count[w] == 1:
                    s_bad.append(w)
            
            if not s_bad:
                # c_remove is redundant, handled by removal phase
                return None
            
            # Find c_add not in code_set that covers all s_bad
            # Optimization: 
            # A necessary condition: c_add must cover all words in s_bad.
            # We can iterate c_add candidates.
            # To make it faster, we can check:
            # For a candidate c_add, does it cover s_bad?
            # np.all(np.isin(s_bad, cover_list[c_add])) -- slow
            
            # Better:
            # Precompute for each word w, the list of codewords covering it? 
            # That's 729 * 13 ~ 9000 entries. 
            # Let's build `covered_by[w]` = list of codewords covering w.
            # Then candidates for c_add must be in intersection of covered_by[w] for w in s_bad.
            # Since s_bad is small (usually 1 or 2), this is fast.
            
            # But we don't have covered_by precomputed in this scope. Let's compute it once outside?
            # It's 729 lists.
            
            # Let's just iterate all c_add not in code_set.
            # 729 - |code| ~ 650 candidates.
            # For each, check if it covers s_bad.
            # s_bad size is small.
            
            for c_add in range(num_words):
                if c_add in code_set:
                    continue
                # Check if c_add covers all s_bad
                # Convert s_bad to set for fast lookup? Or just loop s_bad (small)
                valid = True
                for w in s_bad:
                    if w not in cover_list[c_add]:
                        valid = False
                        break
                if valid:
                    return c_add
            return None

        # Precompute covered_by for faster swap search?
        # Let's try without first, as s_bad is small.
        
        while time.time() < deadline:
            progress = False
            
            # Phase 1: Removals
            # Try to remove any redundant codeword
            # Randomize order to avoid bias
            current_code = list(code_set)
            random.shuffle(current_code)
            
            for c in current_code:
                if c not in code_set:
                    continue
                if is_redundant(c):
                    # Remove
                    cover_count[cover_list[c]] -= 1
                    code_set.remove(c)
                    progress = True
                    if len(code_set) < best_len:
                        best_len = len(code_set)
                        best_code = list(code_set)
            
            if not progress:
                # Phase 2: Swaps
                # Try to find a swap that enables further removals
                # We pick a random codeword to remove and find a valid replacement
                swap_found = False
                current_code = list(code_set)
                random.shuffle(current_code)
                
                for c_remove in current_code:
                    if time.time() > deadline:
                        break
                    if c_remove not in code_set:
                        continue
                    
                    c_add = find_swap(c_remove)
                    if c_add is not None:
                        # Perform swap
                        cover_count[cover_list[c_remove]] -= 1
                        cover_count[cover_list[c_add]] += 1
                        code_set.remove(c_remove)
                        code_set.add(c_add)
                        swap_found = True
                        progress = True
                        # We don't update best_len here because size didn't change
                        # But the new configuration might allow removals
                        break # Restart removal phase with new configuration
                
                if not swap_found:
                    # Stuck completely. Break or restart?
                    # For ILS, we would perturb. 
                    # For now, break and return best.
                    break
            
            # If we made progress, loop again to try more removals/swaps
            # If no progress in removal, we tried swaps. If swap found, we loop.
            # If no swap found, we break.

        return best_code

    # Main Logic
    # 1. Greedy
    initial_code = greedy_construct()
    
    # 2. Local Search
    best_code = local_search(initial_code)
    
    # 3. Iterated Local Search (if time permits)
    # Try a few random restarts of greedy?
    # Greedy is deterministic if we break ties consistently. 
    # Let's add randomization to greedy.
    
    best_len = len(best_code)
    
    while time.time() < deadline:
        # Randomized Greedy
        uncovered = np.ones(num_words, dtype=bool)
        code = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
            
            # Pick random among top gainers? Or just random?
            # Randomized greedy: pick from top 10% gainers?
            # Or just pick a random codeword that covers max uncovered?
            
            gains = []
            for c_idx in range(num_words):
                if c_idx in code:
                    continue
                gain = np.sum(uncovered[cover_list[c_idx]])
                if gain > 0:
                    gains.append((gain, c_idx))
            
            if not gains:
                break
                
            # Sort by gain, pick random from top K
            gains.sort(reverse=True)
            top_k = min(10, len(gains))
            _, best_c = random.choice(gains[:top_k])
            
            code.append(best_c)
            uncovered[cover_list[best_c]] = False
        
        if len(code) < best_len:
             # Run local search on this new candidate
             candidate = local_search(code)
             if len(candidate) < best_len:
                 best_len = len(candidate)
                 best_code = candidate

    return [list(all_words[c]) for c in best_code]
