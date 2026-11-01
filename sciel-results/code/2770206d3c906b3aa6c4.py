import numpy as np
import time
import itertools
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute neighbors for all words in {0..q-1}^n
    space_size = q ** n
    
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # neighbors[i] is a list of indices j such that dist(i, j) <= radius
    # For radius=1, this is i itself + words differing in 1 position
    neighbors = [[] for _ in range(space_size)]
    for i in range(space_size):
        w = idx_to_word(i)
        neighbors[i].append(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    w[pos] = new_val
                    neighbors[i].append(word_to_idx(w))
            w[pos] = orig_val
            
    # Convert neighbors to sets for faster "covers" checks if needed, 
    # but lists are fine for counting.
    
    def greedy_construct(random_seed):
        random.seed(random_seed)
        covered = [False] * space_size
        codewords = []
        # To make it faster, we can pick random uncovered words
        uncovered_indices = list(range(space_size))
        random.shuffle(uncovered_indices)
        
        # Actually, standard greedy: pick word covering most uncovered.
        # But for small space, we can just pick a random uncovered word 
        # and a random codeword covering it? No, that's not greedy.
        # Greedy: Pick c that maximizes |N(c) \ Covered|.
        
        # Optimization: 
        # We can maintain a list of uncovered words.
        # For each candidate c, count how many of its neighbors are uncovered.
        
        # Given space_size=729, and we add ~73 words, 
        # 73 * 729 checks is ~50k checks, which is fast.
        
        while any(covered): # Wait, while there are uncovered
            if not any(not c for c in covered):
                break
            
            best_c = -1
            best_count = -1
            
            # We can iterate through all possible codewords
            # To speed up, we can just pick a random uncovered word u,
            # and then pick the best c covering u.
            # This is a "lazy" greedy which is often faster and comparable.
            
            # Let's use the lazy greedy:
            # 1. Pick random uncovered word u.
            # 2. Find c in {0..space_size-1} not in codewords 
            #    that covers u and maximizes coverage of other uncovered words.
            
            # To pick random uncovered word:
            # We can keep a list of uncovered indices and pop from it.
            
            if not uncovered_indices:
                break
            
            u = uncovered_indices[0] # Or random.choice
            # Let's pick random from remaining
            idx_in_list = random.randint(0, len(uncovered_indices) - 1)
            u = uncovered_indices[idx_in_list]
            
            # Find best c covering u
            # Candidates are all c such that u in neighbors[c]
            # We can precompute reverse neighbors? 
            # Or just iterate all c and check if u in neighbors[c].
            # Since radius is small, neighbors[c] is small.
            # But checking "u in neighbors[c]" is O(1) if we use a set, or O(13) if list.
            
            # Let's precompute "predecessors" or just use the fact that 
            # if u is covered by c, then c is in neighbors[u] (symmetric).
            # So candidates for c are neighbors[u].
            
            candidates = neighbors[u]
            
            for c in candidates:
                if c in codewords: # We need a set for O(1) check
                    continue
                
                # Count how many uncovered words c covers
                count = 0
                for nb in neighbors[c]:
                    if not covered[nb]:
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_c = c
            
            if best_c == -1:
                # Should not happen if u is uncovered and we consider all neighbors
                # But if all neighbors are already in codewords, we have a problem?
                # No, if u is uncovered, no codeword covers it.
                # So none of its neighbors can be in codewords?
                # Wait. If c is in codewords, it covers N(c).
                # If u is uncovered, u is not in N(c) for any c in codewords.
                # So u is not in neighbors[c] for any c in codewords.
                # Thus, no c in codewords is in neighbors[u].
                # So best_c will always be found.
                
                # Actually, we check `if c in codewords`. 
                # If u is uncovered, then for any c in codewords, u is not in neighbors[c].
                # Since dist is symmetric, c is not in neighbors[u].
                # So all candidates in neighbors[u] are NOT in codewords.
                # So best_c will be set.
                
                pass
            
            # Add best_c
            codewords.append(best_c)
            codewords_set.add(best_c)
            
            # Update covered and uncovered_indices
            for nb in neighbors[best_c]:
                if not covered[nb]:
                    covered[nb] = True
                    # Remove from uncovered_indices
                    # This is O(N) if we do linear search. 
                    # For 729 items, it's fine.
                    try:
                        uncovered_indices.remove(nb)
                    except ValueError:
                        pass

        return codewords

    # We need a set for codewords for O(1) lookup in greedy_construct
    # But greedy_construct returns a list. 
    # Let's modify greedy_construct to use a local set.
    
    def greedy_construct_v2(random_seed):
        random.seed(random_seed)
        covered = [False] * space_size
        codewords = []
        codewords_set = set()
        uncovered_indices = list(range(space_size))
        
        while uncovered_indices:
            # Pick random uncovered word
            idx_in_list = random.randint(0, len(uncovered_indices) - 1)
            u = uncovered_indices[idx_in_list]
            
            # Candidates are neighbors of u
            candidates = neighbors[u]
            
            best_c = -1
            best_count = -1
            
            for c in candidates:
                # c is guaranteed not to be in codewords_set because u is uncovered
                # (if c were in codewords_set, u would be covered)
                
                count = 0
                for nb in neighbors[c]:
                    if not covered[nb]:
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_c = c
            
            if best_c == -1:
                # Fallback
                best_c = candidates[0]
            
            codewords.append(best_c)
            codewords_set.add(best_c)
            
            # Update coverage
            for nb in neighbors[best_c]:
                if not covered[nb]:
                    covered[nb] = True
                    try:
                        uncovered_indices.remove(nb)
                    except ValueError:
                        pass

        return codewords

    def optimize(codewords, random_seed):
        random.seed(random_seed)
        # Convert to set for fast lookup
        codewords_set = set(codewords)
        # Calculate coverage counts
        counts = np.zeros(space_size, dtype=np.int32)
        for c in codewords:
            for nb in neighbors[c]:
                counts[nb] += 1
        
        # Local Search Loop
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to remove redundant codewords
            # A codeword c is redundant if all its neighbors have count > 1
            # We iterate through a copy of the list
            to_remove = []
            for c in codewords:
                # Check if any neighbor has count == 1
                redundant = True
                for nb in neighbors[c]:
                    if counts[nb] == 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(c)
            
            if to_remove:
                for c in to_remove:
                    codewords.remove(c)
                    codewords_set.remove(c)
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                improved = True
                continue # Restart the pass as counts changed
            
            # 2. If no removals, try swaps
            # Pick a random codeword
            if not codewords:
                break
                
            c_idx = random.randint(0, len(codewords) - 1)
            c = codewords[c_idx]
            
            # Find essential neighbors (count == 1)
            essential = [nb for nb in neighbors[c] if counts[nb] == 1]
            
            if not essential:
                # Should have been removed in step 1
                # But maybe counts changed? No, we continue if removed.
                # So if we are here, essential is not empty.
                pass
            
            # Try to find a replacement c' that covers all essential neighbors
            # and is not in codewords_set
            # We sample random words
            found_swap = False
            for _ in range(50): # Try 50 random candidates
                if time.time() > deadline:
                    break
                
                c_prime = random.randint(0, space_size - 1)
                if c_prime in codewords_set:
                    continue
                
                # Check if c_prime covers all essential neighbors
                covers_all = True
                for nb in essential:
                    if nb not in neighbors[c_prime]:
                        covers_all = False
                        break
                
                if not covers_all:
                    continue
                
                # Check if swapping c with c_prime is valid
                # We need to ensure that after removing c and adding c_prime,
                # no word becomes uncovered.
                # Words that lose coverage: N(c) \ N(c_prime)
                # Words that gain coverage: N(c_prime) \ N(c)
                
                # Simulate
                # Decrement counts for N(c)
                # Increment counts for N(c_prime)
                # Check if any count becomes 0
                # If so, revert.
                
                # Optimization: 
                # Only words in N(c) \ N(c_prime) are at risk of becoming 0.
                # Words in N(c_prime) \ N(c) will increase.
                # Words in intersection stay same.
                
                risk_words = [nb for nb in neighbors[c] if nb not in neighbors[c_prime]]
                
                valid = True
                for nb in risk_words:
                    # Current count is counts[nb]
                    # After removing c, it becomes counts[nb] - 1
                    # After adding c_prime, it stays counts[nb] - 1 (since c_prime doesn't cover it)
                    if counts[nb] - 1 < 1:
                        valid = False
                        break
                
                if valid:
                    # Perform swap
                    codewords[c_idx] = c_prime
                    codewords_set.remove(c)
                    codewords_set.add(c_prime)
                    
                    for nb in neighbors[c]:
                        counts[nb] -= 1
                    for nb in neighbors[c_prime]:
                        counts[nb] += 1
                    
                    found_swap = True
                    improved = True
                    break
            
            # If no swap found, we are stuck. 
            # We could try a "kick" (remove random, re-add greedy) 
            # but for now, we break to avoid infinite loop of failed swaps.
            if not found_swap:
                # To escape local minimum, we can do a perturbation
                # Remove 2 random words and re-add 2 greedily?
                # Or just break and rely on restarts.
                # Let's try a small perturbation: 
                # Remove a random word (even if it causes holes) and re-add greedily.
                # This is risky but can escape.
                
                # Actually, just breaking is safer. The main loop will restart.
                break

        return codewords

    best_codewords = None
    best_size = float('inf')
    
    # Run multiple restarts
    max_restarts = 200 # We have time
    
    for i in range(max_restarts):
        if time.time() > deadline:
            break
        
        seed = 1000 + i
        # 1. Construct
        init_codewords = greedy_construct_v2(seed)
        
        # 2. Optimize
        optimized = optimize(init_codewords, seed)
        
        if len(optimized) < best_size:
            best_size = len(optimized)
            best_codewords = optimized
            
            # If we found 73, we can stop early? 
            # The best known is 73. 
            # If we find 73, it's optimal (or close).
            # But we should verify.
            # For now, just keep going to see if we can get lower (unlikely) 
            # or just save time.
            if best_size <= 73:
                # We can try a few more to be sure, or just break.
                # Let's break to save time for other potential improvements.
                pass

    if best_codewords is None:
        # Fallback
        best_codewords = greedy_construct_v2(42)
        best_codewords = optimize(best_codewords, 42)
        
    # Convert indices back to words
    result = [idx_to_word(idx) for idx in best_codewords]
    return result
