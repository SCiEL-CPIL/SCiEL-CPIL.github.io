import time
import random
import numpy as np

def construct(q, n, radius):
    # q=3, n=6, radius=1
    # Total words = 3**6 = 729
    total_words = q ** n
    indices = list(range(total_words))
    
    # Precompute neighbors for each word (Hamming distance <= 1)
    # A word is a tuple of n ints
    words = []
    for i in range(total_words):
        # Convert index to base-q tuple
        w = [0] * n
        temp = i
        for j in range(n):
            w[j] = temp % q
            temp //= q
        words.append(tuple(w))
        
    # Map index to word and word to index
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # neighbors[i] is a list of indices j such that dist(words[i], words[j]) <= 1
    neighbors = []
    for i in range(total_words):
        w = words[i]
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w_list = list(w)
                new_w_list[pos] = val
                new_w = tuple(new_w_list)
                neigh.append(word_to_idx[new_w])
        neighbors.append(neigh)
    
    # Initial Code: Tetracode direct sum
    # Tetracode: 9 words of length 4
    tetracode = []
    for a in range(q):
        for b in range(q):
            c = (a + b) % q
            d = (a + 2 * b) % q
            tetracode.append((a, b, c, d))
            
    # Expand to length 6 by adding 2 more coordinates
    initial_code_indices = []
    for h in tetracode:
        for x in range(q):
            for y in range(q):
                full_word = h + (x, y)
                idx = word_to_idx[full_word]
                initial_code_indices.append(idx)
                
    C = initial_code_indices
    C_set = set(C)
    cover_count = np.zeros(total_words, dtype=np.int32)
    
    # Calculate initial cover counts
    for c in C:
        for nb in neighbors[c]:
            cover_count[nb] += 1
            
    def can_remove(c_idx):
        """Check if removing c_idx leaves all words covered."""
        # If any neighbor has cover_count == 1, it will become 0, so we cannot remove.
        # We only need to check neighbors of c_idx.
        for nb in neighbors[c_idx]:
            if cover_count[nb] == 1:
                return False
        return True

    def greedy_reduce():
        """Remove any codeword that is not essential."""
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of C to allow removal during iteration
            for c in list(C):
                if can_remove(c):
                    # Remove c
                    cover_count[neighbors[c]] -= 1
                    C.remove(c)
                    C_set.remove(c)
                    changed = True

    def is_valid():
        return np.min(cover_count) >= 1

    # SA Parameters
    start_time = time.time()
    deadline = 115.0 # 5s margin for 120s limit
    
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    
    best_C = C.copy()
    best_cover_count = cover_count.copy()
    best_size = len(C)
    
    Lambda = 10.0
    
    # Initial greedy reduction
    greedy_reduce()
    if len(C) < best_size:
        best_size = len(C)
        best_C = C.copy()
        best_cover_count = cover_count.copy()

    while time.time() < deadline:
        # Pick a random move
        # 80% Remove, 20% Swap
        if random.random() < 0.8:
            if not C:
                break
            c_old = random.choice(C)
            
            # Check if we can remove it validly (cover_count >= 1 for all neighbors)
            # If yes, delta_cost = -1 (size decreases)
            # If no, delta_cost = -1 + Lambda * newly_uncovered
            
            # Calculate newly uncovered
            newly_uncovered = 0
            for nb in neighbors[c_old]:
                if cover_count[nb] == 1:
                    newly_uncovered += 1
            
            delta_cost = -1.0 + Lambda * newly_uncovered
            
            if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                # Accept
                cover_count[neighbors[c_old]] -= 1
                C.remove(c_old)
                C_set.remove(c_old)
                
                # Greedy reduce to clean up
                greedy_reduce()
                
                # Update best if valid
                if is_valid():
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_cover_count = cover_count.copy()
        else:
            if not C:
                break
            c_old = random.choice(C)
            # Pick a random c_new not in C
            while True:
                c_new = random.randint(0, total_words - 1)
                if c_new not in C_set:
                    break
            
            # Calculate change
            # Remove c_old, Add c_new
            # newly_uncovered: neighbors of c_old that had count 1 and are NOT covered by c_new
            # new_covered: neighbors of c_new that had count 0 (if any, but we start valid usually)
            # Actually, we just need to check the union of neighbors.
            
            # For efficiency, we can just compute the new cover counts for the affected neighbors
            # and check if any become 0.
            
            # Let's just do the swap on a temporary basis for the check?
            # Or just calculate delta.
            
            # Neighbors of c_old that will lose 1 count
            # Neighbors of c_new that will gain 1 count
            
            # A word is uncovered if:
            # 1. It is a neighbor of c_old, has count 1, and is NOT a neighbor of c_new.
            # 2. It is NOT a neighbor of c_old, has count 0 (impossible if currently valid), and is not a neighbor of c_new.
            
            # Since we maintain validity in the "best" state, but current state might be invalid during SA,
            # we should track the actual number of uncovered words.
            
            # Let's stick to the previous logic but optimized:
            # Count how many words become uncovered.
            # A word w is uncovered after swap if:
            # - w is in N(c_old) \ N(c_new) and cover_count[w] == 1
            # - w is in N(c_new) \ N(c_old) and cover_count[w] == 0 (if we allow invalid states)
            # - w is in N(c_old) \cap N(c_new) and cover_count[w] == 0 (impossible if valid before)
            
            # Let's just compute the new cover_count for the affected neighbors and count zeros.
            # Affected neighbors = N(c_old) U N(c_new)
            
            affected = set(neighbors[c_old])
            affected.update(neighbors[c_new])
            
            newly_uncovered = 0
            # We need to know the new cover count for each affected word.
            # new_count[w] = cover_count[w] - (1 if w in N(c_old) else 0) + (1 if w in N(c_new) else 0)
            
            # To avoid creating a dict, we can just iterate.
            # But we need to check if new_count[w] == 0.
            
            # Optimization:
            # 1. Words in N(c_old) \ N(c_new): new_count = old_count - 1. Uncovered if old_count == 1.
            # 2. Words in N(c_new) \ N(c_old): new_count = old_count + 1. Uncovered if old_count == -1 (impossible) or if we are in invalid state.
            #    If we allow invalid states, we need to track uncovered count.
            
            # Let's assume we track `uncovered_count` globally?
            # No, let's just check validity of the new state.
            
            # Simpler approach for Swap:
            # Perform swap on cover_count, check min, then revert if rejected.
            # This is O(19+19) operations.
            
            for nb in neighbors[c_old]:
                cover_count[nb] -= 1
            for nb in neighbors[c_new]:
                cover_count[nb] += 1
                
            # Check if valid
            # We can check min of cover_count, but that's O(729).
            # Or we can check the affected neighbors.
            # If any affected neighbor has 0, it's invalid.
            # But what if a word NOT in affected neighbors was 0?
            # If the previous state was valid, only affected neighbors can become 0.
            # If the previous state was invalid, we might have other 0s.
            # To be safe and simple, let's just check `np.min(cover_count) >= 1`.
            # With 729 elements, this is fast enough.
            
            valid = np.min(cover_count) >= 1
            
            if valid:
                delta_cost = 0.0 # Size unchanged, validity maintained
            else:
                # Count uncovered
                uncovered = np.sum(cover_count == 0)
                delta_cost = Lambda * uncovered # Approximate, or track delta uncovered
                
            if delta_cost == 0 or random.random() < np.exp(-delta_cost / T):
                # Accept
                C.remove(c_old)
                C_set.remove(c_old)
                C.append(c_new)
                C_set.add(c_new)
                
                # Greedy reduce
                greedy_reduce()
                
                if is_valid():
                    if len(C) < best_size:
                        best_size = len(C)
                        best_C = C.copy()
                        best_cover_count = cover_count.copy()
            else:
                # Reject: Revert cover_count
                for nb in neighbors[c_old]:
                    cover_count[nb] += 1
                for nb in neighbors[c_new]:
                    cover_count[nb] -= 1
                # C and C_set unchanged
                
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Return best valid code found
    result = []
    for idx in best_C:
        result.append(list(words[idx]))
        
    return result
