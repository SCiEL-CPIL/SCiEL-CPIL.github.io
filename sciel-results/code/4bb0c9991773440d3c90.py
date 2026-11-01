import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Specific case q=3, n=6, radius=1
    # General fallback if parameters differ, though the task specifies this instance.
    if q != 3 or n != 6 or radius != 1:
        # Fallback to simple greedy for other cases
        total_words = q ** n
        def word_to_idx(word):
            idx = 0
            for i in range(n):
                idx = idx * q + word[i]
            return idx
        def idx_to_word(idx):
            word = [0] * n
            for i in range(n - 1, -1, -1):
                word[i] = idx % q
                idx //= q
            return word
        
        neighbors = [[] for _ in range(total_words)]
        for idx in range(total_words):
            w = idx_to_word(idx)
            neighbors[idx].append(idx)
            if radius >= 1:
                for pos in range(n):
                    orig = w[pos]
                    for new_val in range(q):
                        if new_val != orig:
                            w[pos] = new_val
                            nb_idx = word_to_idx(w)
                            neighbors[idx].append(nb_idx)
                    w[pos] = orig
        
        neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]
        
        def greedy_cover(restart_seed=0):
            rng = random.Random(restart_seed)
            uncovered = np.ones(total_words, dtype=bool)
            codewords = []
            nm = np.zeros((total_words, total_words), dtype=bool)
            for i in range(total_words):
                nm[i, neighbors_np[i]] = True
            
            while time.time() < deadline:
                if not uncovered.any():
                    break
                counts = nm.dot(uncovered.astype(np.int32))
                best_count = counts.max()
                if best_count == 0:
                    break
                best_cw = int(counts.argmax())
                codewords.append(best_cw)
                uncovered[neighbors_np[best_cw]] = False
            return codewords
        
        result = greedy_cover(0)
        # Simple redundancy removal
        coverage = np.zeros(total_words, dtype=np.int32)
        for cw in result:
            coverage[neighbors_np[cw]] += 1
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(result)):
                if time.time() > deadline:
                    break
                cw = result[i]
                if coverage[neighbors_np[cw]].min() >= 2:
                    coverage[neighbors_np[cw]] -= 1
                    result.pop(i)
                    changed = True
                    break
        return [idx_to_word(idx) for idx in result]

    # --- Optimized path for q=3, n=6, radius=1 ---
    total_words = 729
    
    def word_to_idx(word):
        idx = 0
        for i in range(6):
            idx = idx * 3 + word[i]
        return idx
        
    def idx_to_word(idx):
        word = [0] * 6
        for i in range(5, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word

    # Precompute neighbors
    neighbors = [[] for _ in range(total_words)]
    for idx in range(total_words):
        w = idx_to_word(idx)
        neighbors[idx].append(idx)
        for pos in range(6):
            orig = w[pos]
            for new_val in range(3):
                if new_val != orig:
                    w[pos] = new_val
                    nb_idx = word_to_idx(w)
                    neighbors[idx].append(nb_idx)
            w[pos] = orig
    
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]
    
    # Precompute neighbor matrix for fast coverage counting
    nm = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        nm[i, neighbors_np[i]] = True

    def get_initial_greedy(seed=0):
        rng = random.Random(seed)
        uncovered = np.ones(total_words, dtype=bool)
        codewords = []
        
        # Randomize tie-breaking slightly
        while time.time() < deadline:
            if not uncovered.any():
                break
            counts = nm.dot(uncovered.astype(np.int32))
            # Add small noise to break ties
            if rng.random() < 0.1:
                counts += rng.randint(0, 1, size=total_words).astype(np.int32)
            
            best_count = counts.max()
            if best_count == 0:
                break
            # Find all with max count
            max_indices = np.where(counts == best_count)[0]
            best_cw = int(rng.choice(max_indices))
            
            codewords.append(best_cw)
            uncovered[neighbors_np[best_cw]] = False
        
        return codewords

    def remove_redundant(codewords, coverage=None):
        if coverage is None:
            coverage = np.zeros(total_words, dtype=np.int32)
            for cw in codewords:
                coverage[neighbors_np[cw]] += 1
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Shuffle to avoid bias
            indices = list(range(len(codewords)))
            random.shuffle(indices)
            for i in indices:
                if time.time() > deadline:
                    break
                cw = codewords[i]
                # Check if removing cw leaves all its neighbors covered by others
                # coverage[nb] >= 2 for all nb in neighbors_np[cw]
                if coverage[neighbors_np[cw]].min() >= 2:
                    coverage[neighbors_np[cw]] -= 1
                    codewords.pop(i)
                    changed = True
                    break
        return codewords, coverage

    # 1. Get a good starting point
    best_codewords = get_initial_greedy(0)
    best_codewords, best_coverage = remove_redundant(best_codewords)
    best_len = len(best_codewords)
    
    # 2. Simulated Annealing
    # State: set of codewords
    # Objective: minimize len(codewords) while maintaining coverage
    # We track coverage counts.
    # Move: 
    #   - Remove a random codeword. If valid (min coverage in its ball >= 2), accept (improvement).
    #   - If invalid, we might accept it with probability exp(-delta/T) where delta is the "damage" (number of newly uncovered words).
    #   - Add a random codeword. This increases size, so we only accept if it allows a subsequent removal or if we are in a "repair" phase. 
    #     Actually, standard SA on set cover is tricky. 
    #     Alternative: 
    #     1. Remove random cw. If invalid, calculate set of uncovered words U.
    #     2. If U is empty, accept.
    #     3. Else, try to add a cw' that covers many of U. If found, swap cw -> cw'.
    
    current_codewords = best_codewords.copy()
    current_coverage = best_coverage.copy()
    current_len = len(current_codewords)
    
    T = 5.0
    cooling = 0.99
    min_T = 0.01
    
    rng_sa = random.Random(42)
    
    while time.time() < deadline and T > min_T:
        if not current_codewords:
            break
            
        # Pick a random codeword to remove
        idx = rng_sa.randint(0, len(current_codewords) - 1)
        cw = current_codewords[idx]
        
        # Tentatively remove
        current_coverage[neighbors_np[cw]] -= 1
        current_codewords.pop(idx)
        
        # Check validity
        # A word is uncovered if coverage[word] == 0
        # We only care about words in the ball of the removed cw, as others were covered before and are still covered by others? 
        # No, if a word was covered ONLY by cw, it is now uncovered.
        # Words in ball(cw) with coverage 0 are newly uncovered.
        newly_uncovered_mask = (current_coverage == 0) & (nm[cw, :] == True) # nm[cw, :] is True for neighbors of cw
        num_uncovered = np.sum(newly_uncovered_mask)
        
        if num_uncovered == 0:
            # Valid improvement
            current_len -= 1
            if current_len < best_len:
                best_len = current_len
                best_codewords = current_codewords.copy()
                best_coverage = current_coverage.copy()
        else:
            # Invalid move. Try to repair by adding a new codeword.
            # Find a codeword that covers the most newly uncovered words.
            # We can check all 729 candidates, or just a random subset for speed.
            # Let's check all 729, it's fast enough (729 * 13 ops ~ 10k ops)
            # counts = nm.dot(newly_uncovered_mask.astype(np.int32))
            # best_new = counts.argmax()
            
            # Optimization: Only consider candidates that cover at least 1 newly uncovered word.
            # And we want to maximize coverage of newly uncovered words.
            # Note: Adding a codeword increases len by 1. Net change: -1 (remove) + 1 (add) = 0.
            # So we are looking for a swap that maintains validity (or improves if we can remove more later).
            # Actually, if we just swap, len stays same. We want to eventually reduce len.
            # So we should only accept the swap if it leads to a state where we CAN remove more.
            # This is complex. 
            # Simpler heuristic: Accept swap if it covers ALL newly uncovered words. 
            # Then we are back to a valid state with same size. 
            # This doesn't improve size directly, but might lead to a different local optimum.
            
            # Let's try to find a cw' such that neighbors(cw') covers all newly_uncovered.
            # This is a set cover sub-problem for a small set.
            # If found, we swap. 
            # To make progress, we should bias towards swaps that allow further removals.
            
            # For now, let's just reject if we can't fix it immediately with 1 add, 
            # OR accept with probability exp(-num_uncovered / T) and keep the invalid state? 
            # No, we must maintain validity for the final answer. 
            # We can maintain an "invalid" state in SA as long as we track the penalty.
            
            # Let's try: Accept the removal (invalid state) with prob exp(-num_uncovered / T).
            # If accepted, we are in an invalid state. We MUST fix it before the next step or final output.
            # This is risky. 
            
            # Better: Try to find a replacement.
            # Calculate coverage gain for each candidate x: gain[x] = sum(newly_uncovered_mask[x's neighbors])
            # We want gain[x] >= num_uncovered.
            
            # Vectorized check
            # gain = nm.dot(newly_uncovered_mask.astype(np.int32))
            # candidates = np.where(gain == num_uncovered)[0]
            
            # If candidates exist, pick one randomly.
            # If not, reject the move (restore state).
            
            # To save time, we only do this if num_uncovered is small (e.g., < 5).
            if num_uncovered <= 5:
                gain = nm.dot(newly_uncovered_mask.astype(np.int32))
                # We need a candidate that covers ALL newly uncovered words.
                # Since newly_uncovered words are in ball(cw), and any neighbor of a newly uncovered word 
                # might not cover it if it's not in its ball.
                # Wait, gain[x] counts how many newly uncovered words are in ball(x).
                # If gain[x] == num_uncovered, then ball(x) covers all newly uncovered words.
                candidates = np.where(gain == num_uncovered)[0]
                
                if len(candidates) > 0:
                    new_cw = int(rng_sa.choice(candidates))
                    # Perform swap
                    current_codewords.append(new_cw)
                    current_coverage[neighbors_np[new_cw]] += 1
                    # current_len unchanged
                    # This move is neutral in size, but changes the configuration.
                    # We accept it to escape local minima.
                else:
                    # Reject move
                    current_codewords.insert(idx, cw)
                    current_coverage[neighbors_np[cw]] += 1
            else:
                # Reject move
                current_codewords.insert(idx, cw)
                current_coverage[neighbors_np[cw]] += 1
        
        T *= cooling
        
        # Occasionally try to add a random codeword to explore space?
        # Or just rely on swaps.
        
        # Polishing step every few iterations?
        # If we have time, run remove_redundant on current state?
        # It might be expensive.
        
    # Final polish
    best_codewords, _ = remove_redundant(best_codewords)
    
    # If we found 73, we can stop early?
    # But we have time, let's keep polishing or do more SA.
    
    # If we have significant time left, do more restarts or SA iterations
    if time.time() < deadline - 10:
        # Run a second SA phase with different seed
        T = 3.0
        cooling = 0.995
        rng_sa2 = random.Random(123)
        
        # Start from best found
        current_codewords = best_codewords.copy()
        current_coverage = np.zeros(total_words, dtype=np.int32)
        for cw in current_codewords:
            current_coverage[neighbors_np[cw]] += 1
        current_len = len(current_codewords)
        
        while time.time() < deadline - 2 and T > min_T:
            if not current_codewords:
                break
            idx = rng_sa2.randint(0, len(current_codewords) - 1)
            cw = current_codewords[idx]
            
            current_coverage[neighbors_np[cw]] -= 1
            current_codewords.pop(idx)
            
            newly_uncovered_mask = (current_coverage == 0) & (nm[cw, :] == True)
            num_uncovered = np.sum(newly_uncovered_mask)
            
            if num_uncovered == 0:
                current_len -= 1
                if current_len < best_len:
                    best_len = current_len
                    best_codewords = current_codewords.copy()
                    best_coverage = current_coverage.copy()
            else:
                if num_uncovered <= 5:
                    gain = nm.dot(newly_uncovered_mask.astype(np.int32))
                    candidates = np.where(gain == num_uncovered)[0]
                    if len(candidates) > 0:
                        new_cw = int(rng_sa2.choice(candidates))
                        current_codewords.append(new_cw)
                        current_coverage[neighbors_np[new_cw]] += 1
                    else:
                        current_codewords.insert(idx, cw)
                        current_coverage[neighbors_np[cw]] += 1
                else:
                    current_codewords.insert(idx, cw)
                    current_coverage[neighbors_np[cw]] += 1
            
            T *= cooling
            
    # Final cleanup
    best_codewords, _ = remove_redundant(best_codewords)
    
    return [idx_to_word(idx) for idx in best_codewords]
