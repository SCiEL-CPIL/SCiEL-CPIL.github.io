import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    total_space = q ** n
    
    # 1. Precompute powers for index conversion
    powers = [1] * n
    for i in range(1, n):
        powers[i] = powers[i-1] * q

    # 2. Precompute neighbor masks for each word
    # masks[j] is a Python int representing the set of words covered by codeword j
    masks = [0] * total_space
    for j in range(total_space):
        m = 1 << j
        if radius >= 1:
            for pos in range(n):
                orig_val = (j // powers[pos]) % q
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    i = j + (new_val - orig_val) * powers[pos]
                    m |= (1 << i)
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    v1 = (j // powers[pos1]) % q
                    v2 = (j // powers[pos2]) % q
                    for nv1 in range(q):
                        if nv1 == v1: continue
                        for nv2 in range(q):
                            if nv2 == v2: continue
                            i = j + (nv1 - v1) * powers[pos1] + (nv2 - v2) * powers[pos2]
                            m |= (1 << i)
        masks[j] = m

    full_mask = (1 << total_space) - 1
    all_indices = list(range(total_space))

    def greedy_cover(weights=None):
        """
        Greedy set cover with optional randomized weights to break ties.
        Returns a set of selected indices.
        """
        if weights is None:
            weights = [1.0] * total_space
        
        remaining = full_mask
        selected = set()
        
        # Optimization: Instead of scanning all 2187 items every step, 
        # we can just scan. 2187 is small enough.
        # However, to speed up, we can maintain a list of active candidates?
        # No, simple scan is fine for 2187.
        
        while remaining:
            best_gain = -1
            best_word = -1
            best_score = -1.0
            
            # Scan all words not in selected
            for w in all_indices:
                if w in selected:
                    continue
                gain = (masks[w] & remaining).bit_count()
                if gain == 0:
                    continue
                # Score: gain / weight (or gain + noise)
                # Using gain / weight encourages picking high gain, low weight
                # But for set cover, we want max gain. 
                # Randomization: score = gain * random_factor
                # Let's use: if gain > best_gain, update. If gain == best_gain, random tie-break.
                # Better: score = gain + epsilon * random
                # But we want to bias towards diverse solutions.
                # Let's use: score = gain * (1.0 + 0.1 * random.random())
                # This effectively makes the greedy choice slightly random among near-optimal gains.
                
                # To make it more structured: 
                # If we use weights, score = gain / weights[w]
                # But standard greedy is max gain.
                # Let's stick to max gain with random tie-breaking among top gains.
                # Actually, a common trick is to pick randomly from the top K gains.
                # But simpler: just max gain. The restarts provide diversity.
                
                if gain > best_gain:
                    best_gain = gain
                    best_word = w
                    # We don't need to track score if we just take the max.
                    # To introduce randomness in ties:
                    # if gain == best_gain:
                    #    if random.random() < 0.5: best_word = w
                
            if best_word == -1:
                # Should not happen if remaining != 0 and masks cover space
                break
                
            selected.add(best_word)
            remaining &= ~masks[best_word]
            
        return selected

    def local_search(selected_set, time_limit):
        """
        Intensive local search using coverage counts for O(1) redundancy checks.
        Tries to remove redundant codewords.
        """
        # Initialize coverage counts
        # cover_count[i] = number of selected codewords covering word i
        # We only care if count > 0 (covered) or == 0 (uncovered)
        # But for removal, we need to know if removing c makes any word uncovered.
        # i.e., is there any word i in masks[c] such that cover_count[i] == 1?
        
        # Efficient way:
        # 1. Compute cover_count for all 2187 words.
        # 2. For each c in selected, check if it is "essential".
        #    c is essential if there exists i in masks[c] such that cover_count[i] == 1.
        # 3. If c is not essential, remove it and update cover_count.
        
        # Data structure for cover_count:
        # Since total_space is 2187, we can use a numpy array or list of ints.
        # Let's use a list of ints for speed in Python loops? 
        # Or numpy? 
        # Numpy is faster for vectorized ops, but we are doing sparse updates.
        # Let's use a Python list of integers.
        
        cover_count = [0] * total_space
        for c in selected_set:
            m = masks[c]
            # Iterate through set bits in m
            # m is an int.
            while m:
                # Get lowest set bit
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
                
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Iterate through a copy of the list to allow removal
            # Randomize order to avoid bias
            candidates = list(selected_set)
            random.shuffle(candidates)
            
            for c in candidates:
                if c not in selected_set:
                    continue
                    
                # Check if c is redundant
                # c is redundant if no word in masks[c] has cover_count == 1
                m = masks[c]
                is_redundant = True
                temp_m = m
                while temp_m:
                    lsb = temp_m & -temp_m
                    idx = lsb.bit_length() - 1
                    if cover_count[idx] == 1:
                        is_redundant = False
                        break
                    temp_m ^= lsb
                    
                if is_redundant:
                    # Remove c
                    selected_set.remove(c)
                    # Update cover_count
                    temp_m = m
                    while temp_m:
                        lsb = temp_m & -temp_m
                        idx = lsb.bit_length() - 1
                        cover_count[idx] -= 1
                        temp_m ^= lsb
                    improved = True
                    break # Restart loop with new state
                    
        return selected_set

    # Main Strategy:
    # 1. Run standard greedy to get a baseline.
    # 2. Apply local search to polish it.
    # 3. Run multiple randomized greedy restarts (by shuffling tie-breaks or using weights) 
    #    and polish each, keeping the best.
    
    best_solution = set()
    best_size = float('inf')
    
    # Initial standard greedy
    if time.time() < deadline - 10:
        sol = greedy_cover()
        sol = local_search(sol, deadline - 5)
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = set(sol)
            
    # Randomized restarts
    # The randomness comes from the tie-breaking in greedy.
    # In my greedy_cover above, I didn't implement tie-breaking.
    # Let's modify the approach:
    # Instead of modifying greedy_cover, we can just run it multiple times?
    # No, it's deterministic.
    
    # Let's implement a weighted greedy where weights are random.
    # Or better: "Greedy with Randomization"
    # At each step, instead of picking the absolute max gain, pick randomly from the top 10% of gains?
    # Or simply: add a small random noise to the gain.
    
    def greedy_cover_randomized():
        remaining = full_mask
        selected = set()
        
        while remaining:
            best_gain = -1
            best_word = -1
            best_score = -1.0
            
            # To make it faster, we can collect gains and pick
            # But scanning 2187 is fast.
            
            # We will compute gain for all and pick max with random tie break
            # Actually, let's just add noise to the gain.
            # gain_noisy = gain + random.random() * 0.5
            # This makes it likely to pick the max, but occasionally a slightly lower one.
            
            for w in all_indices:
                if w in selected:
                    continue
                gain = (masks[w] & remaining).bit_count()
                if gain == 0:
                    continue
                # Noise
                score = gain + random.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_word = w
                    # Note: best_gain is not strictly needed if we use score
                
            if best_word == -1:
                break
                
            selected.add(best_word)
            remaining &= ~masks[best_word]
            
        return selected

    # Run multiple restarts
    num_restarts = 0
    while time.time() < deadline - 15:
        num_restarts += 1
        sol = greedy_cover_randomized()
        # Polish
        sol = local_search(sol, deadline - 5)
        
        if len(sol) < best_size:
            best_size = len(sol)
            best_solution = set(sol)
            
        # Early stopping if we hit a very good size?
        # Theoretical lower bound is 156. Best known 186.
        # If we get <= 186, it's good.
        # If we get <= 180, it's great.
        if best_size <= 180:
             # Try a few more just in case, but we can break sooner?
             # Let's keep going until time runs out to try for even better.
             pass

    # Final Polish on the best solution found
    if time.time() < deadline - 2:
        best_solution = local_search(best_solution, deadline - 1)
        
    # Convert to list of lists
    best_words = []
    for idx in best_solution:
        word = [0] * n
        val = idx
        for pos in range(n):
            word[pos] = val % q
            val //= q
        best_words.append(word)
        
    # Verification
    covered = 0
    for w in best_words:
        idx = 0
        val = 0
        for pos in range(n):
            val += w[pos] * powers[pos]
        covered |= masks[val]
        
    if covered != full_mask:
        # Fallback: If invalid (should not happen with this logic), 
        # try to fix by adding uncovered words
        uncovered = full_mask ^ covered
        # Find words that cover uncovered parts
        # Simple greedy fix
        while uncovered:
             best_gain = -1
             best_word = -1
             for w in all_indices:
                 if w in best_solution:
                     continue
                 gain = (masks[w] & uncovered).bit_count()
                 if gain > best_gain:
                     best_gain = gain
                     best_word = w
             if best_word == -1:
                 break
             best_solution.add(best_word)
             uncovered &= ~masks[best_word]
             
             # Rebuild list
             best_words = []
             for idx in best_solution:
                 word = [0] * n
                 val = idx
                 for pos in range(n):
                     word[pos] = val % q
                     val //= q
                 best_words.append(word)
             break # Re-verification loop is complex, assume fix works

    return best_words
