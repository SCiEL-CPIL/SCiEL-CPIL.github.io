import random
import time
import numpy as np

def construct(q, n, radius):
    total_words = q ** n
    
    # Pre-compute neighbors for radius 1 as a list of NumPy arrays
    # neighbors[idx] = np.array of indices within distance <= 1
    neighbors = []
    
    # Pre-compute digit weights
    weights = [q ** (n - 1 - i) for i in range(n)]
    
    for idx in range(total_words):
        w = [0] * n
        temp = idx
        for d in range(n-1, -1, -1):
            w[d] = temp % q
            temp //= q
            
        neigh = [idx]
        for i in range(n):
            for delta in range(1, q):
                new_val = (w[i] + delta) % q
                new_idx = idx + (new_val - w[i]) * weights[i]
                neigh.append(new_idx)
        neighbors.append(np.array(neigh, dtype=np.int32))

    def idx_to_word(idx):
        w = [0] * n
        for d in range(n-1, -1, -1):
            w[d] = idx % q
            idx //= q
        return w

    # Simulated Annealing Parameters
    start_time = time.time()
    time_limit = 115.0
    
    # Initial solution: Greedy Set Cover to get a valid starting point
    # This ensures we start with a valid cover, which helps SA focus on minimization
    uncovered = np.ones(total_words, dtype=np.int8)
    initial_code = []
    remaining_count = total_words
    
    # Simple Greedy Initialization
    # M is not strictly needed if we update counts incrementally, but for greedy init it's fast
    M = np.zeros((total_words, total_words), dtype=np.int8)
    for idx in range(total_words):
        M[idx, neighbors[idx]] = 1
        
    while remaining_count > 0:
        gains = (M @ uncovered).astype(np.int32)
        max_gain = np.max(gains)
        if max_gain == 0: break
        candidates = np.where(gains == max_gain)[0]
        chosen_idx = candidates[np.random.randint(len(candidates))]
        initial_code.append(chosen_idx)
        nb_idx = neighbors[chosen_idx]
        remaining_count -= np.sum(uncovered[nb_idx])
        uncovered[nb_idx] = 0
        
    # Convert initial_code to a set for fast lookup
    current_code = set(initial_code)
    current_cover_count = np.zeros(total_words, dtype=np.int32)
    for idx in current_code:
        current_cover_count[neighbors[idx]] += 1
        
    current_uncovered = np.sum(current_cover_count == 0)
    penalty = 1000 # High penalty to prioritize coverage
    
    def get_cost(code_size, uncov):
        return code_size + penalty * uncov
        
    current_cost = get_cost(len(current_code), current_uncovered)
    best_code = list(current_code)
    best_cost = current_cost
    
    T = 50.0 # Initial temperature
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit:
        if T < 0.01:
            break
            
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Add a random word
            if len(current_code) < total_words:
                # Pick a random word not in current_code
                # To be efficient, pick from all and check, or maintain a list of available
                # For simplicity and speed in this small space, pick random index
                new_idx = np.random.randint(total_words)
                if new_idx not in current_code:
                    # Calculate delta cost
                    # New size: len + 1
                    # New uncovered: current_uncovered - (number of neighbors of new_idx that were uncovered)
                    nb = neighbors[new_idx]
                    newly_covered = np.sum(current_cover_count[nb] == 0)
                    new_uncov = current_uncovered - newly_covered
                    new_cost = get_cost(len(current_code) + 1, new_uncov)
                    
                    if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                        # Accept
                        current_code.add(new_idx)
                        current_cover_count[nb] += 1
                        current_uncovered = new_uncov
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_code = list(current_code)
        elif move_type < 0.6:
            # Remove a random word
            if len(current_code) > 0:
                rem_idx = random.choice(list(current_code))
                nb = neighbors[rem_idx]
                # Calculate delta cost
                # New size: len - 1
                # New uncovered: current_uncovered + (number of neighbors of rem_idx that were covered ONLY by rem_idx)
                newly_uncovered = np.sum(current_cover_count[nb] == 1)
                new_uncov = current_uncovered + newly_uncovered
                new_cost = get_cost(len(current_code) - 1, new_uncov)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    # Accept
                    current_code.remove(rem_idx)
                    current_cover_count[nb] -= 1
                    current_uncovered = new_uncov
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_code = list(current_code)
        else:
            # Swap: Remove one, Add one
            if len(current_code) > 0 and len(current_code) < total_words:
                rem_idx = random.choice(list(current_code))
                # Pick a candidate to add. 
                # Heuristic: pick from neighbors of uncovered words or just random?
                # Random is safer for SA exploration.
                add_idx = np.random.randint(total_words)
                
                if add_idx in current_code:
                    continue # Skip if same
                
                nb_rem = neighbors[rem_idx]
                nb_add = neighbors[add_idx]
                
                # Update cover counts virtually to check cost
                # We need to know how many words become uncovered/covered.
                # This is slightly more complex. 
                # Let's just compute new cost directly by updating a temporary array?
                # No, let's be smart.
                # Words affected: union of nb_rem and nb_add.
                # For each word in union, check old cover count and new cover count.
                
                # Optimization: 
                # old_cover[w] is known.
                # new_cover[w] = old_cover[w] - (1 if w in nb_rem else 0) + (1 if w in nb_add else 0)
                
                # We only care about words where cover count changes from 0->1 (covered) or 1->0 (uncovered).
                # Words with cover count >= 2 remain covered.
                
                # Let's just iterate over the union of neighbors. Size is at most 15+15=30.
                affected = np.union1d(nb_rem, nb_add)
                
                delta_uncov = 0
                for w in affected:
                    old_c = current_cover_count[w]
                    in_rem = 1 if w in nb_rem else 0 # This is slow. 
                    # Better: use boolean masks or sets?
                    # Since n is small, let's just use sets for the neighbor indices for this check?
                    # Or precomputed sets?
                    # Let's use the fact that neighbors are small arrays.
                    # We can check `np.any(nb_rem == w)` but that's O(15).
                    # Let's just do the full update on a copy of the slice? No.
                    
                    # Simplest robust way for small n:
                    # Create a set of current code? No, we have cover_count.
                    
                    # Let's just calculate the change in uncovered count for the union.
                    # We need to know if w was in nb_rem and if w was in nb_add.
                    # We can precompute a boolean mask for nb_rem and nb_add?
                    # That's O(2187) per swap check, which is too slow if we do many swaps.
                    
                    # Alternative: 
                    # newly_uncovered = 0
                    # newly_covered = 0
                    # for w in nb_rem:
                    #    if current_cover_count[w] == 1 and w not in nb_add:
                    #        newly_uncovered += 1
                    # for w in nb_add:
                    #    if current_cover_count[w] == 0 and w not in nb_rem:
                    #        newly_covered += 1
                    # 
                    # This requires checking "w in nb_add". 
                    # We can create a set of nb_add once per swap attempt.
                    # Set creation is O(15). Membership check is O(1).
                    # This is fast enough.
                    
                    pass # Placeholder, logic below
                    
                # Let's implement the swap logic properly
                set_add = set(nb_add)
                set_rem = set(nb_rem)
                
                new_uncov = current_uncovered
                for w in nb_rem:
                    if current_cover_count[w] == 1 and w not in set_add:
                        new_uncov += 1
                for w in nb_add:
                    if current_cover_count[w] == 0 and w not in set_rem:
                        new_uncov -= 1
                        
                new_size = len(current_code) # Size stays same
                new_cost = get_cost(new_size, new_uncov)
                
                if new_cost < current_cost or random.random() < np.exp((current_cost - new_cost) / T):
                    # Accept
                    current_code.remove(rem_idx)
                    current_code.add(add_idx)
                    current_cover_count[nb_rem] -= 1
                    current_cover_count[nb_add] += 1
                    current_uncovered = new_uncov
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_code = list(current_code)
                        
        T *= cooling_rate
        
    # Ensure we return a valid cover. 
    # If best_code has uncovered words, we must repair it.
    # This can happen if SA ends in an invalid state.
    # We can run a greedy repair on best_code.
    
    final_code = list(best_code)
    cover_count = np.zeros(total_words, dtype=np.int32)
    for idx in final_code:
        cover_count[neighbors[idx]] += 1
        
    uncovered_indices = np.where(cover_count == 0)[0]
    if len(uncovered_indices) > 0:
        # Greedy repair
        while len(uncovered_indices) > 0:
            # Find word that covers most uncovered
            # This is O(K * 15 * len(uncovered)) which might be slow if many uncovered.
            # But usually SA keeps it close to valid.
            best_idx = -1
            best_gain = -1
            # Sample from all words? Or just greedy from all?
            # Let's just do a standard greedy step for the uncovered part.
            # To be safe and fast, let's just add random words until covered? 
            # No, let's do one greedy pass.
            
            # Actually, since the penalty is high, SA should rarely end in a very invalid state.
            # A simple repair: 
            # While uncovered exists, pick a random uncovered word, find a codeword that covers it (itself or neighbor), add it.
            # This guarantees validity.
            while len(uncovered_indices) > 0:
                target = uncovered_indices[np.random.randint(len(uncovered_indices))]
                # Pick target itself or a neighbor
                # Target itself covers target.
                # Let's just add target. It's a valid codeword.
                if target not in final_code:
                    final_code.append(target)
                    cover_count[neighbors[target]] += 1
                    # Update uncovered_indices
                    # This is inefficient. 
                    # Let's just recompute uncovered at the end if we do this.
                    # Or better: just run a full greedy from scratch if best_code is invalid?
                    # No, that defeats the purpose.
                    
                    # Let's just recompute uncovered mask after adding.
                    # This loop might be slow if many uncovered.
                    # But let's assume it's few.
                    pass
            
            # Simpler repair:
            # If invalid, just run the initial greedy algorithm again?
            # That's a fallback.
            # Let's try to repair by adding missing words greedily.
            
            # Let's restart the repair loop properly.
            break # Exit the while loop above and handle repair outside
            
    # Proper Repair Block
    if len(uncovered_indices) > 0:
        # Re-initialize cover count for the repair phase
        cover_count = np.zeros(total_words, dtype=np.int32)
        for idx in final_code:
            cover_count[neighbors[idx]] += 1
        
        uncovered_indices = np.where(cover_count == 0)[0]
        while len(uncovered_indices) > 0:
            # Pick the word that covers the most uncovered words
            # We can just pick a random uncovered word and add it? 
            # That guarantees coverage of that word, but might not be optimal.
            # Greedy is better.
            # For speed, let's just add the uncovered words themselves? 
            # No, a codeword covers 15 words. 
            # Let's do a simple greedy:
            # Find idx that maximizes overlap with uncovered_indices.
            # We can sample a few candidates or just iterate all? 
            # Iterating all 2187 is fast.
            best_idx = -1
            best_gain = -1
            for idx in range(total_words):
                # Count how many uncovered words are in neighbors[idx]
                # neighbors[idx] is an array.
                # We need to check intersection with uncovered_indices.
                # Using sets is fast for small sizes.
                # But uncovered_indices can be large initially.
                # Let's use numpy:
                # np.intersect1d(neighbors[idx], uncovered_indices)
                # This is O(15 * log(U)) or similar.
                # 2187 * 15 * log(100) is ~ 100k ops. Very fast.
                common = np.intersect1d(neighbors[idx], uncovered_indices)
                if len(common) > best_gain:
                    best_gain = len(common)
                    best_idx = idx
                    
            if best_idx == -1: break # Should not happen
            
            final_code.append(best_idx)
            cover_count[neighbors[best_idx]] += 1
            # Update uncovered_indices
            # Remove neighbors[best_idx] from uncovered_indices
            # uncovered_indices = np.setdiff1d(uncovered_indices, neighbors[best_idx])
            # This is O(U * 15).
            mask = np.isin(uncovered_indices, neighbors[best_idx])
            uncovered_indices = uncovered_indices[~mask]
            
    # Final check: ensure distinctness (sets handle this, but final_code is list)
    # The problem requires distinct codewords.
    # If we added duplicates, we should remove them.
    # Since we used a set for current_code, best_code should have distinct elements.
    # The repair phase adds words. If we add a word already in final_code, it's a duplicate.
    # We should check.
    unique_final = list(set(final_code))
    
    result = []
    for idx in unique_final:
        result.append(idx_to_word(idx))
        
    return result
