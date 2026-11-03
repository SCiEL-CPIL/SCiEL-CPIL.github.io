import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We target 145s to be safe.
    deadline = start_time + 145.0

    total_space = q ** n
    
    # Precompute neighbor masks for each word
    # mask[w] is a binary vector of length total_space where mask[w][v] = 1 if dist(w, v) <= radius
    # For q=3, n=7, radius=1: each word covers 1 + n*(q-1) = 15 words.
    
    # To speed up SA/ILS, we represent the state as:
    # 1. selected: a boolean array of length total_space (True if word is in code)
    # 2. counts: an integer array of length total_space (coverage count for each word)
    
    # Precompute the binary masks for all words
    # masks is a (total_space, total_space) boolean array? 
    # 2187 * 2187 * 1 byte ~ 4.7 MB. This is feasible and allows vectorized operations.
    # However, iterating over 2187 candidates for each move might be slow if we do full vector ops.
    # Better: Precompute the list of neighbors for each word.
    
    # Let's use a hybrid: 
    # masks_sparse: list of lists of indices covered by each word.
    # For update: counts[masks_sparse[w]] += 1. This is fast in NumPy.
    
    neighbors = [[] for _ in range(total_space)]
    for i in range(total_space):
        neighbors[i].append(i)
        for pos in range(n):
            digit_val = (i // (q ** pos)) % q
            for new_val in range(q):
                if new_val == digit_val:
                    continue
                diff = new_val - digit_val
                neighbors[i].append(i + diff * (q ** pos))
    
    # Convert to numpy arrays for faster indexing if needed, but lists of ints are fine for indexing counts
    # Actually, for counts[masks] += 1, a list of ints works with NumPy fancy indexing.
    
    def int_to_tuple(val):
        res = [0] * n
        for i in range(n):
            res[i] = (val // (q ** i)) % q
        return res

    def get_energy(selected_count, uncovered_count, lambda_val):
        return selected_count + lambda_val * uncovered_count

    # Phase 1: Greedy Construction
    # We want a valid starting point.
    counts = np.zeros(total_space, dtype=np.int32)
    selected_set = set()
    selected_list = []
    
    remaining = np.ones(total_space, dtype=bool)
    
    while np.any(remaining):
        if time.time() > deadline - 10:
            break
        
        best_gain = -1
        best_word = -1
        
        # Sample a subset of candidates to speed up greedy? 
        # For 2187 items, full scan is ~2187 * 15 ops = 30k ops. 
        # Doing this ~100-200 times is fine.
        # Let's do full scan for robustness.
        
        # Optimization: Only consider words that cover at least one remaining word.
        # Since remaining is small initially? No, initially all are remaining.
        
        for w in range(total_space):
            if w in selected_set:
                continue
            gain = np.sum(remaining[neighbors[w]])
            if gain > best_gain:
                best_gain = gain
                best_word = w
        
        if best_word == -1:
            break
        
        selected_set.add(best_word)
        selected_list.append(best_word)
        counts[neighbors[best_word]] += 1
        remaining[neighbors[best_word]] = False
        
        # Early stop if we have a good enough start? 
        # Greedy usually gives ~200-250 words. We want to go lower.
        # We can stop greedy once we have a valid solution, then ILS will optimize.
        
    # Ensure we have a valid solution. If time ran out during greedy, we might have invalid.
    # If invalid, we need to add words to cover gaps.
    uncovered_indices = np.where(counts == 0)[0]
    while len(uncovered_indices) > 0 and time.time() < deadline - 5:
        # Pick a random uncovered word and add a random neighbor to cover it
        # Or greedy add: pick word that covers most uncovered
        if len(uncovered_indices) > 100:
             # Greedy add
             best_gain = -1
             best_word = -1
             for w in range(total_space):
                 if w in selected_set: continue
                 gain = np.sum((counts[neighbors[w]] == 0))
                 if gain > best_gain:
                     best_gain = gain
                     best_word = w
             if best_word == -1: break
             selected_set.add(best_word)
             selected_list.append(best_word)
             counts[neighbors[best_word]] += 1
        else:
             # Random fix
             u = random.choice(uncovered_indices.tolist())
             # Find a word that covers u
             # We can just pick a random word from neighbors of u? No, we need to add a word W such that u in neighbors[W].
             # Since neighbors are symmetric (distance is symmetric), u in neighbors[W] <=> W in neighbors[u].
             candidates = neighbors[u]
             # Pick one not in selected_set? Or any?
             # If we pick one in selected_set, it's already covered? No, counts[u]==0 means no selected word covers u.
             # So no word in selected_set covers u.
             # So any w in candidates is not in selected_set? 
             # Wait, if w in selected_set, then counts[u] would be > 0.
             # So all w in candidates are valid additions.
             w = random.choice(candidates)
             selected_set.add(w)
             selected_list.append(w)
             counts[neighbors[w]] += 1
        
        uncovered_indices = np.where(counts == 0)[0]
    
    if len(uncovered_indices) > 0:
        # Fallback: return whatever we have? No, must be valid.
        # Force add until valid
        while len(uncovered_indices) > 0:
             u = random.choice(uncovered_indices.tolist())
             w = random.choice(neighbors[u])
             if w not in selected_set:
                 selected_set.add(w)
                 selected_list.append(w)
                 counts[neighbors[w]] += 1
             uncovered_indices = np.where(counts == 0)[0]
    
    best_solution = selected_list[:]
    best_size = len(best_solution)
    best_counts = counts.copy()
    best_selected_set = set(best_solution)
    
    # Phase 2: Iterated Local Search with Penalty
    # Energy = |S| + Lambda * Uncovered
    # We start with a valid solution (Uncovered=0).
    # We allow moves that increase Uncovered if they reduce |S| enough (via Lambda).
    
    current_solution = best_solution[:]
    current_counts = best_counts.copy()
    current_set = set(current_solution)
    current_size = len(current_solution)
    current_uncovered = np.sum(current_counts == 0)
    
    # Lambda tuning
    # If Lambda is high, we prefer valid solutions.
    # If Lambda is low, we might end up with many uncovered words.
    # We want to find a valid solution with min size.
    # Strategy: Start with high Lambda, anneal it down? 
    # Or just use a fixed Lambda that is slightly > 1? 
    # If we remove a word (cost -1) and uncover 2 words (cost +2*Lambda), delta E = -1 + 2*Lambda.
    # If Lambda = 0.5, delta E = 0. Accept.
    # If Lambda = 1.0, delta E = 1. Reject (unless random).
    # Let's use Simulated Annealing on top of ILS structure? 
    # The prompt suggests "custom energy function... allowing algorithm to accept temporary increases".
    # This implies SA-like acceptance or just accepting if energy drops.
    # Let's use SA acceptance: P(accept) = exp(-delta_E / T).
    
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    lambda_val = 2.0 # Heavy penalty for uncovered words initially
    
    all_words = list(range(total_space))
    
    while time.time() < deadline - 2:
        
        # --- Local Search Phase (Hill Climbing with SA Acceptance) ---
        steps_in_ls = 0
        max_ls_steps = 500
        
        while steps_in_ls < max_ls_steps and time.time() < deadline - 2:
            steps_in_ls += 1
            
            # Choose move type
            r = random.random()
            
            if r < 0.4 and current_size > 1:
                # Remove
                idx = random.randint(0, current_size - 1)
                w_out = current_solution[idx]
                
                # Check impact
                # New counts = current_counts - 1 for neighbors[w_out]
                # Uncovered change: 
                # Words that were count=1 and in neighbors[w_out] become 0.
                # Words that were count=0 and in neighbors[w_out] stay 0 (but were already uncovered).
                # Actually, we just need delta_uncovered.
                # delta_uncovered = sum(1 for v in neighbors[w_out] if current_counts[v] == 1)
                # Because those become 0.
                # If current_counts[v] > 1, it becomes >=1, still covered.
                # If current_counts[v] == 0, it stays 0 (already uncovered).
                
                mask = neighbors[w_out]
                delta_uncovered = np.sum(current_counts[mask] == 1)
                
                delta_size = -1
                delta_E = delta_size + lambda_val * delta_uncovered
                
                # Acceptance
                accept = False
                if delta_E <= 0:
                    accept = True
                else:
                    if T > T_min:
                        if random.random() < np.exp(-delta_E / T):
                            accept = True
                    else:
                        accept = False
                
                if accept:
                    # Execute
                    current_counts[mask] -= 1
                    current_solution.pop(idx)
                    current_set.remove(w_out)
                    current_size -= 1
                    current_uncovered += delta_uncovered
                    
                    if current_uncovered == 0 and current_size < best_size:
                        best_size = current_size
                        best_solution = current_solution[:]
                        best_counts = current_counts.copy()
                        best_selected_set = set(current_solution)
                        
            elif r < 0.7:
                # Add
                w_in = random.choice(all_words)
                if w_in in current_set:
                    continue
                
                mask = neighbors[w_in]
                # Words that were count=0 and in mask become 1.
                # delta_uncovered = - sum(1 for v in mask if current_counts[v] == 0)
                delta_uncovered = -np.sum(current_counts[mask] == 0)
                
                delta_size = 1
                delta_E = delta_size + lambda_val * delta_uncovered
                
                accept = False
                if delta_E <= 0:
                    accept = True
                else:
                    if T > T_min:
                        if random.random() < np.exp(-delta_E / T):
                            accept = True
                    else:
                        accept = False
                
                if accept:
                    # Execute
                    current_counts[mask] += 1
                    current_solution.append(w_in)
                    current_set.add(w_in)
                    current_size += 1
                    current_uncovered += delta_uncovered
                    
                    if current_uncovered == 0 and current_size < best_size:
                        best_size = current_size
                        best_solution = current_solution[:]
                        best_counts = current_counts.copy()
                        best_selected_set = set(current_solution)
                        
            else:
                # Swap
                if current_size == 0:
                    continue
                idx = random.randint(0, current_size - 1)
                w_out = current_solution[idx]
                w_in = random.choice(all_words)
                
                if w_in == w_out or w_in in current_set:
                    continue
                
                mask_out = neighbors[w_out]
                mask_in = neighbors[w_in]
                
                # Calculate delta_uncovered
                # We need to know which words change status from covered to uncovered or vice versa.
                # This is tricky to do in O(1) without iterating.
                # Approximation: Iterate over the union of masks? 
                # Or just calculate new counts for the affected set.
                # Affected set = mask_out U mask_in.
                # Since n=7, q=3, masks are small (size 15).
                # Union size <= 30.
                
                affected = set(mask_out) | set(mask_in)
                
                # Simulate
                # We need to check if any word in affected goes from 1 to 0 (uncovered increases)
                # or from 0 to 1 (uncovered decreases).
                
                delta_uncov = 0
                for v in affected:
                    c = current_counts[v]
                    # c_new = c - (1 if v in mask_out else 0) + (1 if v in mask_in else 0)
                    c_new = c
                    if v in mask_out: c_new -= 1
                    if v in mask_in: c_new += 1
                    
                    if c == 0 and c_new > 0:
                        delta_uncov -= 1
                    elif c > 0 and c_new == 0:
                        delta_uncov += 1
                    # else: status unchanged (0->0 or 1->1 or 2->2 etc)
                
                delta_size = 0
                delta_E = delta_size + lambda_val * delta_uncov
                
                accept = False
                if delta_E <= 0:
                    accept = True
                else:
                    if T > T_min:
                        if random.random() < np.exp(-delta_E / T):
                            accept = True
                    else:
                        accept = False
                
                if accept:
                    # Execute
                    current_counts[mask_out] -= 1
                    current_counts[mask_in] += 1
                    current_solution[idx] = w_in
                    current_set.remove(w_out)
                    current_set.add(w_in)
                    # Size unchanged
                    current_uncovered += delta_uncov
                    
                    if current_uncovered == 0 and current_size < best_size:
                        best_size = current_size
                        best_solution = current_solution[:]
                        best_counts = current_counts.copy()
                        best_selected_set = set(current_solution)
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # --- Perturbation Phase ---
        # If we are stuck or just to explore new basins
        # Perturb the current solution (or best solution?)
        # Let's perturb the current solution.
        # If current solution is invalid (uncovered > 0), we might want to move towards best valid?
        # Or just perturb current.
        
        # Number of perturbations
        num_pert = random.randint(3, 10)
        
        for _ in range(num_pert):
            if time.time() > deadline - 2:
                break
            
            r = random.random()
            if r < 0.3 and current_size > 1:
                # Remove random
                idx = random.randint(0, current_size - 1)
                w_out = current_solution[idx]
                mask = neighbors[w_out]
                delta_uncov = np.sum(current_counts[mask] == 1)
                
                current_counts[mask] -= 1
                current_solution.pop(idx)
                current_set.remove(w_out)
                current_size -= 1
                current_uncovered += delta_uncov
                
            elif r < 0.7:
                # Add random
                w_in = random.choice(all_words)
                if w_in not in current_set:
                    mask = neighbors[w_in]
                    delta_uncov = -np.sum(current_counts[mask] == 0)
                    
                    current_counts[mask] += 1
                    current_solution.append(w_in)
                    current_set.add(w_in)
                    current_size += 1
                    current_uncovered += delta_uncov
                    
            else:
                # Swap random
                if current_size > 0:
                    idx = random.randint(0, current_size - 1)
                    w_out = current_solution[idx]
                    w_in = random.choice(all_words)
                    if w_in not in current_set and w_in != w_out:
                        mask_out = neighbors[w_out]
                        mask_in = neighbors[w_in]
                        
                        affected = set(mask_out) | set(mask_in)
                        delta_uncov = 0
                        for v in affected:
                            c = current_counts[v]
                            c_new = c
                            if v in mask_out: c_new -= 1
                            if v in mask_in: c_new += 1
                            
                            if c == 0 and c_new > 0: delta_uncov -= 1
                            elif c > 0 and c_new == 0: delta_uncov += 1
                        
                        current_counts[mask_out] -= 1
                        current_counts[mask_in] += 1
                        current_solution[idx] = w_in
                        current_set.remove(w_out)
                        current_set.add(w_in)
                        current_uncovered += delta_uncov
        
        # Restart Temperature
        T = max(T_min, T * 0.8)
        
        # If current solution is invalid and we've been invalid for a while, 
        # maybe snap back to best valid?
        # For now, let the search handle it. The penalty lambda pushes it back.
        
        # Check time
        if time.time() > deadline - 1:
            break

    # Final Polish: Ensure we return the best VALID solution found.
    # best_solution is always valid because we only update it when current_uncovered == 0.
    
    # One last quick polish on best_solution to remove any redundant words?
    # The ILS might have left redundant words if lambda was low.
    # Let's do a simple removal pass on best_solution.
    
    temp_counts = np.zeros(total_space, dtype=np.int32)
    for w in best_solution:
        temp_counts[neighbors[w]] += 1
        
    changed = True
    while changed:
        changed = False
        for i in range(len(best_solution)):
            w = best_solution[i]
            # Check if removable
            mask = neighbors[w]
            if np.all(temp_counts[mask] > 1):
                temp_counts[mask] -= 1
                best_solution.pop(i)
                changed = True
                break # Restart loop since indices changed
    
    best_words = [int_to_tuple(v) for v in best_solution]
    return best_words
