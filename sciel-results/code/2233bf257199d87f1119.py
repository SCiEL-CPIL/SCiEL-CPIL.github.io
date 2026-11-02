import itertools
import numpy as np
import random
import time

def construct(q, n, radius):
    if q == 3 and n == 7 and radius == 1:
        return _construct_3_7_1_ils()
    return _greedy_cover(q, n, radius)

def _construct_3_7_1_ils():
    start_time = time.time()
    time_limit = 110.0
    
    # Precompute all words and neighbors
    all_words = list(itertools.product(range(3), repeat=7))
    N = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # neighbors[i] is a list of indices of words at Hamming distance <= 1 from all_words[i]
    # Note: In the previous code, neighbors included the word itself. 
    # For covering radius 1, a codeword covers itself and its neighbors.
    neighbors = []
    for w in all_words:
        nb = [word_to_idx[w]]
        for i in range(7):
            for d in [1, 2]:
                new_w = list(w)
                new_w[i] = (w[i] + d) % 3
                nb.append(word_to_idx[tuple(new_w)])
        neighbors.append(nb)
        
    # covers[i] is a set of indices covered by codeword i
    covers = [set(nb) for nb in neighbors]

    best_code = None
    best_size = float('inf')
    
    # Strategy: Iterated Local Search
    # 1. Start with a random subset of size K
    # 2. Repair: Add codewords to cover all words
    # 3. Reduce: Remove codewords that are redundant
    # 4. Kick: Perturb the code (remove a few, add a few random)
    
    # We know the answer is around 186. 
    # Greedy usually gives ~200-210.
    # We can start with random sizes around 180-190.
    
    for trial in range(200): # Max trials, limited by time
        if time.time() - start_time > time_limit:
            break
            
        # 1. Initialize
        # Try different initial sizes
        init_size = random.randint(170, 190)
        code_set = set(random.sample(range(N), init_size))
        code_list = list(code_set)
        
        # 2. Repair: Ensure validity
        # Compute coverage
        cover_count = np.zeros(N, dtype=np.int32)
        for c in code_list:
            for nb in neighbors[c]:
                cover_count[nb] += 1
                
        uncovered = np.where(cover_count == 0)[0]
        
        # Greedy repair
        while len(uncovered) > 0:
            # Find codeword that covers most uncovered
            # To speed up, we can just pick a random uncovered word and find a random codeword covering it?
            # No, greedy is better for repair.
            best_c = -1
            best_count = -1
            # Optimization: Only check codewords that are NOT in code_set? 
            # Actually, we can check all. N=2187 is small.
            # But we can be smarter: Pick a random uncovered word u.
            # Any codeword in neighbors[u] covers u.
            # Among those, pick one that covers the most other uncovered words.
            
            if len(uncovered) == 0:
                break
                
            # Pick a random uncovered word to focus on
            u = random.choice(uncovered)
            candidates = [c for c in neighbors[u] if c not in code_set]
            
            if not candidates:
                # This shouldn't happen if we allow adding existing? No, we add new.
                # If all neighbors are in code_set, then u is covered? 
                # Wait, if u is uncovered, no codeword in code_set covers it.
                # So candidates must be non-empty because neighbors[u] contains words not in code_set?
                # Unless code_set contains ALL neighbors of u, but then u would be covered.
                # So candidates is always non-empty if u is uncovered.
                
                pass
                
            # Evaluate candidates
            for c in candidates:
                # Count how many uncovered words c covers
                count = 0
                for nb in covers[c]:
                    if cover_count[nb] == 0:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_c = c
                    
            if best_c == -1:
                # Should not happen
                break
                
            # Add best_c
            code_set.add(best_c)
            code_list.append(best_c)
            for nb in neighbors[best_c]:
                cover_count[nb] += 1
                
            # Update uncovered
            uncovered = np.where(cover_count == 0)[0]
            
        # 3. Reduce: Remove redundant codewords
        # We can try to remove any codeword that doesn't uncover any word.
        # Random order to avoid bias.
        random.shuffle(code_list)
        i = 0
        while i < len(code_list):
            c = code_list[i]
            # Check if c is essential
            # Remove c temporarily
            for nb in neighbors[c]:
                cover_count[nb] -= 1
                
            if np.min(cover_count) >= 1:
                # Valid removal
                code_set.remove(c)
                code_list.pop(i)
            else:
                # Revert
                for nb in neighbors[c]:
                    cover_count[nb] += 1
                i += 1
                
        # 4. Check Best
        current_size = len(code_list)
        if current_size < best_size:
            best_size = current_size
            best_code = list(code_list)
            
        # 5. Kick (Perturb) for next iteration
        # If we found a good solution, we can start next iteration from it with a small perturbation.
        # Or just restart random.
        # Let's do a mix: 50% restart random, 50% perturb best.
        if trial > 0 and best_code is not None and random.random() < 0.5:
            # Perturb best_code
            code_list = list(best_code)
            code_set = set(code_list)
            
            # Remove k random codewords
            k = random.randint(1, 3)
            if len(code_list) > k:
                to_remove = random.sample(code_list, k)
                for c in to_remove:
                    code_set.remove(c)
                    code_list.remove(c)
                    
            # Add k random codewords
            to_add = random.sample(range(N), k)
            for c in to_add:
                if c not in code_set:
                    code_set.add(c)
                    code_list.append(c)
                    
            # Recalculate cover_count for the perturbed code
            cover_count = np.zeros(N, dtype=np.int32)
            for c in code_list:
                for nb in neighbors[c]:
                    cover_count[nb] += 1
            uncovered = np.where(cover_count == 0)[0]
        else:
            # Continue loop, next iteration will init random
            continue
    if best_code is None:
        # Fallback to greedy if something went wrong
        return _greedy_cover(3, 7, 1)
        
    return [list(all_words[idx]) for idx in best_code]

def _greedy_cover(q, n, radius):
    # Fallback for other parameters (not used in this specific task instance)
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    N = len(all_words)
    
    covers = []
    for w in all_words:
        covered = set()
        covered.add(word_to_idx[w])
        for r in range(1, radius + 1):
            for positions in itertools.combinations(range(n), r):
                for deltas in itertools.product(range(1, q), repeat=r):
                    new_w = list(w)
                    for pos, d in zip(positions, deltas):
                        new_w[pos] = (w[pos] + d) % q
                    covered.add(word_to_idx[tuple(new_w)])
        covers.append(covered)
    
    uncovered = set(range(N))
    code = []
    while uncovered:
        best = None
        best_count = -1
        for i in range(N):
            count = len(covers[i] & uncovered)
            if count > best_count:
                best_count = count
                best = i
        code.append(list(all_words[best]))
        uncovered -= covers[best]
    
    return code
