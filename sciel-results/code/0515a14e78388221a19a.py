import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    if q != 3 or n != 6 or radius != 1:
        # Fallback: return all words (valid but large)
        all_words = []
        for i in range(q ** n):
            word = [(i // (q ** p)) % q for p in range(n)]
            all_words.append(word)
        return all_words

    total = 3 ** 6
    powers = [3 ** p for p in range(6)]
    
    # Precompute neighbors as numpy array
    neighbors = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        d = [(i // powers[p]) % 3 for p in range(6)]
        neighbors[i, 0] = i
        k = 1
        for p in range(6):
            for alt in range(3):
                if alt != d[p]:
                    new_i = i + (alt - d[p]) * powers[p]
                    neighbors[i, k] = new_i
                    k += 1
    
    # Precompute neighbor sets for fast subset checks (used in swap validation)
    # Actually, for SA we might not need full sets if we use vectorized checks,
    # but for swap validity (covering all critical words), sets are useful.
    # However, creating 729 sets is cheap.
    neighbor_sets = [set(neighbors[i].tolist()) for i in range(total)]
    
    def idx_to_list(idx):
        return [(idx // powers[p]) % 3 for p in range(6)]

    def greedy_cover(seed_val):
        random.seed(seed_val)
        uncovered = np.ones(total, dtype=np.int8)
        codewords = []
        
        while uncovered.sum() > 0:
            if time.time() > deadline - 30:
                break
            counts = np.sum(uncovered[neighbors], axis=1)
            # Randomize tie-breaking to get diverse solutions
            max_count = counts.max()
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_c = random.choice(candidates.tolist())
            codewords.append(best_c)
            uncovered[neighbors[best_c]] = 0
            
        # Ensure full coverage if we broke early due to time (shouldn't happen usually)
        if uncovered.sum() != 0:
            while uncovered.sum() > 0:
                c = int(np.random.randint(total))
                codewords.append(c)
                uncovered[neighbors[c]] = 0
                
        return codewords

    # 1. Initial Solution: Try multiple greedy runs
    best_initial = None
    best_initial_size = float('inf')
    
    # Run 5 greedy passes with different seeds
    for i in range(5):
        if time.time() > deadline - 100:
            break
        cw = greedy_cover(42 + i)
        if len(cw) < best_initial_size:
            best_initial_size = len(cw)
            best_initial = cw
            
    codewords = best_initial
    if codewords is None:
        codewords = greedy_cover(42)

    # Initialize coverage counts
    coverage_counts = np.zeros(total, dtype=np.int32)
    for c in codewords:
        coverage_counts[neighbors[c]] += 1
        
    in_set = set(codewords)
    
    best_codewords = list(codewords)
    best_size = len(codewords)
    
    # 2. Simulated Annealing
    T = 5.0
    T_min = 0.01
    cooling = 0.999
    steps = 0
    last_improve = time.time()
    
    while time.time() < deadline - 2:
        steps += 1
        
        # Check for stagnation and restart/cooling adjustment? 
        # For now, just let it cool.
        
        # Choose a move type: 
        # 1. Remove (if possible)
        # 2. Swap
        # 3. Add (Expansion)
        
        # Identify removable words
        removable_indices = []
        for idx, c in enumerate(codewords):
            if coverage_counts[neighbors[c]].min() >= 2:
                removable_indices.append(idx)
        
        move_type = random.random()
        
        if move_type < 0.3 and removable_indices:
            # Move: Remove
            idx_to_remove = random.choice(removable_indices)
            c_to_remove = codewords[idx_to_remove]
            
            new_codewords = codewords[:]
            new_codewords.pop(idx_to_remove)
            new_size = len(new_codewords)
            
            # Always accept removal
            codewords = new_codewords
            coverage_counts[neighbors[c_to_remove]] -= 1
            in_set.discard(c_to_remove)
            
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = list(codewords)
                last_improve = time.time()
                
        elif move_type < 0.8:
            # Move: Swap
            if len(codewords) == 0:
                continue
            idx_to_remove = random.randint(0, len(codewords) - 1)
            c_to_remove = codewords[idx_to_remove]
            
            # Find critical words: neighbors of c_to_remove with coverage exactly 1
            crit_mask = (coverage_counts[neighbors[c_to_remove]] == 1)
            critical_words = neighbors[c_to_remove][crit_mask]
            
            if len(critical_words) == 0:
                # This is actually a removal, but we are in swap branch. 
                # Treat as removal if valid, else skip.
                # If coverage_counts[neighbors[c_to_remove]].min() >= 2, it's removable.
                if coverage_counts[neighbors[c_to_remove]].min() >= 2:
                    # Do removal
                    c_to_remove = codewords[idx_to_remove]
                    new_codewords = codewords[:]
                    new_codewords.pop(idx_to_remove)
                    codewords = new_codewords
                    coverage_counts[neighbors[c_to_remove]] -= 1
                    in_set.discard(c_to_remove)
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        best_codewords = list(codewords)
                        last_improve = time.time()
                    T = max(T * cooling, T_min)
                    continue
                else:
                    # Not removable, so we MUST swap. But if no critical words, it IS removable.
                    # So this case (crit_mask empty but min < 2) is impossible.
                    # Wait, if min < 2, then some neighbor has coverage 1. So crit_mask is not empty.
                    # So if crit_mask is empty, min >= 2.
                    pass
            else:
                # Find candidates that cover all critical words
                w0 = int(critical_words[0])
                candidates = neighbors[w0]
                
                valid_candidates = []
                for cand in candidates:
                    if cand == c_to_remove:
                        continue
                    # Check if cand covers all critical words
                    if all(w in neighbor_sets[int(cand)] for w in critical_words):
                        valid_candidates.append(int(cand))
                
                if valid_candidates:
                    c_to_add = random.choice(valid_candidates)
                    
                    # Perform swap
                    new_codewords = codewords[:]
                    new_codewords[idx_to_remove] = c_to_add
                    new_size = len(new_codewords)
                    
                    # Swap keeps size same. Always accept? 
                    # Or accept based on some heuristic? 
                    # Since size is same, we accept to explore.
                    codewords = new_codewords
                    coverage_counts[neighbors[c_to_remove]] -= 1
                    coverage_counts[neighbors[c_to_add]] += 1
                    in_set.discard(c_to_remove)
                    in_set.add(c_to_add)
                    
                    # Check if this swap allows a subsequent removal? 
                    # We don't look ahead in SA, just accept.
                    
        else:
            # Move: Add (Expansion)
            # Pick a random word to add
            c_to_add = int(np.random.randint(total))
            if c_to_add in in_set:
                continue
                
            new_codewords = codewords[:]
            new_codewords.append(c_to_add)
            new_size = len(new_codewords)
            
            # Accept with probability exp(-1/T)
            if random.random() < np.exp(-1.0 / T):
                codewords = new_codewords
                coverage_counts[neighbors[c_to_add]] += 1
                in_set.add(c_to_add)
                
        T = max(T * cooling, T_min)
        
        # Occasional final cleanup pass to remove redundancies introduced by expansions
        if steps % 100 == 0:
            improved = True
            while improved:
                improved = False
                for idx in range(len(codewords)):
                    c = codewords[idx]
                    if coverage_counts[neighbors[c]].min() >= 2:
                        codewords.pop(idx)
                        coverage_counts[neighbors[c]] -= 1
                        in_set.discard(c)
                        improved = True
                        if len(codewords) < best_size:
                            best_size = len(codewords)
                            best_codewords = list(codewords)
                        break

    # Final cleanup
    improved = True
    while improved:
        improved = False
        for idx in range(len(codewords)):
            c = codewords[idx]
            if coverage_counts[neighbors[c]].min() >= 2:
                codewords.pop(idx)
                coverage_counts[neighbors[c]] -= 1
                improved = True
                if len(codewords) < best_size:
                    best_size = len(codewords)
                    best_codewords = list(codewords)
                break

    if len(best_codewords) < len(codewords):
        final_indices = best_codewords
    else:
        final_indices = codewords

    return [idx_to_list(idx) for idx in final_indices]

# EVOLVE-BLOCK-END
