import time
import random
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback: simple greedy set cover
        total_words = q ** n
        
        def word_to_idx(w):
            idx = 0
            for val in w:
                idx = idx * q + val
            return idx
            
        def idx_to_word(idx):
            w = [0] * n
            temp = idx
            for i in range(n - 1, -1, -1):
                w[i] = temp % q
                temp //= q
            return tuple(w)

        # Precompute neighbors
        neighbors = [None] * total_words
        for idx in range(total_words):
            w = idx_to_word(idx)
            nbrs = [idx]
            for i in range(n):
                for val in range(q):
                    if val != w[i]:
                        nw = list(w)
                        nw[i] = val
                        nbrs.append(word_to_idx(tuple(nw)))
            neighbors[idx] = nbrs

        uncovered = np.ones(total_words, dtype=bool)
        code = []
        
        while any(uncovered):
            best_idx = -1
            best_count = -1
            for idx in range(total_words):
                count = sum(1 for nbr in neighbors[idx] if uncovered[nbr])
                if count > best_count:
                    best_count = count
                    best_idx = idx
            if best_idx == -1:
                break
            code.append(best_idx)
            for nbr in neighbors[best_idx]:
                uncovered[nbr] = False
                
        return [list(idx_to_word(idx)) for idx in code]

    # Specific optimization for q=3, n=8, radius=1
    total_words = 3 ** 8  # 6561
    
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * 3 + val
        return idx
        
    def idx_to_word(idx):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        return tuple(w)

    # Precompute neighbors
    neighbors = [None] * total_words
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbrs = [idx]
        for i in range(8):
            for val in range(3):
                if val != w[i]:
                    nw = list(w)
                    nw[i] = val
                    nbrs.append(word_to_idx(tuple(nw)))
        neighbors[idx] = nbrs

    # Precompute neighbor sets for fast intersection
    neighbor_sets = [set(neighbors[i]) for i in range(total_words)]

    # 1. Initial Code: Tetracode Product (729 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code_indices = []
    for head in tetracode:
        for tail in np.ndindex(3, 3, 3, 3):
            word = tuple(head + list(tail))
            initial_code_indices.append(word_to_idx(word))
    
    code_set = set(initial_code_indices)
    code_list = list(code_set)
    
    # Cover count array
    cover_count = np.zeros(total_words, dtype=int)
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1

    start_time = time.time()
    deadline = start_time + 115 # 115s limit to be safe within 120s/150s
    
    # 2. Greedy Removal Phase
    # Remove any word whose sphere is fully covered by others
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        for i in range(len(code_list) - 1, -1, -1):
            c = code_list[i]
            redundant = True
            for nbr in neighbors[c]:
                if cover_count[nbr] <= 1:
                    redundant = False
                    break
            if redundant:
                code_list.pop(i)
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                removed_any = True
                break # Restart scan after removal

    # 3. Simulated Annealing / Stochastic Search
    # We accept swaps that keep size same or reduce it.
    # We accept swaps that increase size (kick) with probability exp(-delta/E).
    # Here delta is the increase in size (1 for a kick).
    # We also allow "double kicks" (remove 2, add 3) if beneficial later? 
    # For simplicity, stick to single removals and single swaps (1-for-1 or 1-for-0).
    # A 1-for-1 swap changes the code but not size. It can enable future 1-for-0 removals.
    
    # Temperature schedule
    T_start = 1.0
    T_end = 0.01
    # We don't know exact number of steps, so use time-based cooling or step-based.
    # Let's use a fixed number of iterations if time permits, cooling linearly.
    # Estimate iterations: ~10ms per iter? 10000 iters = 100s.
    # Let's aim for 5000-10000 iterations.
    
    max_iters = 8000
    T = T_start
    
    for _ in range(max_iters):
        if time.time() >= deadline:
            break
            
        # Cool down
        # Simple geometric cooling
        T = T_start * (T_end / T_start) ** (_ / max_iters)
        
        # 1. Try to remove a word (1-for-0)
        # Pick a random word from the code
        idx_c = random.randint(0, len(code_list) - 1)
        c = code_list[idx_c]
        
        # Check if removable
        removable = True
        for nbr in neighbors[c]:
            if cover_count[nbr] <= 1:
                removable = False
                break
        
        if removable:
            # Remove it
            code_list.pop(idx_c)
            for nbr in neighbors[c]:
                cover_count[nbr] -= 1
            continue
        
        # 2. If not removable, try a swap (1-for-1)
        # Temporarily remove c
        for nbr in neighbors[c]:
            cover_count[nbr] -= 1
        
        # Find uncovered words U
        uncovered = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
        
        if not uncovered:
            # Should have been removable, but logic error? Or race condition?
            # Just revert and continue
            for nbr in neighbors[c]:
                cover_count[nbr] += 1
            continue
            
        # Find a replacement c' that covers all words in U
        # c' must be in intersection of neighbors[u] for u in U
        # And c' not in code_list
        
        # Optimization: If |U| > 2, it's unlikely a single word covers all.
        # Max distance between any two words in U is at most 2 (since both are dist 1 from c).
        # If dist(u1, u2) > 2, no common neighbor.
        # If dist(u1, u2) == 2, there are 2 common neighbors (in ternary, usually).
        # If dist(u1, u2) == 1, there are many.
        
        # Calculate candidates
        candidates = None
        for u in uncovered:
            if candidates is None:
                candidates = set(neighbor_sets[u])
            else:
                candidates.intersection_update(neighbor_sets[u])
                if not candidates:
                    break
        
        if candidates:
            # Remove current code words
            candidates.difference_update(code_set)
            
            if candidates:
                # Pick a candidate
                # Heuristic: Pick one that maximizes redundancy (max cover_count of its neighbors)
                # to facilitate future removals.
                # Or random. Random helps exploration.
                # Let's pick randomly from top 10 by "redundancy score" or just random.
                # Random is safer for SA.
                cp = random.choice(list(candidates))
                
                # Perform swap
                code_list[idx_c] = cp
                code_set.add(cp)
                code_set.discard(c) # c was removed from list, but we need to update set
                # Wait, code_list.pop(idx_c) wasn't done, we replaced it.
                # We need to update code_set properly.
                # Old code: code_set.discard(c) was done implicitly? No.
                # Let's manage code_set carefully.
                
                # Actually, let's just rebuild code_set if we swap? No, too slow.
                # code_list[idx_c] was c. Now it is cp.
                # code_set should have c removed and cp added.
                code_set.remove(c)
                code_set.add(cp)
                
                # Update cover counts
                for nbr in neighbors[cp]:
                    cover_count[nbr] += 1
                    
                # We accepted this swap. It maintains size.
                # Does it improve "energy"? 
                # In standard SA for set cover, we minimize size. Size is constant.
                # So we accept all valid 1-for-1 swaps? 
                # That might just wander.
                # Better: Only accept if it increases the minimum cover count (more redundancy)?
                # Or accept with probability based on some metric?
                # For now, accept all valid swaps to explore state space.
                # The 1-for-0 removals will drive the optimization.
                
            else:
                # No candidate, revert
                for nbr in neighbors[c]:
                    cover_count[nbr] += 1
        else:
            # No candidate, revert
            for nbr in neighbors[c]:
                cover_count[nbr] += 1

    # Final cleanup: try removals again
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        for i in range(len(code_list) - 1, -1, -1):
            c = code_list[i]
            redundant = True
            for nbr in neighbors[c]:
                if cover_count[nbr] <= 1:
                    redundant = False
                    break
            if redundant:
                code_list.pop(i)
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                removed_any = True
                break

    return [list(idx_to_word(c)) for c in code_list]
