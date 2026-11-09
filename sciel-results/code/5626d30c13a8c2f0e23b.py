import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 105  # Safety margin for 120s limit
    
    if not (q == 3 and n == 8 and radius == 1):
        # Fallback for other parameters
        total_words = q ** n
        centers = []
        covered = set()
        all_words = list(itertools.product(range(q), repeat=n))
        candidates = list(all_words)
        random.shuffle(candidates)
        
        while covered != set(all_words) and time.time() < deadline:
            best_center = None
            max_new = 0
            for c in candidates:
                neigh = [c]
                for i in range(n):
                    for val in range(q):
                        if val == c[i]: continue
                        new_c = list(c)
                        new_c[i] = val
                        neigh.append(tuple(new_c))
                new_covered = [w for w in neigh if w not in covered]
                if len(new_covered) > max_new:
                    max_new = len(new_covered)
                    best_center = c
                    best_neigh = neigh
            if best_center is None or max_new == 0:
                break
            centers.append(list(best_center))
            for w in best_neigh:
                covered.add(w)
            candidates.remove(best_center)
        return centers
    
    # --- Optimized Path for q=3, n=8, radius=1 ---
    total_words = 3 ** 8  # 6561
    
    # 1. Pre-compute mapping and neighborhoods
    word_to_idx = {}
    idx_to_word = [None] * total_words
    
    count = 0
    for word in itertools.product(range(3), repeat=8):
        idx = 0
        for digit in word:
            idx = idx * 3 + digit
        word_to_idx[word] = idx
        idx_to_word[idx] = word
        count += 1
    
    # neighbors[i] is a list of indices covered by center i (including i itself)
    neighbors = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        center = idx_to_word[i]
        neighbors[i].append(i)
        for pos in range(8):
            old_val = center[pos]
            for new_val in range(3):
                if new_val == old_val:
                    continue
                neighbor = list(center)
                neighbor[pos] = new_val
                n_idx = word_to_idx[tuple(neighbor)]
                neighbors[i].append(n_idx)
    
    # 2. Start from the 729-word construction (All_4 x Tetracode_4)
    # This is a valid covering code of radius 1.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    all4 = list(itertools.product(range(3), repeat=4))
    code_words = [list(h) + list(t) for h in all4 for t in tetracode]
    
    # Convert to indices
    code_indices = [word_to_idx[tuple(w)] for w in code_words]
    code_set = set(code_indices)
    
    # Build counts: count[j] = number of centers in code_set covering j
    counts = [0] * total_words
    
    # Helper to update counts
    def add_counts(idx):
        for j in neighbors[idx]:
            counts[j] += 1
            
    def remove_counts(idx):
        for j in neighbors[idx]:
            counts[j] -= 1

    for idx in code_indices:
        add_counts(idx)
        
    # 3. Local Search Minimization
    # Phase A: Greedy Removal
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        for idx in list(code_set):
            # Temporarily remove
            remove_counts(idx)
            
            can_remove = True
            for j in neighbors[idx]:
                if counts[j] == 0:
                    can_remove = False
                    break
            
            if can_remove:
                code_set.remove(idx)
                removed_any = True
            else:
                # Restore
                add_counts(idx)
                
    # Phase B: Swap Search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Run removal phase first
        removed_any = True
        while removed_any:
            removed_any = False
            for idx in list(code_set):
                remove_counts(idx)
                can_remove = True
                for j in neighbors[idx]:
                    if counts[j] == 0:
                        can_remove = False
                        break
                if can_remove:
                    code_set.remove(idx)
                    removed_any = True
                else:
                    add_counts(idx)
                    
        # Now try swaps
        for c in list(code_set):
            if time.time() > deadline:
                break
                
            # Find neighbors of c that are ONLY covered by c
            critical = []
            for j in neighbors[c]:
                if counts[j] == 1:
                    critical.append(j)
                    
            if not critical:
                continue
                
            # Find c' not in code_set that covers all 'critical'
            found_swap = False
            for c_prime in range(total_words):
                if c_prime in code_set:
                    continue
                    
                valid_swap = True
                for j in critical:
                    if j not in neighbors[c_prime]:
                        valid_swap = False
                        break
                        
                if valid_swap:
                    # Perform swap
                    remove_counts(c)
                    code_set.remove(c)
                    
                    add_counts(c_prime)
                    code_set.add(c_prime)
                    
                    improved = True
                    found_swap = True
                    break
                    
            if found_swap:
                break
        
    # Final verification
    covered_set = set()
    for i in code_set:
        covered_set.update(neighbors[i])
        
    if len(covered_set) != total_words:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        all4 = list(itertools.product(range(3), repeat=4))
        result = [list(h) + list(tail) for h in all4 for tail in tetracode]
        return result
        
    # Convert to list of lists
    result = [list(idx_to_word[idx]) for idx in code_set]
    return result
