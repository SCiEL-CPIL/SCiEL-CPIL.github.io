import itertools
import random
import time

def construct(q, n, radius):
    # Constants for the specific task instance
    # q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback for other instances, though task implies specific
        # Just return a trivial valid code (all words) if not the target case
        # This is inefficient but valid
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    # Precompute neighbors for all 729 words
    # Index: v[0]*3^5 + v[1]*3^4 + ... + v[5]
    # Convert index to tuple
    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n):
            t[n - 1 - i] = idx % q
            idx //= q
        return tuple(t)
    
    def tuple_to_idx(t):
        idx = 0
        for i in range(n):
            idx = idx * q + t[i]
        return idx

    # Precompute balls
    all_indices = list(range(q**n))
    neighbors = [[] for _ in range(q**n)]
    
    # Generate all words
    words = [tuple(t) for t in itertools.product(range(q), repeat=n)]
    
    for idx, w in enumerate(words):
        neighbors[idx].append(idx)
        for i in range(n):
            for d in range(1, q):
                new_w = list(w)
                new_w[i] = (new_w[i] + d) % q
                neighbors[idx].append(tuple_to_idx(tuple(new_w)))
        # Remove duplicates just in case, though logic above is unique
        neighbors[idx] = list(set(neighbors[idx]))
    
    # Convert neighbors to sets for faster membership testing
    neighbor_sets = [set(nb) for nb in neighbors]

    start_time = time.time()
    deadline = 145.0 # Safety margin (150s limit, 5s buffer)

    # Tetracode seed: K_3(4,1) has 9 words, direct product with K_3(2,1)
    # Tetracode is a perfect ternary code of length 4, radius 1
    # C = {(a,b,a+b,a+2b) : a,b in {0,1,2}}
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    # Direct product with all pairs of length 2
    code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            code.append(tuple(h + list(tail)))
    
    # Initialize cover counts
    cover_count = [0] * (q**n)
    for c in code:
        c_idx = tuple_to_idx(c)
        for nb in neighbors[c_idx]:
            cover_count[nb] += 1
    
    # Local Search
    def try_remove(c_tuple):
        c_idx = tuple_to_idx(c_tuple)
        uncovered = []
        for nb in neighbors[c_idx]:
            cover_count[nb] -= 1
            if cover_count[nb] == 0:
                uncovered.append(nb)
        
        if not uncovered:
            return True, [] # Successfully removed, no uncovered
        else:
            # Restore
            for nb in neighbors[c_idx]:
                cover_count[nb] += 1
            return False, uncovered

    def try_add(c_tuple):
        c_idx = tuple_to_idx(c_tuple)
        for nb in neighbors[c_idx]:
            cover_count[nb] += 1

    def find_swap_target(uncovered_indices):
        # Find a word w such that all indices in uncovered_indices are in neighbors[w]
        # Heuristic: Pick a random point from uncovered, check its neighbors
        if not uncovered_indices:
            return None
        
        # Try multiple uncovered points to increase chance of finding a good swap
        candidates_to_try = uncovered_indices[:min(5, len(uncovered_indices))]
        random.shuffle(candidates_to_try)
        
        for u0 in candidates_to_try:
            candidates = list(neighbors[u0])
            random.shuffle(candidates)
            
            for c_idx in candidates:
                # Check if c_idx covers all uncovered
                c_nb_set = neighbor_sets[c_idx]
                if all(u in c_nb_set for u in uncovered_indices):
                    return idx_to_tuple(c_idx)
        
        # If no single neighbor covers all, try to find a codeword that covers
        # at least the most uncovered words (for partial improvement)
        if len(uncovered_indices) > 1:
            # Find the codeword that covers the most uncovered words
            best_c_idx = -1
            best_covered = 0
            u0 = uncovered_indices[0]
            candidates = list(neighbors[u0])
            random.shuffle(candidates)
            
            for c_idx in candidates:
                c_nb_set = neighbor_sets[c_idx]
                covered = sum(1 for u in uncovered_indices if u in c_nb_set)
                if covered > best_covered:
                    best_covered = covered
                    best_c_idx = c_idx
            
            if best_c_idx != -1 and best_covered > len(uncovered_indices) // 2:
                return idx_to_tuple(best_c_idx)
        
        return None

    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(code) # Randomize order to avoid bias
        
        # Phase 1: Remove
        for i in range(len(code)):
            if time.time() > deadline:
                break
            c = code[i]
            success, uncovered = try_remove(c)
            if success:
                code.pop(i)
                improved = True
                break # Restart scan
        
        if time.time() > deadline:
            break
            
        if improved:
            continue
            
        # Phase 2: Swap - try to improve by swapping codewords
        for i in range(len(code)):
            if time.time() > deadline:
                break
            c = code[i]
            c_idx = tuple_to_idx(c)
            success, uncovered = try_remove(c)
            if success:
                # Should have been caught in Phase 1, but just in case
                code.pop(i)
                improved = True
                break
            
            # Try to find a swap that covers the uncovered words
            if uncovered:
                new_c = find_swap_target(uncovered)
                if new_c is not None:
                    # Perform swap
                    # c is already removed from cover_count
                    try_add(new_c)
                    code[i] = new_c
                    improved = True
                    break
                else:
                    # Restore c
                    for nb in neighbors[c_idx]:
                        cover_count[nb] += 1
            else:
                # This case (uncovered empty) means removal was possible, handled above
                pass
        
        # Phase 3: Try adding new codewords if we have room
        if not improved and len(code) < 75 and time.time() < deadline - 10:
            # Try to find a codeword that covers many uncovered words
            # First, find which words are not covered
            uncovered_words = set()
            for w in range(q**n):
                if cover_count[w] == 0:
                    uncovered_words.add(w)
            
            if uncovered_words:
                # Find the best codeword to add
                best_add_idx = -1
                best_add_count = 0
                
                for c_idx in range(q**n):
                    if time.time() > deadline - 10:
                        break
                    count = 0
                    for nb in neighbors[c_idx]:
                        if nb in uncovered_words:
                            count += 1
                    if count > best_add_count:
                        best_add_count = count
                        best_add_idx = c_idx
                
                if best_add_idx != -1 and best_add_count > 0:
                    # Check if this codeword is already in the code
                    if idx_to_tuple(best_add_idx) not in code:
                        try_add(idx_to_tuple(best_add_idx))
                        code.append(idx_to_tuple(best_add_idx))
                        improved = True

    # Final cleanup: ensure no duplicate codewords
    seen = set()
    unique_code = []
    for c in code:
        c_key = tuple(c)
        if c_key not in seen:
            seen.add(c_key)
            unique_code.append(c)
    code = unique_code
    
    # Convert back to lists
    return [list(c) for c in code]
