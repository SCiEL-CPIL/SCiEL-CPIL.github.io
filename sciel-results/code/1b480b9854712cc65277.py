import time
import random
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: simple greedy
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
        
        # Simple Greedy Set Cover
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

    # Precompute neighbors in a NumPy array for fast vectorized operations
    # Shape: (total_words, 17)
    # 17 = 1 (self) + 8 * 2 (flips)
    neighbors_matrix = np.empty((total_words, 17), dtype=np.int32)
    
    for idx in range(total_words):
        w = [0] * 8
        temp = idx
        for i in range(7, -1, -1):
            w[i] = temp % 3
            temp //= 3
        
        neighbors_matrix[idx, 0] = idx
        k = 1
        for i in range(8):
            for val in range(3):
                if val != w[i]:
                    nw = w.copy()
                    nw[i] = val
                    neighbors_matrix[idx, k] = word_to_idx(tuple(nw))
                    k += 1

    start_time = time.time()
    deadline = start_time + 115 # 115s limit to be safe within 120s/150s

    # Phase 1: Greedy Set Cover Construction
    # This is a global construction method, distinct from refining a product code.
    uncovered = np.ones(total_words, dtype=bool)
    selected = np.zeros(total_words, dtype=bool)
    code_list = []
    
    # We might run multiple randomized greedy attempts if time permits, 
    # but a single deterministic greedy is a strong baseline.
    # To add structure/randomness, we can break ties randomly.
    
    while uncovered.any():
        if time.time() > deadline - 20: # Leave time for local search
            break
            
        # Vectorized count of uncovered neighbors for all candidates
        # uncovered[neighbors_matrix] creates a (6561, 17) bool array
        # .sum(axis=1) sums over the 17 neighbors for each candidate
        counts = uncovered[neighbors_matrix].sum(axis=1)
        
        # Mask out already selected codewords
        counts[selected] = -1
        
        best_idx = np.argmax(counts)
        
        if counts[best_idx] <= 0:
            # No more coverage possible? Should not happen if uncovered is not empty
            break
            
        code_list.append(best_idx)
        selected[best_idx] = True
        # Update uncovered status
        uncovered[neighbors_matrix[best_idx]] = False

    # Phase 2: Local Search Refinement
    # Convert code_list to a set for O(1) lookup
    code_set = set(code_list)
    
    # Initialize cover_count
    cover_count = np.zeros(total_words, dtype=np.int32)
    for c in code_list:
        cover_count[neighbors_matrix[c]] += 1

    # Helper to check if a codeword is redundant
    def is_redundant(c):
        # A codeword is redundant if all its neighbors are covered by >= 1 other codeword
        # i.e., cover_count[nbr] >= 2 for all nbr in neighbors[c]
        # Note: cover_count includes c itself.
        # So we need min(cover_count[neighbors_matrix[c]]) >= 2
        return np.min(cover_count[neighbors_matrix[c]]) >= 2

    # Removal Phase
    removed_any = True
    while removed_any and time.time() < deadline - 5:
        removed_any = False
        for i in range(len(code_list) - 1, -1, -1):
            c = code_list[i]
            if is_redundant(c):
                # Remove c
                code_list.pop(i)
                code_set.discard(c)
                cover_count[neighbors_matrix[c]] -= 1
                removed_any = True
                break # Restart scan to maintain efficiency/correctness of indices

    # Swap Phase (1-for-1) to enable further removals
    # We try to swap a codeword c with c' such that:
    # 1. c' is not in code_set
    # 2. After removing c, the uncovered points U are covered by c'
    # 3. c' is "better" (e.g., leads to higher redundancy)
    
    # Precompute neighbor sets for intersection logic if needed, 
    # but for 1-for-1 swaps in covering codes, we can just check coverage.
    
    # To speed up swap search, we can iterate through code_list and try to find a swap.
    # Randomization helps escape local minima.
    
    max_swap_iters = 2000
    for _ in range(max_swap_iters):
        if time.time() > deadline - 5:
            break
            
        if not code_list:
            break
            
        # Pick a random codeword to potentially swap
        idx_c = random.randint(0, len(code_list) - 1)
        c = code_list[idx_c]
        
        # Temporarily remove c
        cover_count[neighbors_matrix[c]] -= 1
        
        # Find uncovered points U
        # U = { nbr in neighbors[c] | cover_count[nbr] == 0 }
        # Note: If c was the only cover for some nbr, that nbr is now uncovered.
        u_mask = (cover_count[neighbors_matrix[c]] == 0)
        if not np.any(u_mask):
            # c was redundant, should have been removed. Revert.
            cover_count[neighbors_matrix[c]] += 1
            continue
        
        u_indices = neighbors_matrix[c][u_mask]
        
        # Find candidates c' that cover all points in u_indices
        # c' must be in intersection of neighbors[u] for all u in u_indices
        # And c' not in code_set
        
        # Optimization: If len(u_indices) is large, intersection is small.
        # Start with the neighbors of the first uncovered point
        candidates = set(neighbors_matrix[u_indices[0]].tolist())
        
        for u in u_indices[1:]:
            candidates.intersection_update(neighbors_matrix[u].tolist())
            if not candidates:
                break
        
        if candidates:
            # Remove current code words from candidates
            candidates.difference_update(code_set)
            
            if candidates:
                # Pick a candidate. 
                # Heuristic: Pick one that maximizes the minimum cover count of its neighbors 
                # (excluding the current state) to promote future removals.
                # Or just random.
                # Let's pick randomly to explore.
                cp = random.choice(list(candidates))
                
                # Perform swap
                code_list[idx_c] = cp
                code_set.add(cp)
                code_set.discard(c)
                
                # Update cover counts
                cover_count[neighbors_matrix[cp]] += 1
                
                # Try to remove redundant words immediately after a successful swap
                # This helps chain improvements
                for j in range(len(code_list) - 1, -1, -1):
                    cj = code_list[j]
                    if is_redundant(cj):
                        code_list.pop(j)
                        code_set.discard(cj)
                        cover_count[neighbors_matrix[cj]] -= 1
                        # Continue removing if possible
                        # We break to restart the outer swap loop or continue?
                        # Let's just break the inner removal loop and let the outer loop continue
                        break
            else:
                # No candidate, revert
                cover_count[neighbors_matrix[c]] += 1
        else:
            # No candidate, revert
            cover_count[neighbors_matrix[c]] += 1

    # Final cleanup
    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        for i in range(len(code_list) - 1, -1, -1):
            c = code_list[i]
            if is_redundant(c):
                code_list.pop(i)
                code_set.discard(c)
                cover_count[neighbors_matrix[c]] -= 1
                removed_any = True
                break

    return [list(idx_to_word(c)) for c in code_list]
