import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: simple greedy
        # This task is specific to q=3, n=8, radius=1
        # But we must handle the general case or at least not crash
        # The contract says: "The evaluator passes this task's alphabet size q, length n and covering radius."
        # So we can assume q=3, n=8, radius=1 for the main logic, but keep a fallback.
        
        # Fallback: Greedy Set Cover
        total_words = q ** n
        # Map word tuple to index
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
        neighbors = {}
        for idx in range(total_words):
            w = idx_to_word(idx)
            nbrs = {idx}
            for i in range(n):
                for val in range(q):
                    if val != w[i]:
                        nw = list(w)
                        nw[i] = val
                        nbrs.add(word_to_idx(tuple(nw)))
            neighbors[idx] = list(nbrs)

        uncovered = [True] * total_words
        code = []
        
        while any(uncovered):
            # Find word covering most uncovered
            best_idx = -1
            best_count = -1
            for idx in range(total_words):
                if idx in code: # Optimization: skip if already in code, though we can just check coverage
                    continue
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
    
    # 1. Precomputation
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

    # Precompute neighbors for all words
    # neighbors[i] is a list of indices within distance 1 of i
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

    # 2. Greedy Construction
    # uncovered: boolean array
    uncovered = np.ones(total_words, dtype=bool)
    # gains: array of how many uncovered words each candidate covers
    # Initially, all candidates cover 17 words (1 + 8*2)
    gains = np.full(total_words, 17, dtype=int)
    
    code = set()
    code_list = []
    
    # Distance 2 ball precomputation for gain updates
    # For a given center c, which words w have neighbors that intersect neighbors[c]?
    # This is equivalent to dist(w, c) <= 2.
    # We can precompute this list for each c, or calculate on the fly.
    # On the fly: for c, iterate all 6561 words? No, too slow.
    # Precompute dist2_ball[c] = list of w such that dist(w,c) <= 2.
    # There are 6561 * 129 ints. ~800k ints. Memory is fine.
    
    dist2_balls = [None] * total_words
    for c in range(total_words):
        # Find all words within distance 2 of c
        # dist 0: c
        # dist 1: neighbors[c]
        # dist 2: neighbors of neighbors[c] that are not in neighbors[c]
        ball = set(neighbors[c])
        for n1 in neighbors[c]:
            for n2 in neighbors[n1]:
                ball.add(n2)
        dist2_balls[c] = list(ball)

    start_time = time.time()
    deadline = start_time + 110  # Leave 10s for cleanup and return
    
    # Greedy Loop
    while np.any(uncovered) and time.time() < deadline:
        # Find max gain
        # Optimization: only consider words not in code?
        # Actually, if a word is in code, its gain is 0 effectively for selection, 
        # but we just need to pick the best available.
        # We can maintain a max-heap or just scan. Scanning 6561 is fast.
        best_idx = -1
        best_gain = -1
        for i in range(total_words):
            if i in code:
                continue
            if gains[i] > best_gain:
                best_gain = gains[i]
                best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            break
            
        # Add best_idx to code
        code.add(best_idx)
        code_list.append(best_idx)
        
        # Update uncovered
        for nbr in neighbors[best_idx]:
            uncovered[nbr] = False
            
        # Update gains for all w in dist2_balls[best_idx]
        # gain[w] decreases by the number of neighbors of w that are in neighbors[best_idx] AND were just uncovered
        # Actually, gain[w] = count of uncovered in neighbors[w].
        # When we uncover a set S (neighbors[best_idx]), 
        # for any w, gain[w] decreases by |neighbors[w] intersect S|.
        
        for w in dist2_balls[best_idx]:
            if w in code:
                continue
            # Calculate decrease
            dec = 0
            for nbr_w in neighbors[w]:
                if not uncovered[nbr_w]: # It was just uncovered (or was already uncovered)
                    # We need to know if it was uncovered *before* this step.
                    # Since we update uncovered immediately, we can't distinguish.
                    # Better: calculate intersection of neighbors[w] and neighbors[best_idx]
                    pass
            
        # This update is O(129 * 17) per step if we do intersection.
        # 129 * 17 ~ 2200 ops.
        # 486 steps * 2200 ~ 1M ops. Very fast.
        
        # Let's do it properly.
        # We need to update gains[w] for w in dist2_balls[best_idx].
        # gain[w] -= |neighbors[w] intersect neighbors[best_idx]|
        # BUT only for those neighbors that are actually in the space and valid.
        # And we must ensure we don't go below 0.
        
        # Precomputed intersection sizes?
        # inter_size[w][c] = |neighbors[w] intersect neighbors[c]|
        # This is symmetric.
        # We can compute this on the fly.
        
        for w in dist2_balls[best_idx]:
            if w in code:
                continue
            # Count common neighbors
            # neighbors[w] and neighbors[best_idx]
            # We can use sets for fast intersection?
            # neighbors are lists.
            # Convert to sets? 6561 sets of size 17. Memory okay.
            # Let's precompute neighbor sets.
            
    # The above greedy loop update logic was getting messy. 
    # Let's restart the greedy part with a cleaner, slightly slower but correct update, 
    # or just use the simple scan which is fast enough for 6561 items.
    
    # Revised Greedy Strategy:
    # 1. uncovered array
    # 2. Loop:
    #    a. Find best_idx by scanning all i not in code, calculating gain on the fly?
    #       Gain = sum(uncovered[nbr] for nbr in neighbors[i])
    #       This is 6561 * 17 ~ 110k ops per iteration.
    #       486 iterations * 110k ~ 50M ops.
    #       In Python, 50M simple ops might take 5-10 seconds. Acceptable.
    
    code = set()
    code_list = []
    uncovered = np.ones(total_words, dtype=bool)
    
    start_time = time.time()
    
    while np.any(uncovered) and time.time() < deadline:
        best_idx = -1
        best_gain = -1
        for i in range(total_words):
            if i in code:
                continue
            # Calculate gain
            gain = 0
            for nbr in neighbors[i]:
                if uncovered[nbr]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1:
            break
            
        code.add(best_idx)
        code_list.append(best_idx)
        
        for nbr in neighbors[best_idx]:
            uncovered[nbr] = False

    # 3. Local Search / Pruning
    # Convert to list for easy manipulation
    code_list = list(code)
    cover_count = np.zeros(total_words, dtype=int)
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1

    # Precompute neighbor sets for fast intersection checks in swaps
    neighbor_sets = [set(neighbors[i]) for i in range(total_words)]

    removed_any = True
    while removed_any and time.time() < deadline:
        removed_any = False
        for i in range(len(code_list) - 1, -1, -1):
            c = code_list[i]
            # Check if c is redundant
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
                break # Restart scan

    # Kick and Prune Phase
    # Try to escape local minima by swapping words
    # We accept a swap if it allows for a net reduction in size after pruning.
    # To speed this up, we limit the number of swap attempts.
    
    swap_attempts = 0
    max_swap_attempts = 50000 # Tune based on time
    
    while time.time() < deadline and swap_attempts < max_swap_attempts:
        # 1. Try to remove
        removed = False
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
                removed = True
                break
        
        if removed:
            continue # Restart swap search after successful removal
            
        # 2. No removal possible. Try a swap.
        if len(code_list) == 0:
            break
            
        # Pick a random word to remove
        idx_c = random.randint(0, len(code_list) - 1)
        c = code_list[idx_c]
        
        # Temporarily remove c
        for nbr in neighbors[c]:
            cover_count[nbr] -= 1
            
        # Find uncovered words
        uncovered_now = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
        
        if not uncovered_now:
            # Should have been removable
            code_list.pop(idx_c)
            continue
        
        # Find a replacement c' that covers all uncovered_now
        # c' must not be in code_list
        # We search for c' among all words
        # Optimization: c' must be within distance 1 of all words in uncovered_now?
        # No, c' just needs to be within distance 1 of each word in uncovered_now.
        # This means c' is in the intersection of neighbors[u] for u in uncovered_now.
        
        # Calculate intersection of neighbor sets of uncovered words
        # Start with set of neighbors of first uncovered word
        candidates = set(neighbors[uncovered_now[0]])
        for u in uncovered_now[1:]:
            candidates.intersection_update(neighbors[u])
            if not candidates:
                break
        
        # Remove current code words from candidates
        # Also remove c itself? c is removed, so it's a valid candidate technically, 
        # but swapping c with c is useless.
        candidates.difference_update(code_list)
        
        if candidates:
            # Pick a candidate. 
            # Heuristic: Pick the one that covers the most *already covered* words 
            # to create redundancy for future removals?
            # Or just pick random?
            # Let's pick the one with max overlap with existing code neighbors.
            best_cp = -1
            best_overlap = -1
            for cp in candidates:
                # Count how many neighbors of cp are already in code?
                # Or how many neighbors of cp have high cover_count?
                # Let's just pick random for simplicity and speed, 
                # or the one that maximizes cover_count of its neighbors.
                # Max cover_count of neighbors means it's adding redundancy.
                max_cov = max(cover_count[nbr] for nbr in neighbors[cp])
                if max_cov > best_overlap:
                    best_overlap = max_cov
                    best_cp = cp
            
            if best_cp != -1:
                # Perform swap
                code_list[idx_c] = best_cp
                for nbr in neighbors[best_cp]:
                    cover_count[nbr] += 1
                swap_attempts += 1
            else:
                # Revert
                for nbr in neighbors[c]:
                    cover_count[nbr] += 1
                swap_attempts += 1
        else:
            # No candidate found, revert
            for nbr in neighbors[c]:
                cover_count[nbr] += 1
            swap_attempts += 1

    return [list(idx_to_word(c)) for c in code_list]
