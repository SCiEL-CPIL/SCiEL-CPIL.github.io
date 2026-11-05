import time
import random
import itertools

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters (not expected in this task)
        # Simple greedy
        code = []
        covered = [False] * (q**n)
        # Precompute neighbors for fallback
        def get_neighbors(w):
            nbrs = [w]
            for i in range(n):
                for val in range(q):
                    if val != w[i]:
                        nw = list(w)
                        nw[i] = val
                        nbrs.append(tuple(nw))
            return nbrs

        def word_to_idx(w):
            idx = 0
            for val in w:
                idx = idx * q + val
            return idx

        def idx_to_word(idx):
            w = [0]*n
            temp = idx
            for i in range(n-1, -1, -1):
                w[i] = temp % q
                temp //= q
            return tuple(w)

        all_words = [idx_to_word(i) for i in range(q**n)]
        neighbors = {w: [word_to_idx(x) for x in get_neighbors(w)] for w in all_words}
        
        uncovered = set(range(q**n))
        while uncovered:
            best_word = None
            best_count = -1
            # Randomly sample a few candidates to speed up
            candidates = list(uncovered)
            random.shuffle(candidates)
            for c_idx in candidates[:100]:
                c = all_words[c_idx]
                cnt = sum(1 for nb in neighbors[c] if nb in uncovered)
                if cnt > best_count:
                    best_count = cnt
                    best_word = c
            code.append(best_word)
            for nb in neighbors[best_word]:
                if nb in uncovered:
                    uncovered.remove(nb)
                    covered[nb] = True
            if time.time() > 100:
                break
        return [list(c) for c in code]

    # Specific logic for q=3, n=8, radius=1
    start_time = time.time()
    deadline = start_time + 110 # 10s margin
    
    total_words = q**n
    # Precompute neighbors for all words
    # neighbors[idx] = list of indices within distance 1
    neighbors = []
    for idx in range(total_words):
        w = [0]*n
        temp = idx
        for i in range(n-1, -1, -1):
            w[i] = temp % q
            temp //= q
        nbrs = [idx]
        for i in range(n):
            for val in range(q):
                if val != w[i]:
                    nw = list(w)
                    nw[i] = val
                    nidx = 0
                    for j in range(n):
                        nidx = nidx * q + nw[j]
                    nbrs.append(nidx)
        neighbors.append(nbrs)
        
    def idx_to_word(idx):
        w = [0]*n
        temp = idx
        for i in range(n-1, -1, -1):
            w[i] = temp % q
            temp //= q
        return tuple(w)

    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    # 1. Greedy Initialization
    # We want to cover all words.
    # Start with empty code.
    code_set = set()
    cover_count = [0] * total_words
    uncovered = set(range(total_words))
    
    # To speed up greedy, we can pick random uncovered words and find best cover
    # Or just iterate.
    # 6561 words. Greedy steps ~500.
    # Each step: scan all 6561 words? No, too slow.
    # Only consider words that cover uncovered nodes?
    # Let's just do a standard greedy:
    # Pick c that maximizes |N[c] ∩ Uncovered|
    
    while uncovered:
        if time.time() > deadline - 5:
            break
        best_c = None
        best_gain = -1
        # Sample from uncovered to find candidates?
        # No, we need the codeword that covers the most uncovered.
        # We can maintain a list of "active" codewords?
        # Simpler: Just iterate through all possible codewords (6561) and pick best.
        # 500 iterations * 6561 * 17 checks ~ 56M ops. Might be slow in Python.
        
        # Optimization: 
        # Pick a random uncovered word u.
        # The best codeword must be in N[u].
        # So we only check candidates in N[u].
        # There are 17 candidates.
        # Pick the one with max gain.
        
        u = random.choice(list(uncovered))
        candidates = neighbors[u]
        
        for c in candidates:
            gain = 0
            for nb in neighbors[c]:
                if nb in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_c = c
                
        if best_c is None:
            break
            
        code_set.add(best_c)
        for nb in neighbors[best_c]:
            cover_count[nb] += 1
            if cover_count[nb] == 1:
                uncovered.discard(nb)
                
    code_list = list(code_set)
    
    # 2. Local Search
    # Try to remove or swap.
    
    def try_remove_swap():
        # Returns True if a change was made
        # We scan the code_list.
        # To avoid bias, we can shuffle the scan order.
        
        # Pass 1: Removal
        for i in range(len(code_list) - 1, -1, -1):
            c = code_list[i]
            c_idx = word_to_idx(c)
            removable = True
            for nb in neighbors[c_idx]:
                if cover_count[nb] <= 1:
                    removable = False
                    break
            if removable:
                code_list.pop(i)
                for nb in neighbors[c_idx]:
                    cover_count[nb] -= 1
                return True
                
        # Pass 2: Swap
        # For each c, find U (uncovered if c removed).
        # If U is not empty, find C_and = intersection of N[u] for u in U.
        # If C_and is not empty, pick c' in C_and.
        # Swap c with c'.
        
        # To be efficient, we can just pick a random c to try swapping first?
        # Or iterate.
        
        # Let's iterate to find any valid swap.
        for i in range(len(code_list) - 1, -1, -1):
            c = code_list[i]
            c_idx = word_to_idx(c)
            
            # Find U
            U = []
            for nb in neighbors[c_idx]:
                if cover_count[nb] == 1:
                    U.append(nb)
            
            if not U:
                continue # Handled in removal pass (or redundant)
                
            # Find C_and
            # Start with neighbors of first u
            # C_and = set(neighbors[U[0]])
            # For each subsequent u, C_and = C_and & set(neighbors[u])
            
            # Optimization: If |U| is large, intersection is likely empty.
            # Max |U| is 17.
            
            # Use a set for intersection
            c_and = set(neighbors[U[0]])
            for u in U[1:]:
                c_and &= set(neighbors[u])
                if not c_and:
                    break
            
            if c_and:
                # Pick a candidate
                # Prefer one that is not in code_set (guaranteed)
                # and maybe covers many other high-count words?
                # Just pick first.
                c_prime_idx = next(iter(c_and))
                c_prime = idx_to_word(c_prime_idx)
                
                # Perform swap
                code_list[i] = c_prime
                code_set.remove(c)
                code_set.add(c_prime)
                
                # Update cover counts
                for nb in neighbors[c_idx]:
                    cover_count[nb] -= 1
                for nb in neighbors[c_prime_idx]:
                    cover_count[nb] += 1
                    
                return True
                
        return False

    # Main loop
    no_improve_count = 0
    while time.time() < deadline:
        if try_remove_swap():
            no_improve_count = 0
        else:
            no_improve_count += 1
            # If stuck, do a kick?
            # Kick: Remove a random c.
            # Find best c' to cover U (max coverage, not necessarily all).
            # Add c'.
            # This might increase size by 1 if we can't remove another?
            # No, we remove 1, add 1. Size constant.
            # But if c' doesn't cover all U, we have uncovered words.
            # We must fix them.
            
            if no_improve_count > 100:
                # Try a random restart of the local search phase?
                # Or just accept the current result if time is running out.
                # Let's try a "Random Swap" kick.
                # Pick random c.
                # Remove it.
                # U is uncovered.
                # Find c' that covers max U.
                # If c' covers all U, swap.
                # Else, revert.
                
                # This is effectively what try_remove_swap does, but it requires C_and to be non-empty.
                # If C_and is empty, no single word can cover U.
                # So we are stuck.
                
                # To escape, we might need to remove 2 and add 2?
                # Or just break and return.
                
                # Given the time limit, if we are stuck, we might just return.
                # But let's try one more thing: 
                # If stuck, shuffle the code_list order and try again?
                # The logic in try_remove_swap is deterministic based on list order.
                # Shuffling might find a different swap.
                
                random.shuffle(code_list)
                no_improve_count = 0
                continue
                
                # If we are truly stuck (no swaps possible in any order), we are at a local minimum.
                # We can try to "perturb" by removing a word and adding a random word, 
                # even if it breaks coverage, then repair?
                # Too complex.
                
                # Let's just break if we've tried many times.
                if no_improve_count > 1000:
                    break

    return [list(c) for c in code_list]

