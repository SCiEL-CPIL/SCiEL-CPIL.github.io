import time
import numpy as np
import random

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    space_size = q ** n

    # Precompute neighbors (ball of radius 1) for each word
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    neighbors = np.empty(space_size, dtype=object)
    for idx in range(space_size):
        nbs = [idx]
        for pos in range(n):
            digit = (idx // q_powers[n - 1 - pos]) % q
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                delta = (new_digit - digit) * q_powers[n - 1 - pos]
                nbs.append(idx + delta)
        neighbors[idx] = np.array(nbs, dtype=np.int32)

    # Precompute cover_map: cover_map[w] = list of codewords whose ball covers w
    # This is the reverse of neighbors: for each codeword c, for each w in neighbors[c], add c to cover_map[w]
    cover_map = np.empty(space_size, dtype=object)
    for w in range(space_size):
        cover_map[w] = []
    for c in range(space_size):
        for w in neighbors[c]:
            cover_map[w].append(c)
    for w in range(space_size):
        cover_map[w] = np.array(cover_map[w], dtype=np.int32)

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Greedy set cover to get initial valid code
    def greedy_cover(uncovered_mask):
        """Greedy: repeatedly pick codeword covering most uncovered words."""
        code = []
        cov_count = np.zeros(space_size, dtype=np.int32)
        in_set = np.zeros(space_size, dtype=bool)
        # We need to track which words are uncovered
        # Use a boolean array for uncovered
        # For speed, we'll track uncovered as a set-like structure via numpy
        # Actually, let's use a different approach:
        # maintain uncovered as a numpy boolean array, and for each candidate,
        # count how many of its neighbors are uncovered.
        
        # To make this fast, precompute neighbor lists as numpy arrays (already done)
        # For each iteration, compute for all c: sum(uncovered[neighbors[c]])
        # This is expensive if done naively. Instead, use the cover_map:
        # For each uncovered word w, all c in cover_map[w] can cover it.
        # So we can do: for each uncovered w, for each c in cover_map[w]: score[c] += 1
        # Then pick argmax score.
        
        # But recomputing score from scratch each iteration is O(|uncovered| * avg_cover_size)
        # Let's try it.
        
        while True:
            if time.time() - t_start > deadline - 10:
                break
            uncovered_indices = np.where(uncovered_mask)[0]
            if len(uncovered_indices) == 0:
                break
            
            # Compute score for each codeword: number of uncovered words it covers
            score = np.zeros(space_size, dtype=np.int32)
            for w in uncovered_indices:
                for c in cover_map[w]:
                    score[c] += 1
            
            best_c = np.argmax(score)
            if score[best_c] == 0:
                # No codeword can cover any uncovered word (shouldn't happen)
                break
            
            code.append(best_c)
            in_set[best_c] = True
            # Update uncovered: mark all neighbors of best_c as covered
            uncovered_mask[neighbors[best_c]] = False
        
        return code

    # Initial greedy cover
    uncovered_mask = np.ones(space_size, dtype=bool)
    initial_code = greedy_cover(uncovered_mask)

    # If greedy didn't finish (timeout), force-cover remaining
    if uncovered_mask.any():
        remaining = np.where(uncovered_mask)[0]
        for w in remaining:
            initial_code.append(w)
            uncovered_mask[neighbors[w]] = False

    # Local search: repeatedly try to remove a codeword that doesn't leave any word uncovered
    # A codeword c is "removable" if for every w in neighbors[c], there exists another codeword c' in the set such that w in neighbors[c']
    # Equivalently, c is removable if min_{w in neighbors[c]} (number of codewords in set covering w) >= 1 after removing c
    # i.e., for all w in neighbors[c], count(w) >= 2 before removal (so after removal count >= 1)
    
    def local_search(code):
        """Try to remove redundant codewords."""
        code_set = set(code)
        # Compute coverage count for each word
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code:
            cov[neighbors[c]] += 1
        
        improved = True
        while improved:
            if time.time() - t_start > deadline - 10:
                break
            improved = False
            # Try to remove codewords in random order
            random.shuffle(code)
            for i in range(len(code)):
                c = code[i]
                # Check if removing c leaves any word with cov==0
                # i.e., is there any w in neighbors[c] with cov[w] == 1?
                # If yes, c is critical (cannot remove)
                # If no, c is redundant (can remove)
                ball = neighbors[c]
                if np.any(cov[ball] == 1):
                    continue  # critical, cannot remove
                # Remove c
                cov[ball] -= 1
                code.pop(i)
                code_set.discard(c)
                improved = True
                break  # restart the loop since code changed
        
        return code

    # Kick: remove k random codewords, then re-greedy cover the newly uncovered words
    def kick(code, k):
        """Remove k random codewords and re-cover."""
        if len(code) <= k:
            return code[:]
        code = code[:]
        # Pick k random indices to remove
        indices_to_remove = random.sample(range(len(code)), k)
        indices_to_remove.sort(reverse=True)
        for i in indices_to_remove:
            code.pop(i)
        
        # Now re-cover: find uncovered words and greedily add codewords
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code:
            cov[neighbors[c]] += 1
        
        uncovered_mask = (cov == 0)
        if not uncovered_mask.any():
            return code
        
        # Greedy cover the uncovered
        added = greedy_cover(uncovered_mask)
        code.extend(added)
        
        # Force-cover any remaining
        if uncovered_mask.any():
            remaining = np.where(uncovered_mask)[0]
            for w in remaining:
                code.append(w)
                uncovered_mask[neighbors[w]] = False
        
        return code

    # ILS: local search, kick, local search, ...
    best_code = initial_code[:]
    best_size = len(best_code)
    
    # Local search the initial code
    current_code = local_search(initial_code)
    best_code = current_code[:]
    best_size = len(best_code)
    
    rng = random.Random(42)
    restart_count = 0
    
    while time.time() - t_start < deadline - 5:
        restart_count += 1
        # Kick: remove 3-8 random codewords
        k = rng.randint(3, min(8, max(1, len(current_code) - 5)))
        kicked = kick(current_code, k)
        
        # Local search
        searched = local_search(kicked)
        
        if len(searched) < best_size:
            best_code = searched[:]
            best_size = len(searched)
            current_code = searched[:]
        else:
            # Keep current or go back to best
            if len(searched) <= len(current_code):
                current_code = searched[:]
            else:
                current_code = best_code[:]
    
    # Verify best_code is valid
    final_cov = np.zeros(space_size, dtype=np.int32)
    for c in best_code:
        final_cov[neighbors[c]] += 1
    
    if np.any(final_cov == 0):
        # Fallback to initial_code
        return [idx_to_word(c) for c in initial_code]
    else:
        return [idx_to_word(c) for c in best_code]
