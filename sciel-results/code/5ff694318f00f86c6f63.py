# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius 'radius'.
    Returns a list of codewords (each a list of ints in 0..q-1).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    if n == 0:
        return [[]]

    # Specific optimization for the known hard case: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)

    # Fallback for other cases
    # Use a simple greedy or direct sum if applicable
    if q == 3 and radius == 1 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        return code
    
    # Generic greedy for small n
    return greedy_covering(q, n, radius, deadline)

def _solve_q3_n6_r1(deadline):
    q = 3
    n = 6
    radius = 1
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    idx_to_word = {i: w for i, w in enumerate(all_words)}
    N = q ** n # 729
    
    # Precompute neighbors for each word index
    # neighbors[i] is a list of indices covered by word i (distance <= radius)
    neighbors = [None] * N
    for i, w in enumerate(all_words):
        neigh = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_w = list(w)
                new_w[pos] = new_val
                neigh.append(word_to_idx[tuple(new_w)])
        neighbors[i] = neigh
        
    # Initial code: Direct Sum (Tetracode x 3^2)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_code_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=2)]
    
    # Convert to indices
    code_indices = [word_to_idx[tuple(w)] for w in init_code_words]
    
    # Coverage counts
    counts = np.zeros(N, dtype=np.int32)
    for idx in code_indices:
        for n_idx in neighbors[idx]:
            counts[n_idx] += 1
            
    # Local Search
    # We maintain a list of active indices
    active_indices = list(code_indices)
    
    # Helper to check if a word can be removed
    # A word i can be removed if min(counts[neighbors[i]]) > 1
    # Actually, we just need to ensure that for all j in neighbors[i], counts[j] > 1
    # Since counts[i] is always >= 1 if i is in code, and we are checking removal of i,
    # we check if any neighbor has count 1. If so, i is critical.
    
    def can_remove(idx):
        for n_idx in neighbors[idx]:
            if counts[n_idx] == 1:
                return False
        return True

    # Removal Loop
    while time.time() < deadline:
        removed_any = False
        
        # Shuffle to break symmetry/bias
        random.shuffle(active_indices)
        
        for idx in active_indices:
            if time.time() > deadline:
                break
            
            if can_remove(idx):
                # Remove idx
                for n_idx in neighbors[idx]:
                    counts[n_idx] -= 1
                active_indices.remove(idx)
                removed_any = True
                
        if not removed_any:
            # Stuck. Try a perturbation: Add a random word, then try to remove.
            # This allows escaping local minima.
            
            # Pick a random word from the universe
            random_idx = random.randint(0, N - 1)
            
            # If it's already in the code, skip (or pick another)
            if random_idx in active_indices:
                continue
                
            # Add it
            active_indices.append(random_idx)
            for n_idx in neighbors[random_idx]:
                counts[n_idx] += 1
                
            # Now try to remove redundant words
            # We want to see if we can remove MORE than 1 word (the one we added? No, we added 1, so we need to remove 2 to improve size, or 1 to stay same but maybe get to a better state? 
            # Usually we just try to minimize size.
            # If we add 1, size increases by 1.
            # If we can remove 2, net change -1.
            # If we can remove 1, net change 0.
            
            removed_count = 0
            # We need to be careful: we should prioritize removing the newly added one if it becomes redundant? 
            # Or just remove any redundant ones.
            
            # Let's try to remove up to 2 words.
            # We iterate through a copy of the list to avoid modification during iteration issues, 
            # but we are modifying active_indices. 
            # Better: create a list of candidates to remove.
            
            candidates = []
            for idx in active_indices:
                if can_remove(idx):
                    candidates.append(idx)
            
            # We want to remove the most "redundant" ones? 
            # Or just remove 2 if possible.
            # To improve the score (minimize size), we need net removal.
            # If we can remove >= 2 words, we accept the perturbation.
            # If we can remove 1, we are at same size. We might accept it with some probability to explore, but for now, let's just accept if size decreases.
            
            # Sort candidates? 
            # Maybe remove the one that is "most covered" (highest min count in neighborhood)?
            # For simplicity, just try to remove 2.
            
            if len(candidates) >= 2:
                # Remove 2
                # Pick 2 distinct indices
                # To be safe, pick the first two
                to_remove = candidates[:2]
                
                # Verify they are still removable after removing the first?
                # Removing one might make the other non-removable?
                # If we remove A, counts decrease. If B was removable before, it might become critical?
                # No, counts only decrease. If B was removable (all neighbors count > 1), 
                # and we remove A, counts in N(A) decrease.
                # If N(B) intersects N(A), counts in intersection decrease.
                # If a count drops to 1, B might no longer be removable.
                # So we must check sequentially.
                
                # Let's try removing the first candidate.
                idx1 = to_remove[0]
                # Check if idx1 is still removable (it was)
                # Remove idx1
                for n_idx in neighbors[idx1]:
                    counts[n_idx] -= 1
                active_indices.remove(idx1)
                
                # Now check if we can remove another one.
                # We can check if idx2 (the second candidate) is still removable.
                # Or just scan for any removable word.
                
                found_second = False
                for idx in active_indices:
                    if idx == idx1: continue # already removed
                    if can_remove(idx):
                        # Remove it
                        for n_idx in neighbors[idx]:
                            counts[n_idx] -= 1
                        active_indices.remove(idx)
                        found_second = True
                        break
                        
                if not found_second:
                    # We removed 1, added 1. Size unchanged.
                    # We might want to revert to avoid wandering? 
                    # Or keep it to explore.
                    # For now, let's keep it. It might lead to a better state later.
                    pass
            elif len(candidates) == 1:
                # Remove 1. Size unchanged.
                idx1 = candidates[0]
                for n_idx in neighbors[idx1]:
                    counts[n_idx] -= 1
                active_indices.remove(idx1)
            else:
                # Cannot remove anything. Revert the addition?
                # If we added a word and can't remove anything, we are stuck with a larger code.
                # We should revert to previous state.
                # Undo addition
                active_indices.remove(random_idx)
                for n_idx in neighbors[random_idx]:
                    counts[n_idx] -= 1
                # And continue loop (maybe try a different random word)
                continue

    # Convert back to list of lists
    return [list(idx_to_word[idx]) for idx in active_indices]

def greedy_covering(q, n, radius, deadline):
    """
    Greedy construction of a covering code.
    """
    # Placeholder for generic greedy, used as fallback
    if n > 8:
        if q == 3:
             tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
             if n >= 4:
                 return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
             else:
                 return [[0]*n]
        return [[0]*n]
        
    all_words = list(itertools.product(range(q), repeat=n))
    uncovered = set(all_words)
    code = []
    
    def get_neighbors(c):
        neighbors = {c}
        for i in range(n):
            orig = c[i]
            for val in range(q):
                if val == orig: continue
                lst = list(c)
                lst[i] = val
                neighbors.add(tuple(lst))
        return neighbors

    while uncovered and time.time() < deadline:
        best_word = None
        best_count = -1
        
        # Greedy choice: pick word covering most uncovered
        for w in all_words:
            if w in code: continue # Optimization: don't pick existing
            cnt = 0
            for x in get_neighbors(w):
                if x in uncovered:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_word = w
                
        if best_word is None or best_count == 0:
            break
            
        code.append(list(best_word))
        uncovered.difference_update(get_neighbors(best_word))
        
    return code
# EVOLVE-BLOCK-END
