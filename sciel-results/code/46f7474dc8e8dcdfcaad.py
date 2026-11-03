import time
import random
import itertools

def construct(q, n, radius):
    # Handle the specific case K_3(7,1)
    if q == 3 and n == 7 and radius == 1:
        return _solve_k3_7_1()
    
    # Fallback for other parameters
    # If n is small, we can potentially do better, but for safety and generality:
    # Return a simple valid code.
    # For q=3, n>=4, use tetracode product
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generic fallback: return all words (always valid, but large)
    # This is only for cases where we don't have a specific strategy
    all_words = []
    for w in itertools.product(range(q), repeat=n):
        all_words.append(list(w))
    return all_words


def _solve_k3_7_1():
    q = 3
    n = 7
    radius = 1
    total_words = q ** n  # 2187
    
    # Map word (tuple) to index (int)
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return tuple(w)

    # Precompute neighbors (indices of words within distance <= 1)
    neighbors = [None] * total_words
    for i in range(total_words):
        w = idx_to_word(i)
        neigh = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                neigh.append(word_to_idx(tuple(new_w)))
        neighbors[i] = neigh

    # Precompute coverage masks
    # cover_masks[i] is a bitmask where bit j is 1 if word j is covered by word i
    cover_masks = [0] * total_words
    for i in range(total_words):
        mask = 0
        for j in neighbors[i]:
            mask |= (1 << j)
        cover_masks[i] = mask

    # Greedy Construction
    U = (1 << total_words) - 1
    C = []
    counts = [0] * total_words
    
    # To speed up greedy, we can use a simple loop.
    # 2187 candidates, bit_count is fast.
    while U != 0:
        best_c = -1
        max_covered = 0
        # Optimization: We can skip words that are already in C? 
        # Actually, adding a word already in C is useless. 
        # But in the first phase, C is empty. 
        # As C grows, we should avoid re-adding.
        # However, checking `if c in C` is O(|C|). 
        # Since we update U, if a word was added, its coverage is already removed from U.
        # So `cover_masks[c] & U` for c in C will be 0 (or small if there's overlap not yet removed? No, U is updated).
        # So we don't strictly need to skip, but it saves bit_count calls.
        # Let's just iterate all.
        for c in range(total_words):
            covered = (cover_masks[c] & U).bit_count()
            if covered > max_covered:
                max_covered = covered
                best_c = c
        
        if best_c == -1:
            break # Should not happen if U != 0
            
        C.append(best_c)
        mask = cover_masks[best_c]
        U &= ~mask
        for j in neighbors[best_c]:
            counts[j] += 1

    # Local Search
    start_time = time.time()
    time_limit = 110.0 # Leave 10s margin for the rest of the program/evaluation overhead
    
    best_C = C[:]
    best_size = len(C)
    
    # Helper to check if a word can be removed
    # A word c can be removed if for all j in neighbors[c], counts[j] > 1
    # (since counts[j] includes c's contribution)
    
    while time.time() - start_time < time_limit:
        # 1. Try to remove redundant words
        # Iterate in random order to avoid bias
        perm = C[:]
        random.shuffle(perm)
        removed_any = False
        
        for c in perm:
            if time.time() - start_time > time_limit:
                break
            removable = True
            for j in neighbors[c]:
                if counts[j] == 1:
                    removable = False
                    break
            
            if removable:
                # Remove c
                C.remove(c)
                for j in neighbors[c]:
                    counts[j] -= 1
                removed_any = True
                if len(C) < best_size:
                    best_size = len(C)
                    best_C = C[:]
                # Break to restart shuffle with new state
                break
        
        if removed_any:
            continue # Restart the loop to try more removals
        
        # 2. If no removal, try swaps
        # Pick a random word c in C to try replacing
        if C:
            c = random.choice(C)
            
            # Temporarily remove c from counts
            for j in neighbors[c]:
                counts[j] -= 1
            
            # Find uncovered words U_temp
            U_temp = 0
            for j in neighbors[c]:
                if counts[j] == 0:
                    U_temp |= (1 << j)
            
            if U_temp == 0:
                # c was redundant, but we should have caught it in removal phase.
                # This can happen if counts were updated in a way that made it redundant
                # but the removal phase didn't pick it up (e.g. if we just did a swap).
                # In that case, just keep it removed.
                C.remove(c)
                if len(C) < best_size:
                    best_size = len(C)
                    best_C = C[:]
            else:
                # We need to find a replacement c' not in C
                # such that cover_masks[c'] covers U_temp
                found = False
                # Candidate pool: All words not in C
                # To speed up, we can check words that cover at least one word in U_temp
                # But checking all 2187 is also fast enough (2187 * bit ops)
                
                # We must ensure c' is not in C
                # Converting C to a set for O(1) lookup
                C_set = set(C)
                
                # Try to find a replacement
                # We can prioritize words that cover more of U_temp?
                # Or just iterate.
                # Let's iterate through all indices
                for c_prime in range(total_words):
                    if c_prime in C_set:
                        continue
                    if (cover_masks[c_prime] & U_temp) == U_temp:
                        # Valid replacement
                        C.remove(c)
                        C.append(c_prime)
                        C_set.add(c_prime)
                        # Update counts
                        for j in neighbors[c_prime]:
                            counts[j] += 1
                        found = True
                        break
                
                if not found:
                    # Backtrack: restore counts for c
                    for j in neighbors[c]:
                        counts[j] += 1
                    # Note: c is still in C

    # Convert best_C indices back to words
    result = [list(idx_to_word(idx)) for idx in best_C]
    return result
