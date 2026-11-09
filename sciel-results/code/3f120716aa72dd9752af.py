# EVOLVE-BLOCK-START
import time
import itertools
import random


def construct(q, n, radius):
    # Task: q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Ball size: 1 + 7*2 = 15
    
    # Fallback for other parameters (not expected in this specific evaluation)
    if q != 3 or n != 7 or radius != 1:
        tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            return [list(word) for word in itertools.product(range(q), repeat=n)]

    total_words = q ** n
    deadline = time.time() + 110.0  # Safety margin before 120s limit

    # Precompute powers of q for index conversion
    q_pows = [1]
    for i in range(n):
        q_pows.append(q_pows[-1] * q)
    # q_pows[i] = q^i

    # Precompute balls as list of indices and bitmask
    ball_indices = [[] for _ in range(total_words)]
    ball_masks = [0] * total_words
    
    for i in range(total_words):
        neighbors = [i]
        mask = 1 << i
        # For each position
        for pos in range(n):
            current_digit = (i // q_pows[pos]) % q
            for new_digit in range(q):
                if new_digit == current_digit:
                    continue
                diff = new_digit - current_digit
                new_idx = i + diff * q_pows[pos]
                neighbors.append(new_idx)
                mask |= (1 << new_idx)
        ball_indices[i] = neighbors
        ball_masks[i] = mask

    full_mask = (1 << total_words) - 1

    # Convert an index to a list of digits
    def idx_to_word(idx):
        word = []
        temp = idx
        for _ in range(n):
            word.append(temp % q)
            temp //= q
        return word  # Note: word[0] is LSB, word[n-1] is MSB

    best_code = None
    best_size = float('inf')

    # Use the seed solution as a starting point
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    seed_words = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    # Convert seed words to indices
    seed_indices = []
    for w in seed_words:
        # w is in MSB-first order (word[0] is most significant)
        # But idx_to_word gives LSB-first
        # So we need to reverse
        idx = 0
        for k in range(n):
            idx += w[k] * (q_pows[n - 1 - k])
        seed_indices.append(idx)
    
    # Helper to minimize a code using redundancy removal
    def minimize_code(code):
        # code is a list of indices
        # Initialize coverage counts
        counts = [0] * total_words
        for c in code:
            for w in ball_indices[c]:
                counts[w] += 1
        
        # Iterative removal
        changed = True
        while changed:
            changed = False
            # Iterate in reverse to safely remove
            for i in range(len(code) - 1, -1, -1):
                c = code[i]
                # Check if c is essential
                essential = False
                for w in ball_indices[c]:
                    if counts[w] == 1:
                        essential = True
                        break
                
                if not essential:
                    # Remove c
                    code.pop(i)
                    changed = True
                    # Update counts
                    for w in ball_indices[c]:
                        counts[w] -= 1
                    break # Restart the loop from the end? Or just continue? 
                    # Breaking and restarting ensures we don't miss opportunities 
                    # caused by the shift in indices, but it's slower.
                    # However, since we update counts immediately, continuing 
                    # to the next i (which is now a different element) is safe 
                    # because we are moving backwards.
                    # Actually, if we pop i, the element at i is now the one 
                    # that was at i+1. If we don't break, the loop moves to i-1.
                    # This is fine. We don't need to break.
        
        return code

    # Run greedy with minimization
    candidates = list(range(total_words))
    
    while time.time() < deadline:
        uncovered = full_mask
        code = []
        
        while uncovered:
            max_count = -1
            best_candidates = []
            
            # Find the word that covers the most uncovered points
            for c in candidates:
                count = (ball_masks[c] & uncovered).bit_count()
                if count > max_count:
                    max_count = count
                    best_candidates = [c]
                elif count == max_count:
                    best_candidates.append(c)
            
            # Pick one randomly from the best candidates
            if best_candidates:
                chosen = random.choice(best_candidates)
            else:
                break  # Should not happen
            
            code.append(chosen)
            uncovered &= ~ball_masks[chosen]
            
        # Minimize the greedy code
        min_code = minimize_code(code)
        
        if len(min_code) < best_size:
            best_size = len(min_code)
            best_code = min_code[:]
            
            # If we reach the known best (186), we can stop or continue.
            if best_size <= 186 and time.time() > deadline - 10:
                break

    # If we didn't find a better code than the seed, check the seed too?
    # The seed is 243. Greedy usually finds ~200. Minimized ~190-200.
    # Let's ensure we at least return the seed if something goes wrong, 
    # but best_code should be set in the loop.
    
    if best_code is None:
        best_code = seed_indices[:]

    # Convert best_code indices to words
    result = []
    for idx in best_code:
        result.append(idx_to_word(idx))
    
    return result
# EVOLVE-BLOCK-END
