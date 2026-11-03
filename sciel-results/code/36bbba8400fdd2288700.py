import time
import sys
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We aim to finish well before.
    deadline = start_time + 140.0
    
    # 1. Recursive Construction
    word = [1]
    
    for m in range(2, n + 1):
        # Extract unique permutations of length m-1 from word
        # in order of appearance
        perms = []
        seen = set()
        L = len(word)
        for i in range(L - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # If this is the last step (m == n) and n > 4, try to optimize the order
        # of permutations to maximize overlap.
        if m == n and n > 4:
            # We have a list of perms (tuples of length n-1)
            # We want to order them to maximize sum of overlaps.
            
            def calc_overlap(u, v):
                max_k = min(len(u), len(v))
                for k in range(max_k, 0, -1):
                    if u[-k:] == v[:k]:
                        return k
                return 0

            base_order = perms
            base_score = 0
            for i in range(len(base_order) - 1):
                base_score += calc_overlap(base_order[i], base_order[i+1])
                
            # Try to improve with local search
            current_order = list(base_order)
            current_score = base_score
            improved = True
            # Limit iterations to ensure we finish in time
            max_iterations = 500 
            iter_count = 0
            
            while improved and time.time() < deadline - 10.0 and iter_count < max_iterations:
                improved = False
                iter_count += 1
                
                # Random 2-opt: pick a segment and reverse it
                if len(current_order) < 2:
                    break
                
                i = random.randint(0, len(current_order) - 2)
                j = random.randint(i + 1, len(current_order) - 1)
                
                # Calculate score change for reversing current_order[i:j+1]
                # We recalculate the total score for simplicity as N is small (720 for n=7)
                new_order = current_order[:i] + current_order[i:j+1][::-1] + current_order[j+1:]
                
                score = 0
                for k in range(len(new_order) - 1):
                    score += calc_overlap(new_order[k], new_order[k+1])
                    
                if score > current_score:
                    current_score = score
                    current_order = new_order
                    improved = True
                    
            if current_score > base_score:
                perms = current_order
            else:
                perms = base_order
        else:
            # For smaller m, just use the standard order
            pass

        # Construct new word
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, -1, -1):
                if o == 0:
                    break
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word.extend(block[max_overlap:])
        
        word = new_word
        
        if time.time() > deadline:
            break
    
    # If n is small, just return the constructed word
    if n <= 4:
        return "".join(map(str, word))

    # 2. Polishing Phase: Character Deletion
    # This phase attempts to shorten the word by removing redundant characters.
    
    # Map permutations to IDs for fast counting
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    counts = [0] * num_perms
    L = len(word)
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        if p in perm_to_id:
            counts[perm_to_id[p]] += 1
            
    # Check if valid initially (should be)
    if any(c == 0 for c in counts):
        return "".join(map(str, word))

    # Helper to check if a specific deletion at index i is valid
    # Returns True if deletion is valid, and modifies counts if we want to commit (but we'll revert for simplicity in loop)
    # Actually, to be efficient, we will modify counts, check, and revert if invalid.
    
    def is_deletion_valid(i, current_word, current_counts):
        L = len(current_word)
        if L <= n:
            return False
            
        # Old windows containing i: start k in [i-n+1, i]
        # Range of valid window starts in old word: [0, L-n]
        old_start_min = max(0, i - n + 1)
        old_start_max = min(i, L - n)
        
        # New windows formed:
        # When we delete i, the word becomes W_new.
        # Windows in W_new starting at k < i:
        #   If k+n <= i: window is same as old W_old(k). No change.
        #   If k+n > i: window is W_old(k) with i removed. 
        #       This is effectively a new window.
        #       The start index in W_new is k.
        #       The content is W_old[k:i] + W_old[i+1:k+n+1].
        # Windows in W_new starting at k >= i:
        #   W_new[k] = W_old[k+1].
        #   So W_new[k:k+n] = W_old[k+1:k+1+n].
        #   These are just shifted old windows. Their counts are preserved in the set of windows.
        #   However, we are maintaining a global count of *how many times* each permutation appears.
        #   The set of windows changes.
        #   
        #   Let's trace the change in counts:
        #   1. Remove counts for old windows that are gone or changed.
        #      - Windows starting at k in [i-n+1, i] that contain i are "destroyed" or "changed".
        #        Actually, windows starting at k < i-n+1 are unaffected.
        #        Windows starting at k > i are shifted, so the *multiset* of windows starting at k>=i in new word 
        #        is identical to multiset of windows starting at k+1 >= i+1 in old word.
        #        So, the counts for windows starting at k >= i in the new word are exactly the counts 
        #        for windows starting at k+1 >= i+1 in the old word.
        #        
        #   Therefore, the only windows that require recalculation of counts are those that "cross" the deletion point i.
        #   
        #   Old windows affected: Starts k in [i-n+1, i]. (These are the ones that contain i).
        #   New windows affected: Starts k in [i-n+1, i-1]. (These are the ones that cross the gap in the new word).
        #   Note: The new window at start i is just the old window at start i+1.
        #   
        #   So:
        #   1. Decrement counts for old windows starting at k in [i-n+1, i].
        #   2. Increment counts for new windows starting at k in [i-n+1, i-1].
        #      (Note: k=i in new word corresponds to old k=i+1, which was not decremented, so it stays consistent? 
        #       Wait. The old window at i+1 is still present in the new word (at index i). 
        #       Did we decrement it? No, we only decremented up to i. 
        #       So the count for the window content at old i+1 is preserved. 
        #       The new window at index i has the same content as old i+1. 
        #       So we don't need to touch it.)
        
        # Let's refine the ranges:
        # Old windows to remove: k such that k <= i <= k+n-1 => k in [i-n+1, i].
        # Valid old k: [0, L-n].
        # So old_k_range = [max(0, i-n+1), min(i, L-n)]
        
        # New windows to add: k in new word such that the window crosses the deletion site.
        # In new word, deletion site was i. The gap is between i-1 and i (new indices).
        # Windows starting at k in new word that include the gap:
        # k <= i-1 and k+n-1 >= i => k in [i-n+1, i-1].
        # Valid new k: [0, L-1-n] = [0, L-n-1].
        # So new_k_range = [max(0, i-n+1), min(i-1, L-n-1)]
        
        changes = []
        
        # 1. Decrement old
        for k in range(old_start_min, old_start_max + 1):
            p = tuple(current_word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                current_counts[pid] -= 1
                changes.append((pid, -1))
                if current_counts[pid] < 0:
                    # Should not happen if logic is correct, but revert just in case
                    for cp, delta in changes:
                        current_counts[cp] -= delta
                    return False

        # 2. Increment new
        new_start_min = max(0, i - n + 1)
        new_start_max = min(i - 1, L - n - 1)
        
        for k in range(new_start_min, new_start_max + 1):
            # Construct the new window content
            # New word is current_word with i removed.
            # Window at k in new word:
            # If k < i:
            #   It consists of current_word[k:i] + current_word[i+1:k+n+1]
            #   Length is (i-k) + (k+n+1 - (i+1)) = i-k + n-k-i = n-k. Wait.
            #   Let's just slice the virtual new word.
            #   new_word = current_word[:i] + current_word[i+1:]
            #   We don't want to create the full string.
            #   new_window = tuple(current_word[k:i] + current_word[i+1:k+n+1])
            
            w1 = current_word[k:i]
            w2 = current_word[i+1:k+n+1]
            p = tuple(w1 + w2)
            pid = perm_to_id.get(p)
            if pid is not None:
                current_counts[pid] += 1
                changes.append((pid, 1))

        # Check if any count is 0
        # We only need to check the pids that were modified?
        # No, we need to ensure NO pid is 0.
        # But we started with all > 0.
        # We only decremented some.
        # So we just need to check if any of the decremented pids became 0.
        
        valid = True
        for pid, delta in changes:
            if delta == -1: # Only check decremented ones
                if current_counts[pid] == 0:
                    valid = False
                    break
        
        if not valid:
            # Revert
            for pid, delta in changes:
                current_counts[pid] -= delta
            return False
        
        return True

    # Perform deletion
    # We iterate through the word. If we delete index i, the next index to check is still i (which is now the next char).
    # If we don't delete, we move to i+1.
    
    i = 0
    while i < len(word):
        if time.time() > deadline - 5.0:
            break
            
        if is_deletion_valid(i, word, counts):
            # Commit deletion
            # We need to actually remove it from 'word'
            word.pop(i)
            # 'counts' are already updated by is_deletion_valid (since it returned True, it didn't revert)
            # Note: is_deletion_valid modified 'counts' in place.
            # We do NOT increment i, because the next character shifted into position i.
        else:
            i += 1
            
    return "".join(map(str, word))

# Keep the function name and signature as required
# The task says: Interface: construct(n) -> a word ...
# The current program defines construct(n).
