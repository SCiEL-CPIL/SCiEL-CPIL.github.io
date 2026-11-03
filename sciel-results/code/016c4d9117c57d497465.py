import time
import numpy as np
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin (150s wall-clock, leave buffer)

    # 1. Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    if n < 3:
        return "".join(map(str, word))

    # Precompute permutation to index mapping
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Build a fast lookup: for each position, which permutation windows cover it
    # Precompute window indices for the current word
    def get_window_indices(w):
        """Return list of permutation indices for each window start, or -1 if not a permutation."""
        L = len(w)
        indices = []
        for i in range(L - n + 1):
            p = tuple(w[i:i+n])
            idx = perm_to_idx.get(p, -1)
            indices.append(idx)
        return indices

    # Build counts from window indices
    def counts_from_indices(widx):
        c = [0] * num_perms
        for idx in widx:
            if idx >= 0:
                c[idx] += 1
        return c

    # Initialize
    widx = get_window_indices(word)
    counts = counts_from_indices(widx)

    # Helper to check if a specific window tuple is a valid permutation and get its index
    def try_delete(k, w, c, widx):
        """
        Simulates deletion of index k.
        Returns (is_valid, new_counts, new_widx)
        """
        L = len(w)
        # 1. Identify old windows containing k
        j_min = max(0, k - n + 1)
        j_max = min(L - n, k)
        
        valid_deletion = True
        temp_counts = c[:]
        
        for j in range(j_min, j_max + 1):
            idx = widx[j]
            if idx >= 0:
                temp_counts[idx] -= 1
                if temp_counts[idx] == 0:
                    valid_delation = False
                    break

        if not valid_deletion:
            return False, c, widx
        
        # 2. Identify new windows formed by deletion
        # After deleting position k, the word length is L-1
        # New windows start at positions 0 to L-1-n
        # Windows that are affected are those that "span" the deletion point
        # In the new word, position k is removed, so:
        #   new_word[j] = old_word[j] for j < k
        #   new_word[j] = old_word[j+1] for j >= k
        # A new window starting at j_prime in the new word corresponds to:
        #   if j_prime + n - 1 < k: same as old window at j_prime
        #   if j_prime >= k: same as old window at j_prime + 1
        #   if j_prime < k <= j_prime + n - 1: spans the deletion
        # Only the spanning windows are new
        
        j_prime_min = max(0, k - n + 1)
        j_prime_max = min(L - 1 - n, k - 1)
        
        for j_prime in range(j_prime_min, j_prime_max + 1):
            left_part = w[j_prime:k]
            right_part = w[k+1:j_prime+n+1]
            new_window_list = left_part + right_part
            p_new = tuple(new_window_list)
            idx = perm_to_idx.get(p_new)
            if idx is not None:
                temp_counts[idx] += 1
        
        if min(temp_counts) == 0:
            return False, c, widx
        
        # Build new widx
        new_widx = []
        for j in range(L - n):  # new length is L-1, so L-1-n+1 = L-n windows
            if j < j_prime_min:
                new_widx.append(widx[j])
            elif j > j_prime_max:
                new_widx.append(widx[j + 1])
            else:
                left_part = w[j:k]
                right_part = w[k+1:j+n+1]
                new_window_list = left_part + right_part
                p_new = tuple(new_window_list)
                idx = perm_to_idx.get(p_new, -1)
                new_widx.append(idx)
        
        return True, temp_counts, new_widx

    # Helper to check if swapping positions i and i+1 is valid
    def try_swap(i, w, c, widx):
        """
        Simulates swap of positions i and i+1.
        Returns (is_valid, new_counts, new_widx)
        """
        L = len(w)
        if i >= L - 1:
            return False, c, widx
        
        # Windows affected: those that include position i or i+1
        # Window start j includes pos i if j <= i <= j+n-1, i.e., j in [i-n+1, i]
        # Window start j includes pos i+1 if j in [i-n+2, i+1]
        # Union: j in [max(0, i-n+1), min(L-n, i+1)]
        
        j_min = max(0, i - n + 1)
        j_max = min(L - n, i + 1)
        
        temp_counts = c[:]
        
        # Decrement old windows
        for j in range(j_min, j_max + 1):
            idx = widx[j]
            if idx >= 0:
                temp_counts[idx] -= 1
                if temp_counts[idx] == 0:
                    return False, c, widx
        
        # Increment new windows
        for j in range(j_min, j_max + 1):
            # Build the new window
            window = list(w[j:j+n])
            if i >= j and i < j + n:
                # Position i is in this window
                window[i - j] = w[i+1]
            if i + 1 >= j and i + 1 < j + n:
                # Position i+1 is in this window
                window[i+1 - j] = w[i]
            p_new = tuple(window)
            idx = perm_to_idx.get(p_new)
            if idx is not None:
                temp_counts[idx] += 1
        
        if min(temp_counts) == 0:
            return False, c, widx
        
        # Build new widx
        new_widx = list(widx)
        for j in range(j_min, j_max + 1):
            window = list(w[j:j+n])
            if i >= j and i < j + n:
                window[i - j] = w[i+1]
            if i + 1 >= j and i + 1 < j + n:
                window[i+1 - j] = w[i]
            p_new = tuple(window)
            new_widx[j] = perm_to_idx.get(p_new, -1)
        
        return True, temp_counts, new_widx

    # Main optimization loop
    no_improve_rounds = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        L = len(word)
        if L < n:
            break
        
        improved = False
        
        # Phase 1: Try deletions (multiple passes)
        for pass_num in range(3):
            if time.time() > deadline - 5:
                break
            
            for k in range(L):
                if time.time() > deadline - 5:
                    break
                
                valid, new_counts, new_widx = try_delete(k, word, counts, widx)
                if valid:
                    word = word[:k] + word[k+1:]
                    counts = new_counts
                    widx = new_widx
                    L = len(word)
                    improved = True
                    break
            
            if not improved:
                break
            # If we made progress, reset no_improve counter
            no_improve_rounds = 0
        
        if not improved:
            # Phase 2: Try adjacent swaps
            swapped = False
            for i in range(L - 1):
                if time.time() > deadline - 5:
                    break
                
                valid, new_counts, new_widx = try_swap(i, word, counts, widx)
                if valid:
                    word[i], word[i+1] = word[i+1], word[i]
                    counts = new_counts
                    widx = new_widx
                    swapped = True
                    improved = True
                    no_improve_rounds = 0
                    break
            
            if not swapped:
                # Phase 3: Try non-adjacent swaps (i, i+d) for small d
                found_swap = False
                for d in range(2, min(5, L)):
                    if time.time() > deadline - 5:
                        break
                    for i in range(L - d):
                        if time.time() > deadline - 5:
                            break
                        
                        # Swap positions i and i+d
                        word[i], word[i+d] = word[i+d], word[i]
                        new_counts = counts_from_indices(get_window_indices(word))
                        
                        if min(new_counts) > 0:
                            counts = new_counts
                            widx = get_window_indices(word)
                            found_swap = True
                            improved = True
                            no_improve_rounds = 0
                            break
                        else:
                            # Revert
                            word[i], word[i+d] = word[i+d], word[i]
                    
                    if found_swap:
                        break
                
                if not found_swap:
                    no_improve_rounds += 1
                    if no_improve_rounds >= max_no_improve:
                        break

    return "".join(map(str, word))
