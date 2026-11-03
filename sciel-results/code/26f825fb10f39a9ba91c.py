import itertools
import random
import time

def construct(n):
    # Step 1: Generate standard superpermutation
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
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # If n is small, standard is optimal or close.
    # For n=7, we apply minimization.
    
    if n <= 4:
        return "".join(map(str, word))
        
    # Minimization Phase
    # We represent the word as a list of ints.
    # We need to track the count of each permutation of length n.
    
    def get_permutation_counts(w, n):
        counts = {}
        for i in range(len(w) - n + 1):
            p = tuple(w[i:i+n])
            counts[p] = counts.get(p, 0) + 1
        return counts

    def is_valid_counts(counts, n):
        # Check if all n! permutations have count > 0
        # We can just check if any count is 0 for a valid permutation
        # But we only track counts for existing windows.
        # We need to ensure all possible permutations are present.
        # Total permutations = n!
        # If sum of counts == len(w) - n + 1 and len(counts) == n!, it's valid.
        # Actually, just checking len(counts) == factorial(n) is enough if we know no duplicates in keys.
        pass

    # Efficient check:
    # We maintain a dict of counts.
    # If we remove a char, we update counts for windows affected.
    # If any count becomes 0, and that permutation is required (it is, if it was there), it's invalid.
    # Wait, if a permutation count becomes 0, it means we lost that permutation.
    # So we just need to ensure no count drops to 0.
    
    current_word = word[:]
    counts = get_permutation_counts(current_word, n)
    
    # To speed up, we can pre-compute which windows are affected by removing index i.
    # Window j covers indices j .. j+n-1.
    # Index i is covered by windows j such that j <= i <= j+n-1.
    # i-n+1 <= j <= i.
    # Also j must be valid: 0 <= j <= len(current_word) - n.
    
    deadline = time.time() + 110 # Leave 10s margin
    
    # We try to remove characters.
    # We do multiple passes.
    # To avoid getting stuck, we can randomize the order of indices to try.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Shuffle indices to try different orders? 
        # Or just left to right.
        # Left to right is deterministic. 
        # Let's try left to right first.
        indices = list(range(len(current_word)))
        # random.shuffle(indices) # Maybe try this in a restart loop?
        
        for i in indices:
            if time.time() > deadline:
                break
                
            # Windows affected by removing index i
            # j_start = max(0, i - n + 1)
            # j_end = min(i, len(current_word) - n)
            
            j_start = max(0, i - n + 1)
            j_end = min(i, len(current_word) - n)
            
            valid_removal = True
            affected_perms = []
            
            for j in range(j_start, j_end + 1):
                p = tuple(current_word[j:j+n])
                if counts[p] == 1:
                    valid_removal = False
                    break
                affected_perms.append((j, p))
            
            if valid_removal:
                # Perform removal
                for j, p in affected_perms:
                    counts[p] -= 1
                current_word.pop(i)
                improved = True
                # After removal, indices shift. 
                # If we are iterating a fixed list of indices, we should be careful.
                # If we pop i, the element at i+1 becomes i.
                # So we can just continue. The next index in our list might be invalid or skipped.
                # To be safe, we can restart the pass or adjust.
                # Given the small number of removals expected, restarting the pass is cheap.
                break # Restart pass to be safe and simple
        
    return "".join(map(str, current_word))
