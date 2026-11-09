import time
from itertools import permutations
from collections import defaultdict

DEADLINE = time.time() + 140.0

def _build_recursive(n):
    """Classic recursive superpermutation of length 1!+2!+...+n!."""
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
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def construct(n):
    if n <= 1:
        return "1"
    
    start = time.time()
    word = _build_recursive(n)
    
    # Precompute all permutations and their hashes
    all_perms = [tuple(p) for p in permutations(range(1, n + 1))]
    perm_hashes = set()
    for p in all_perms:
        perm_hashes.add(p) # Using tuple as key directly is fine and fast enough for n=7
    
    # Initialize counts
    counts = defaultdict(int)
    for i in range(len(word) - n + 1):
        counts[tuple(word[i:i+n])] += 1
        
    # Local Search: Greedy Deletion
    # We try to remove characters one by one.
    # To avoid getting stuck, we can try multiple passes or randomize.
    # Given the small gap, a simple left-to-right greedy pass repeated until no improvement
    # might be sufficient. If not, we can try right-to-left or random.
    
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        # Try deletions from left to right
        for i in range(len(word) - 1):
            if time.time() > DEADLINE:
                break
            
            # Identify windows in current word containing word[i]
            # Window j contains i if j <= i <= j+n-1
            # So j in [i-n+1, i]
            start_j = max(0, i - n + 1)
            end_j = min(i, len(word) - n)
            
            affected = []
            for j in range(start_j, end_j + 1):
                affected.append(tuple(word[j:j+n]))
            
            # Calculate new mixed windows in the word after deletion
            # New word length is L-1
            # Deletion at i.
            # New windows crossing the gap are those starting at j_new in [i-n+1, i] (in new word coords)
            # But we need to map to original word indices to get the content.
            # j_new is the start index in the NEW word.
            # The gap in the new word is at index i (since chars before i are same, chars after shift).
            # Actually, the "hole" is at i.
            # Windows in new word that cover the hole:
            # Start j_new such that j_new <= i-1 and j_new+n-1 >= i.
            # So j_new in [max(0, i-n+1), min(i-1, len(word)-1-n)]?
            # Wait, if j_new = i, the window is w'[i...i+n-1] which is w[i+1...i+n]. This is an old window shifted.
            # It doesn't cross the "hole" in the sense of mixing old left and old right.
            # The "mixed" windows are those that take some chars from left of i and some from right of i.
            # In the new word, the chars at indices < i are from w[0...i-1].
            # The chars at indices >= i are from w[i+1...].
            # A window w'[j_new ... j_new+n-1] is mixed if it includes indices < i and >= i.
            # So j_new < i and j_new+n-1 >= i.
            # j_new ranges from max(0, i-n+1) to min(i-1, len(word)-1-n).
            
            new_mixed = []
            max_j_new = min(i - 1, len(word) - 1 - n)
            min_j_new = max(0, i - n + 1)
            
            for j_new in range(min_j_new, max_j_new + 1):
                # Construct the window content
                # Part 1: w[j_new ... i-1]
                # Part 2: w[i+1 ... j_new+n]
                # Note: w'[j_new+k] = w[j_new+k] for j_new+k < i
                #       w'[j_new+k] = w[j_new+k+1] for j_new+k >= i
                # The window ends at j_new+n-1 in new word.
                # Since it crosses i, j_new+n-1 >= i.
                # The number of chars from left is i - j_new.
                # The number of chars from right is n - (i - j_new) = n - i + j_new.
                # These come from w[i+1 ... i + n - i + j_new] = w[i+1 ... j_new+n].
                left_part = word[j_new:i]
                right_part = word[i+1:j_new+n+1]
                new_window = left_part + right_part
                new_mixed.append(tuple(new_window))
            
            # Check if deletion is valid
            # Temporarily update counts
            # We need to ensure no permutation count drops to 0.
            # We can just check the affected and new_mixed.
            # If a hash is in affected, its count decreases.
            # If a hash is in new_mixed, its count increases.
            # Net change:
            # For h in affected: count -= 1
            # For h in new_mixed: count += 1
            
            # Optimization: Only check hashes that are in perm_hashes.
            # And only check if count becomes 0.
            
            valid = True
            # We need to be careful if a hash appears in both affected and new_mixed.
            # If h in affected and h in new_mixed, net change is 0.
            # So we only need to check hashes in affected that are NOT in new_mixed (or have net negative change).
            
            # Let's just use a temporary dict for the relevant hashes to be safe and simple.
            # The number of affected/new_mixed windows is small (<= n).
            temp_counts = {}
            for h in affected:
                if h in perm_hashes:
                    temp_counts[h] = counts.get(h, 0)
            
            for h in affected:
                if h in temp_counts:
                    temp_counts[h] -= 1
            
            for h in new_mixed:
                if h in perm_hashes:
                    temp_counts[h] = temp_counts.get(h, 0) + 1
            
            for h, c in temp_counts.items():
                if c <= 0:
                    valid = False
                    break
            
            if valid:
                # Commit deletion
                # Update global counts
                for h in affected:
                    if h in counts:
                        counts[h] -= 1
                        if counts[h] == 0:
                            del counts[h]
                for h in new_mixed:
                    counts[h] = counts.get(h, 0) + 1
                
                word = word[:i] + word[i+1:]
                improved = True
                break # Restart scan
    
    return "".join(map(str, word))
