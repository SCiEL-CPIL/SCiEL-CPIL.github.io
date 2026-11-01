import time

def construct(n):
    start_time = time.time()
    # Use 140s to be safe against 150s limit
    deadline = start_time + 140.0
    
    # Step 1: Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of length m-1 in order of appearance
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Calculate overlap
            max_ov = min(len(new), len(block))
            overlap = 0
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    # Step 2: Greedy Deletion
    # We iteratively remove characters that are not critical.
    # A character is critical if it belongs to a window that is the unique
    # occurrence of that window's permutation.
    
    # If n is small, the standard construction is optimal, so deletion won't help.
    # For n=7, it might help.
    
    if n < 7:
        return "".join(map(str, word))
        
    while time.time() < deadline:
        changed = False
        L = len(word)
        
        # Build frequency map of all n-length windows
        # Since it's a superpermutation, all windows are distinct permutations?
        # No, windows can repeat if the word has repeated patterns, but in a superpermutation
        # of length ~ n!, every permutation appears at least once.
        # Some might appear multiple times.
        
        perm_counts = {}
        for i in range(L - n + 1):
            w = tuple(word[i:i+n])
            # Safety check: ensure it's a valid permutation of 1..n
            # This is fast for n=7
            if len(set(w)) == n:
                perm_counts[w] = perm_counts.get(w, 0) + 1
        
        # Attempt to remove each character
        for i in range(L):
            if time.time() > deadline:
                break
                
            # Check if removing word[i] is safe
            # word[i] is part of windows starting at j where j <= i < j+n
            # j ranges from max(0, i-n+1) to min(i, L-n)
            
            safe = True
            start_j = max(0, i - n + 1)
            end_j = min(i, L - n)
            
            for j in range(start_j, end_j + 1):
                w_j = tuple(word[j:j+n])
                # If this is the only occurrence of this permutation, we cannot remove this char
                if perm_counts[w_j] == 1:
                    safe = False
                    break
            
            if safe:
                word.pop(i)
                changed = True
                # Restart the scan from the beginning (or from i) 
                # Restarting from beginning is safer and simple given low L
                break
                
        if not changed:
            break
            
    return "".join(map(str, word))
