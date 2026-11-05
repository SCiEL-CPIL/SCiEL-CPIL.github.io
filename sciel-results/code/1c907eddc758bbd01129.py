import sys
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    if n <= 3:
        # Standard construction is optimal for n<=3
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
                    if o == 0 or new[len(new) - o:] == block[:o]:
                        overlap = o
                        break
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
    
    # For n >= 4, use standard recursive construction
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
            max_possible = min(len(new), len(block))
            overlap = 0
            for o in range(max_possible, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    L = len(word)
    
    # Greedy deletion pass
    # To speed up, we can maintain counts of permutations.
    # But for n=7, L=5911, n=7.
    # Number of windows is ~5905.
    # We can just re-calculate counts or update them incrementally.
    # Incremental is better.
    
    counts = {}
    # Initialize counts
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        if len(set(w)) == n:
            counts[w] = counts.get(w, 0) + 1
    
    # If not all permutations are present, something is wrong (shouldn't happen)
    # We assume the standard construction is valid.
    
    changed = True
    # We scan from right to left to allow chaining deletions?
    # Actually, deleting a char shifts indices.
    # It's safer to scan left to right and delete, or right to left.
    # Right to left is standard for deletion to avoid index shifting issues for the next check?
    # If we delete at i, indices > i shift left. Indices < i are unchanged.
    # So scanning right to left (from L-1 down to 0) means when we delete at i,
    # we have already processed i+1...L-1. The new element at i+1 (which was i+2)
    # will be checked in the next iteration? No, the loop decrements i.
    # So we might miss the new window formed at i+1?
    # Actually, if we delete word[i], the window starting at i-1 changes, and window starting at i changes (shifts).
    # Let's just do multiple passes until no change.
    
    while changed and time.time() < deadline:
        changed = False
        # Re-initialize counts if we did deletions in previous pass?
        # No, we updated them incrementally.
        
        # Scan from right to left
        for i in range(L - 1, -1, -1):
            # Check if word[i] is in any "critical" window (count == 1)
            # A window j contains i if j <= i < j+n
            # j ranges from max(0, i-n+1) to min(i, L-n)
            
            valid = True
            affected = []
            
            start_j = max(0, i - n + 1)
            end_j = min(i, L - n)
            
            for j in range(start_j, end_j + 1):
                w = tuple(word[j:j+n])
                if len(set(w)) == n:
                    if counts.get(w, 0) == 1:
                        valid = False
                        break
                    else:
                        affected.append(w)
            
            if valid:
                # Delete word[i]
                # Update counts
                for w in affected:
                    counts[w] -= 1
                    if counts[w] == 0:
                        del counts[w]
                
                word.pop(i)
                L -= 1
                changed = True
                
    return "".join(map(str, word))
