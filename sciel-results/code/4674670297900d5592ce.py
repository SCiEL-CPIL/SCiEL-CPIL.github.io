import time
from collections import Counter

def construct(n):
    start_time = time.time()
    deadline = 100.0  # Safety margin
    
    if n <= 1:
        return "1" if n == 1 else ""
        
    # 1. Generate standard superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of length m-1 from current word
        # The seed code does this by scanning windows of length m-1
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # 2. Greedy Deletion
    # Convert to list of ints for easier manipulation if needed, but word is already list of ints
    seq = word
    L = len(seq)
    
    # Initialize counts
    counts = Counter()
    for i in range(L - n + 1):
        w = tuple(seq[i:i+n])
        counts[w] += 1
        
    improved = True
    while improved:
        if time.time() - start_time > deadline:
            break
        improved = False
        i = 0
        while i < len(seq):
            if time.time() - start_time > deadline:
                break
            
            # Check if deletion of seq[i] is valid
            # Windows containing index i
            valid = True
            affected_windows = []
            start_min = max(0, i - n + 1)
            start_max = min(len(seq) - n, i)
            
            for start in range(start_min, start_max + 1):
                w = tuple(seq[start:start+n])
                affected_windows.append(w)
                if counts[w] == 1:
                    valid = False
                    break
            
            if valid:
                # Perform deletion
                for w in affected_windows:
                    counts[w] -= 1
                    if counts[w] == 0:
                        del counts[w]
                
                # Remove element
                seq.pop(i)
                
                # Update counts for new windows formed by merge
                # New windows are those starting in [max(0, i-n+1), min(len(seq)-n, i)]
                new_start_min = max(0, i - n + 1)
                new_start_max = min(len(seq) - n, i)
                
                for start in range(new_start_min, new_start_max + 1):
                    w = tuple(seq[start:start+n])
                    counts[w] += 1
                
                improved = True
                # Do not increment i
            else:
                i += 1
                
    return "".join(map(str, seq))
