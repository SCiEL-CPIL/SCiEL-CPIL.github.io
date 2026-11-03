import time
from itertools import permutations

def generate_superperm(n):
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = 0
            max_ov = min(len(new_word), len(block))
            for o in range(max_ov, -1, -1):
                if new_word[len(new_word) - o:] == block[:o]:
                    overlap = o
                    break
            new_word += block[overlap:]
        word = new_word
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    if n <= 1:
        return "1"
        
    # 1. Generate standard superpermutation
    word = generate_superperm(n)
    
    # 2. Greedy Deletion Optimization
    # Use integer hashing for speed
    # Base is n+1 to safely encode digits 1..n
    base = n + 1
    
    def encode(window):
        h = 0
        for d in window:
            h = h * base + d
        return h

    L = len(word)
    
    # Initialize counts
    perm_counts = {}
    for i in range(L - n + 1):
        w = word[i:i+n]
        # Check if it's a valid permutation (all distinct)
        if len(set(w)) == n:
            h = encode(w)
            if h in perm_counts:
                perm_counts[h] += 1
            else:
                perm_counts[h] = 1
                
    # We need to be able to quickly check if a window is a valid permutation.
    # Since we only delete, and we start with a valid superperm,
    # we can assume that if a hash is in perm_counts, it was valid.
    # But after deletion, new windows are formed.
    # We must verify validity of new windows or maintain a set of all valid hashes?
    # No, new windows might not be valid permutations.
    # So we need to check validity.
    # However, checking validity (len(set(w))==n) is O(n).
    # We can optimize the deletion check.
    
    deleted_any = True
    while deleted_any and time.time() < deadline:
        deleted_any = False
        L = len(word)
        
        for i in range(L):
            if time.time() > deadline - 5:
                break
                
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            
            safe = True
            # We need to check if any window containing index i has count 1.
            # These windows must be valid permutations.
            
            for j in range(start_min, start_max + 1):
                w = word[j:j+n]
                # Quick check: is this a valid permutation?
                # We can check if the hash exists in perm_counts.
                # If it doesn't, it's not a valid permutation (count 0), so it's safe to "delete" from count perspective (it's not counted).
                # But wait, if it's not a valid permutation, it's not in perm_counts.
                # If it IS in perm_counts and count is 1, it's unsafe.
                h = encode(w)
                if h in perm_counts and perm_counts[h] == 1:
                    safe = False
                    break
            
            if safe:
                # Perform deletion
                for j in range(start_min, start_max + 1):
                    w = word[j:j+n]
                    h = encode(w)
                    if h in perm_counts:
                        perm_counts[h] -= 1
                        if perm_counts[h] == 0:
                            del perm_counts[h]
                
                word = word[:i] + word[i+1:]
                deleted_any = True
                break # Restart the loop from i=0
                
    return "".join(map(str, word))
