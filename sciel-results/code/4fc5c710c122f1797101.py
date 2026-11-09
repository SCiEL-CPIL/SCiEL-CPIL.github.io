import time
from itertools import permutations

# Deadline guard: The evaluator kills at 150s. We aim to finish in < 10s.
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
            # Find maximum overlap
            max_overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    max_overlap = o
                    break
            new += block[max_overlap:]
        word = new
    return word

def _compute_hash_counts_and_arrays(word, n):
    """
    Compute hash counts and per-window hash array in one pass.
    Returns (counts_dict, window_hashes_list) where window_hashes_list[j]
    is the hash of word[j:j+n].
    """
    L = len(word)
    if L < n:
        return {}, []
    base = n + 1
    powb = base ** (n - 1)
    counts = {}
    window_hashes = []
    
    # Initial hash
    h = 0
    for x in word[:n]:
        h = h * base + x
    window_hashes.append(h)
    counts[h] = counts.get(h, 0) + 1
    
    # Roll
    for i in range(n, L):
        h = (h - word[i - n] * powb) * base + word[i]
        window_hashes.append(h)
        counts[h] = counts.get(h, 0) + 1
        
    return counts, window_hashes

def construct(n):
    if n <= 1:
        return "1"
    
    start_time = time.time()
    word = _build_recursive(n)
    
    # For small n, the recursive construction is optimal.
    if n <= 5:
        return "".join(map(str, word))
    
    # Main optimization loop
    # We repeatedly try to find a deletable character.
    # Precompute window hashes to make each deletion check O(1) per window.
    
    while time.time() < DEADLINE:
        counts, window_hashes = _compute_hash_counts_and_arrays(word, n)
        found_deletion = False
        
        # Scan left to right for a deletable index
        for i in range(len(word)):
            if time.time() > DEADLINE:
                break
            
            # Identify windows containing index i
            # Window j contains i if j <= i <= j + n - 1
            # => i - n + 1 <= j <= i
            # j must be valid window start: 0 <= j <= len(word) - n
            start_j = max(0, i - n + 1)
            end_j = min(i, len(word) - n)
            
            can_remove = True
            for j in range(start_j, end_j + 1):
                h = window_hashes[j]
                if counts.get(h, 0) == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Perform deletion
                word.pop(i)
                found_deletion = True
                break # Restart scan with new word
        
        if not found_deletion:
            break
            
    return "".join(map(str, word))
