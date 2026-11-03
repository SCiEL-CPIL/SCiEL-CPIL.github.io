import time
import itertools
import numpy as np

def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
    if n == 1:
        return [1]
    word = _classic_superperm(n - 1)
    m = n
    perms, seen = [], set()
    # Extract (n-1)-permutations in order of appearance
    for i in range(len(word) - m + 2):
        window = tuple(word[i:i + m - 1])
        if len(set(window)) == m - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
            
    new = []
    for p in perms:
        block = list(p) + [m] + list(p)
        # Find max overlap with end of new
        overlap = 0
        max_ov = min(len(new), len(block))
        for o in range(max_ov, 0, -1):
            if new[len(new) - o:] == block[:o]:
                overlap = o
                break
        new += block[overlap:]
    return new

def _is_permutation(window, n):
    """Check if window is a permutation of 1..n."""
    if len(window) != n:
        return False
    # Fast check for small n
    if n <= 8:
        return len(set(window)) == n and all(1 <= x <= n for x in window)
    return False

def _compute_perm_counts(word, n):
    """Compute counts of all permutations in word."""
    counts = {}
    L = len(word)
    if L < n:
        return counts
    for i in range(L - n + 1):
        window = tuple(word[i:i+n])
        if _is_permutation(window, n):
            counts[window] = counts.get(window, 0) + 1
    return counts

def _remove_safe_chars(word, n, deadline):
    """
    Iteratively remove characters that are not part of any unique permutation.
    """
    start_time = time.time()
    counts = _compute_perm_counts(word, n)
    
    # Check if we already have all perms
    if len(counts) < int(np.math.factorial(n)):
        # If not, we shouldn't remove anything? 
        # Actually, if we are missing perms, we are invalid. 
        # But we start with a valid superperm.
        pass
        
    while time.time() < deadline:
        L = len(word)
        if L < n:
            break
            
        candidates = []
        
        # Identify indices that are part of any unique permutation
        # An index i is 'unsafe' if there exists a j such that 
        # j <= i < j+n, window W[j:j+n] is a perm, and count[W[j:j+n]] == 1
        
        # Efficiently: 
        # For each unique perm, mark its indices as unsafe.
        unsafe = [False] * L
        
        for p, c in counts.items():
            if c == 1:
                # Find where p occurs
                # Since c==1, it occurs exactly once.
                # We need to find the index.
                # We can search, but that's O(L) per unique perm.
                # Better: during count computation, store the index?
                # For now, let's just scan. 
                # 5040 unique perms * 5900 length = 30M ops. Might be slow.
                
                # Optimization: 
                # We can just check candidates directly.
                pass
        
        # Alternative: Direct candidate check
        for i in range(L):
            if time.time() > deadline:
                return word
            
            safe = True
            # Check windows containing i
            start_j = max(0, i - n + 1)
            end_j = min(i + 1, L - n + 1)
            
            for j in range(start_j, end_j):
                window = tuple(word[j:j+n])
                if _is_permutation(window, n):
                    if counts.get(window, 0) == 1:
                        safe = False
                        break
            
            if safe:
                candidates.append(i)
                
        if not candidates:
            break
            
        # Remove the first candidate
        i = candidates[0]
        word = word[:i] + word[i+1:]
        
        # Recompute counts
        counts = _compute_perm_counts(word, n)
        
        # Optional: Early exit if we hit a known target length?
        # For n=7, target is 5905.
        
    return word

def construct(n):
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "123412134132413241423142314321"
        
    if n <= 6:
        word = _classic_superperm(n)
        return "".join(map(str, word))
        
    if n == 7:
        # Generate classic word
        word = _classic_superperm(n)
        
        # Optimize by removing safe chars
        time_left = deadline - (time.time() - start_time)
        if time_left > 5.0:
            word = _remove_safe_chars(word, n, time_left - 5.0)
            
        return "".join(map(str, word))
        
    # For n > 7, just use classic
    word = _classic_superperm(n)
    return "".join(map(str, word))
