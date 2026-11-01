import time
import itertools
from math import factorial

# EVOLVE-BLOCK-START

def _standard_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!."""
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
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word


def _validate(word, n):
    """Check if word contains all n! permutations of 1..n as n-consecutive symbols."""
    if n <= 1:
        return len(word) == n
    if len(word) < n:
        return False
    required = set(itertools.permutations(range(1, n + 1)))
    found = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            found.add(window)
    return found == required


def _optimize_superperm(word, n, time_limit=100):
    """
    Local search to shorten the superpermutation.
    Uses incremental count updates for O(n) per removal check.
    """
    if n <= 1:
        return word
    
    word = list(word)
    L = len(word)
    
    # Map permutations to indices
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    counts = [0] * num_perms
    
    def update_counts(w, start, end, delta):
        """Update counts for windows w[start..end] (inclusive) by delta."""
        for i in range(start, end + 1):
            if i + n > len(w):
                break
            window = tuple(w[i:i + n])
            if len(set(window)) == n:
                idx = perm_to_idx.get(window, -1)
                if idx != -1:
                    counts[idx] += delta
    
    # Initial counts
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            idx = perm_to_idx.get(window, -1)
            if idx != -1:
                counts[idx] += 1
    
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        L = len(word)
        # Try to remove each symbol
        for i in range(L):
            if (time.time() - start_time) > time_limit:
                return word
            
            # Identify windows to remove: those touching i
            # j such that j <= i <= j+n-1  =>  i-n+1 <= j <= i
            start_remove = max(0, i - n + 1)
            end_remove = min(i, L - n)
            
            # Check if removal is valid
            # We simulate the removal
            # 1. Decrement counts for windows [start_remove, end_remove]
            # 2. Increment counts for new crossing windows
            # 3. Check if any count becomes 0
            
            # First, check if any of the windows to be removed are the ONLY occurrence
            # We need to be careful: if a window is in both remove and add sets?
            # The new windows are different from the old ones (they cross the gap).
            
            # Let's collect the new windows
            new_windows = []
            # New windows start at i-k in the new word, for k=1..n-1
            # In the old word, they correspond to:
            # w[i-k .. i-2] + w[i+1 .. i-1+k-1] ?
            # Let's construct them explicitly
            for k in range(1, n):
                # Left part: w[i-k .. i-1] (length k)
                # Right part: w[i+1 .. i+1+(n-k)-1] (length n-k)
                left = word[i-k:i]
                right = word[i+1:i+1+n-k]
                if len(left) + len(right) != n:
                    continue
                new_win = tuple(left + right)
                if len(set(new_win)) == n:
                    new_windows.append(new_win)
            
            # Check validity
            valid = True
            # We need to know if any permutation count drops to 0.
            # Let's just simulate the count changes on a copy? No, too slow.
            # Let's check the condition directly:
            # For each perm p in the remove set:
            #   if counts[p] == 1 and p is not in new_windows (or counts in new_windows < 1? No, new_windows are added)
            #   Actually, if p is in remove set and counts[p] == 1, it will be 0 unless p is in new_windows.
            #   If p is in new_windows, its count increases by 1 (or more), so it stays >= 1.
            
            # So:
            # 1. Build set of new window indices
            new_indices = set()
            for nw in new_windows:
                idx = perm_to_idx.get(nw, -1)
                if idx != -1:
                    new_indices.add(idx)
            
            # 2. Check remove windows
            for j in range(start_remove, end_remove + 1):
                window = tuple(word[j:j + n])
                if len(set(window)) == n:
                    idx = perm_to_idx.get(window, -1)
                    if idx != -1:
                        if counts[idx] == 1 and idx not in new_indices:
                            valid = False
                            break
            
            if valid:
                # Perform removal
                # Update counts
                for j in range(start_remove, end_remove + 1):
                    window = tuple(word[j:j + n])
                    if len(set(window)) == n:
                        idx = perm_to_idx.get(window, -1)
                        if idx != -1:
                            counts[idx] -= 1
                
                for nw in new_windows:
                    idx = perm_to_idx.get(nw, -1)
                    if idx != -1:
                        counts[idx] += 1
                
                word.pop(i)
                improved = True
                break # Restart loop with new length
    
    return word


def construct(n):
    if n <= 1:
        return "".join(str(i) for i in range(1, n + 1))
    
    start_time = time.time()
    deadline = 110  # safety margin before 120s limit
    
    # For small n, use the exact optimal construction
    if n <= 4:
        return "".join(map(str, _standard_superperm(n)))
    
    # For n=5,6,7: use standard + local search
    base_word = _standard_superperm(n)
    
    # Try to improve with local search
    improved_word = _optimize_superperm(base_word, n, time_limit=deadline - (time.time() - start_time) - 5)
    
    # Validate
    if _validate(improved_word, n):
        return "".join(map(str, improved_word))
    
    return "".join(map(str, base_word))

# EVOLVE-BLOCK-END
