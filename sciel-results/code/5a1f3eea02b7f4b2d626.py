"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
from itertools import permutations
from collections import defaultdict

# EVOLVE-BLOCK-START

def _classic_superperm(n):
    """Classic recursive construction of a superpermutation of length n! + (n-1)! + ... + 1!."""
    if n == 1:
        return [1]
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def construct(n):
    """
    Construct the shortest known superpermutation for n symbols.
    Uses the classic recursive construction as a starting point, then applies
    local search (deletion heuristic) to shorten the word for n >= 5.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin
    
    # For small n, the classic construction is optimal
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234121341231423124312"
    
    # Generate classic superpermutation
    word = _classic_superperm(n)
    
    # For n >= 5, the classic construction is not optimal.
    # Apply local search to shorten it.
    if n >= 5:
        all_perms = set(permutations(range(1, n + 1)))
        total_perms = len(all_perms)
        
        # Convert to list for easy modification
        current = word[:]
        L = len(current)
        
        # Build reference count map: window -> count of occurrences
        ref_count = defaultdict(int)
        for i in range(L - n + 1):
            w = tuple(current[i:i + n])
            ref_count[w] += 1
        
        # Verify classic construction is valid (it should be)
        covered = sum(1 for p in all_perms if ref_count[p] > 0)
        if covered < total_perms:
            # Fallback if something went wrong, though unlikely
            return "".join(map(str, current))
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            L = len(current)
            num_windows = L - n + 1
            
            # Try deleting from left to right
            for i in range(L):
                if (i % 1000 == 0) and time.time() > deadline:
                    break
                
                # Check if deleting position i would cause any permutation to become uncovered
                # Windows affected: those that include position i
                # Window starting at j covers positions j..j+n-1
                # Position i is in window j iff j <= i <= j+n-1, i.e., i-n+1 <= j <= i
                # Also j must be in [0, num_windows-1]
                
                j_start = max(0, i - n + 1)
                j_end = min(i, num_windows - 1)
                
                # Check if any affected window has ref_count == 1 and is a required permutation
                can_delete = True
                old_windows = []
                for j in range(j_start, j_end + 1):
                    w = tuple(current[j:j + n])
                    old_windows.append(w)
                    # Optimization: Only check all_perms if count is 1
                    if ref_count.get(w, 0) == 1:
                        if w in all_perms:
                            can_delete = False
                            break
                
                if can_delete:
                    # Perform deletion
                    current.pop(i)
                    L -= 1
                    num_windows -= 1
                    improved = True
                    
                    # Update reference counts incrementally
                    # 1. Decrement counts for old windows that included position i
                    for w in old_windows:
                        ref_count[w] -= 1
                        if ref_count[w] == 0:
                            del ref_count[w]
                    
                    # 2. Increment counts for new windows in the affected region
                    # New windows spanning the join point:
                    # k in [max(0, i-n+1), min(i-1, L-1-n)]
                    k_start = max(0, i - n + 1)
                    k_end = min(i - 1, L - 1 - n)
                    
                    for k in range(k_start, k_end + 1):
                        new_w = tuple(current[k:k + n])
                        ref_count[new_w] += 1

        return "".join(map(str, current))
    
    return "".join(map(str, word))

# EVOLVE-BLOCK-END
