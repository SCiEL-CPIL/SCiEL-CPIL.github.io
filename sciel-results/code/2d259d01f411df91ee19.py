import time
import sys
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break

        perms, seen = [], set()
        # Extract unique (m-1)-length windows with distinct symbols
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Sort permutations to maximize overlap potential
        perms.sort()
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find maximum overlap
            max_check = min(len(new), len(block))
            overlap = 0
            for o in range(max_check, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new

    # Local optimization: try to remove redundant symbols
    if time.time() < deadline:
        word = _local_optimize(word, n, deadline)

    return "".join(map(str, word))

def _get_all_windows(word, n):
    """Get all n-length windows as tuples."""
    windows = set()
    for i in range(len(word) - n + 1):
        windows.add(tuple(word[i:i+n]))
    return windows

def _is_valid_superperm(word, n):
    """Check if word contains all n! permutations of 1..n."""
    if len(word) < n:
        return False
    all_perms = set(permutations(range(1, n+1)))
    windows = _get_all_windows(word, n)
    return all_perms.issubset(windows)

def _local_optimize(word, n, deadline):
    """Try to remove redundant symbols while maintaining validity."""
    word = list(word)
    best_len = len(word)
    
    # Try removing each symbol position
    i = 0
    while i < len(word) and time.time() < deadline:
        # Try removing word[i]
        test_word = word[:i] + word[i+1:]
        if _is_valid_superperm(test_word, n):
            word = test_word
            best_len = len(word)
            # Don't increment i, try removing next position
        else:
            i += 1
    
    # Multiple passes for better results
    pass_count = 0
    while time.time() < deadline and pass_count < 5:
        pass_count += 1
        removed_any = False
        i = 0
        while i < len(word) and time.time() < deadline:
            test_word = word[:i] + word[i+1:]
            if _is_valid_superperm(test_word, n):
                word = test_word
                removed_any = True
                # Don't increment i
            else:
                i += 1
        if not removed_any:
            break
    
    return word
