import time
import sys
from collections import deque

def construct(n):
    """
    Constructs a superpermutation for n symbols using the classic recursive
    construction, then attempts local shortening to reduce length.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Classic recursive construction
    word = [1]
    for m in range(2, n + 1):
        if time.time() > deadline:
            break
        
        # Extract all unique permutations of length m-1 from the current word
        # in order of first appearance
        perms, seen = [], set()
        k = m - 1
        # Use a bitmask to quickly check if window is a permutation of 1..k
        # Symbols are 1..k, so mask has bits 0..k-1 set
        full_mask = (1 << k) - 1
        
        for i in range(len(word) - k + 1):
            window = word[i:i + k]
            # Quick check: all elements distinct and in range 1..k
            mask = 0
            valid = True
            for sym in window:
                bit = 1 << (sym - 1)
                if mask & bit:
                    valid = False
                    break
                mask |= bit
            if valid and mask == full_mask:
                wkey = tuple(window)
                if wkey not in seen:
                    seen.add(wkey)
                    perms.append(window)
        
        new = []
        for p in perms:
            # Construct the block: p, m, p
            block = p + [m] + p
            
            # Find maximum overlap with the current end of 'new'
            # We want to append block[overlap:]
            max_overlap = min(len(new), len(block))
            overlap = 0
            # Optimization: only check up to max_overlap
            for o in range(max_overlap, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            
            new += block[overlap:]
        
        word = new
    
    # Local shortening pass: try to remove redundant symbols
    # A symbol at position i can be removed if removing it doesn't destroy
    # any permutation window. We check windows that contain position i.
    if time.time() < deadline - 30:
        word = _local_shorten(word, n, deadline)
    
    return "".join(map(str, word))


def _local_shorten(word, n, deadline):
    """
    Attempt to shorten the superpermutation by removing redundant symbols.
    We scan for positions where removal preserves all permutation windows.
    """
    word = list(word)
    length = len(word)
    
    # Build a count of how many times each permutation appears
    # For n=7, there are 5040 permutations. We track which windows contain them.
    # A position i is "covered" by a window [j, j+n-1] if j <= i < j+n.
    # If we remove position i, all windows containing i are destroyed.
    # We can only remove i if every permutation that appeared in those windows
    # also appears in at least one other window not containing i.
    
    # This is expensive for n=7. Instead, use a heuristic:
    # Try removing each position and check if all permutations still exist.
    # To speed up, only try positions that are "redundant" based on local patterns.
    
    # Simpler heuristic: look for repeated patterns where we can drop a symbol.
    # Specifically, if word[i] == word[i+1], we can try removing one.
    # But in a superpermutation, consecutive duplicates are rare.
    
    # Better heuristic: try removing each position i, and verify locally
    # that no permutation is lost. Use a sliding window count.
    
    # For efficiency with n=7 and length ~5913, we'll try a limited number
    # of removals and use a fast check.
    
    removed = 0
    i = 0
    while i < len(word) - 1 and time.time() < deadline:
        # Try removing position i
        # Check if word[i] appears in word[i+1] (adjacent duplicate)
        if word[i] == word[i+1]:
            # Try removing word[i]
            new_word = word[:i] + word[i+1:]
            if _is_valid_superperm(new_word, n):
                word = new_word
                removed += 1
                # Don't advance i, check same position again
            else:
                i += 1
        else:
            i += 1
    
    return word


def _is_valid_superperm(word, n):
    """Check if word contains all permutations of 1..n as consecutive substrings."""
    length = len(word)
    if length < n:
        return False
    
    seen = set()
    k = n
    full_mask = (1 << k) - 1
    
    for i in range(length - k + 1):
        window = word[i:i + k]
        mask = 0
        valid = True
        for sym in window:
            bit = 1 << (sym - 1)
            if mask & bit:
                valid = False
                break
            mask |= bit
        if valid and mask == full_mask:
            wkey = tuple(window)
            if wkey in seen:
                continue
            seen.add(wkey)
            if len(seen) == _factorial(n):
                return True
    
    return len(seen) == _factorial(n)


def _factorial(n):
    """Compute n!"""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
