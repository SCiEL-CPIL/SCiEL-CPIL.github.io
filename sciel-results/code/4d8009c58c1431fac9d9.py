"""Improved superpermutation constructor with efficient local search."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import permutations
from collections import defaultdict

def _classic_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!"""
    word = [1]
    for m in range(2, n + 1):
        # Collect all distinct (m-1)-permutations in order of first appearance
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Maximum overlap between end of new and start of block
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _is_valid_word(word, n):
    """Check if word contains all permutations of 1..n as consecutive windows."""
    L = len(word)
    if L < n:
        return False
    target = n * math.factorial(n - 1)
    seen = set()
    for i in range(L - n + 1):
        seen.add(tuple(word[i:i+n]))
        if len(seen) >= target:
            break
    if len(seen) < target:
        return False
    for perm in permutations(range(1, n + 1)):
        if perm not in seen:
            return False
    return True

def _remove_chars_fast(word, n, time_limit):
    """
    Fast removal pass: try removing each position, check validity incrementally.
    Uses the insight that removing position i only affects windows starting in
    [i-n+1, i]. We precompute which windows are unique (count==1) and only
    attempt removal if no unique window is destroyed without being recreated.
    """
    start_time = time.time()
    word = list(word)
    L = len(word)
    target_perms = n * math.factorial(n - 1)
    all_perms = list(permutations(range(1, n + 1)))
    
    # Build initial window counts
    counts = defaultdict(int)
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        counts[w] += 1
    
    # Count how many permutations are present
    present_count = 0
    for perm in all_perms:
        if counts[perm] >= 1:
            present_count += 1
    
    if present_count < target_perms:
        return word  # Not valid yet
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        L = len(word)
        
        for i in range(L - 1, -1, -1):
            if (time.time() - start_time) >= time_limit:
                break
            if L <= n:
                break
            
            # Determine affected window start positions
            lo = max(0, i - n + 1)
            hi = min(i, L - n)
            
            # Check: after removing position i, do we lose any permutation?
            # Old windows at starts [lo, hi] are destroyed.
            # New windows appear at starts [lo, hi-1] (shifted right by 1 for those
            # that span the gap). Specifically, new window at start s (s in [lo, hi-1])
            # has content word[s:i] + word[i+1:s+n] (length n).
            
            # First, check if any old window at start s in [lo,hi] is unique
            # and not recreated as a new window.
            safe = True
            for s in range(lo, hi + 1):
                old_w = tuple(word[s:s+n])
                if counts[old_w] == 1:
                    # This window is unique. Check if it appears as a new window.
                    # New windows are at starts s' in [lo, hi-1] (if hi-1 >= lo)
                    # New window at start s' spans positions [s', s'+n) in the new word.
                    # In old word coords, it covers [s', i) and [i+1, s'+n+1).
                    # It equals old_w iff word[s':i] + word[i+1:s'+n+1] == old_w
                    # This is complex; simpler to just do a quick check.
                    # For now, be conservative: if any unique window is affected, skip.
                    safe = False
                    break
            
            if not safe:
                continue
            
            # Perform removal and verify
            test_word = word[:i] + word[i+1:]
            TL = len(test_word)
            if TL < n:
                continue
            
            # Quick check: count distinct windows in test_word
            seen = set()
            for j in range(TL - n + 1):
                seen.add(tuple(test_word[j:j+n]))
                if len(seen) >= target_perms:
                    break
            
            if len(seen) < target_perms:
                continue
            
            # Full validation
            valid = True
            for perm in all_perms:
                if perm not in seen:
                    valid = False
                    break
            
            if valid:
                word = test_word
                L = TL
                improved = True
                # Rebuild counts
                counts = defaultdict(int)
                for j in range(L - n + 1):
                    w = tuple(word[j:j+n])
                    counts[w] += 1
                break
    
    return word

def _remove_chars_bidirectional(word, n, time_limit):
    """
    Removal pass that tries both left-to-right and right-to-left passes,
    and also tries removing pairs of adjacent characters.
    """
    start_time = time.time()
    word = list(word)
    target_perms = n * math.factorial(n - 1)
    all_perms = list(permutations(range(1, n + 1)))
    
    def try_remove_single(word, n, time_limit, direction):
        """Try removing single characters in given direction."""
        L = len(word)
        counts = defaultdict(int)
        for i in range(L - n + 1):
            w = tuple(word[i:i+n])
            counts[w] += 1
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            L = len(word)
            
            if direction == 'fwd':
                indices = range(L)
            else:
                indices = range(L - 1, -1, -1)
            
            for i in indices:
                if (time.time() - start_time) >= time_limit:
                    break
                if L <= n:
                    break
                
                lo = max(0, i - n + 1)
                hi = min(i, L - n)
                
                safe = True
                for s in range(lo, hi + 1):
                    w = tuple(word[s:s+n])
                    if counts[w] == 1:
                        safe = False
                        break
                
                if not safe:
                    continue
                
                test_word = word[:i] + word[i+1:]
                TL = len(test_word)
                if TL < n:
                    continue
                
                seen = set()
                for j in range(TL - n + 1):
                    seen.add(tuple(test_word[j:j+n]))
                    if len(seen) >= target_perms:
                        break
                
                if len(seen) < target_perms:
                    continue
                
                valid = True
                for perm in all_perms:
                    if perm not in seen:
                        valid = False
                        break
                
                if valid:
                    word = test_word
                    L = TL
                    improved = True
                    counts = defaultdict(int)
                    for j in range(L - n + 1):
                        w = tuple(word[j:j+n])
                        counts[w] += 1
                    break
        
        return word
    
    def try_remove_pairs(word, n, time_limit):
        """Try removing two adjacent characters."""
        L = len(word)
        counts = defaultdict(int)
        for i in range(L - n + 1):
            w = tuple(word[i:i+n])
            counts[w] += 1
        
        improved = True
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            L = len(word)
            
            for i in range(L - 1):
                if (time.time() - start_time) >= time_limit:
                    break
                if L <= n + 1:
                    break
                
                # Removing positions i and i+1 affects windows in [i-n+1, i+1]
                lo = max(0, i - n + 1)
                hi = min(i + 1, L - n)
                
                safe = True
                for s in range(lo, hi + 1):
                    w = tuple(word[s:s+n])
                    if counts[w] == 1:
                        safe = False
                        break
                
                if not safe:
                    continue
                
                test_word = word[:i] + word[i+2:]
                TL = len(test_word)
                if TL < n:
                    continue
                
                seen = set()
                for j in range(TL - n + 1):
                    seen.add(tuple(test_word[j:j+n]))
                    if len(seen) >= target_perms:
                        break
                
                if len(seen) < target_perms:
                    continue
                
                valid = True
                for perm in all_perms:
                    if perm not in seen:
                        valid = False
                        break
                
                if valid:
                    word = test_word
                    L = TL
                    improved = True
                    counts = defaultdict(int)
                    for j in range(L - n + 1):
                        w = tuple(word[j:j+n])
                        counts[w] += 1
                    break
        
        return word
    
    # Forward pass
    word = try_remove_single(word, n, time_limit, 'fwd')
    
    if (time.time() - start_time) < time_limit:
        # Backward pass
        word = try_remove_single(word, n, time_limit, 'bwd')
    
    if (time.time() - start_time) < time_limit:
        # Pair removal
        word = try_remove_pairs(word, n, time_limit)
    
    return word

def _local_search_improve(word, n, time_limit):
    """
    Local search: try moving characters to reduce length while maintaining validity.
    Focus on finding shorter valid words by reordering.
    """
    start_time = time.time()
    word = list(word)
    L = len(word)
    target_perms = n * math.factorial(n - 1)
    all_perms = list(permutations(range(1, n + 1)))
    
    best_word = list(word)
    best_len = L
    
    # First, try to remove characters (as before)
    word = _remove_chars_fast(word, n, time_limit * 0.4)
    if len(word) < best_len:
        best_word = list(word)
        best_len = len(word)
    
    remaining = time_limit - (time.time() - start_time)
    if remaining < 1:
        return best_word
    
    # Now try local reordering: for each position, try swapping with nearby positions
    # and check if we can then remove more characters
    current_word = list(word)
    current_len = len(word)
    
    # Build a set of all valid permutations for quick lookup
    perm_set = set(all_perms)
    
    def is_valid(w):
        Lw = len(w)
        if Lw < n:
            return False
        seen = set()
        for i in range(Lw - n + 1):
            seen.add(tuple(w[i:i+n]))
            if len(seen) >= target_perms:
                break
        if len(seen) < target_perms:
            return False
        return all(p in seen for p in perm_set)
    
    # Try random swaps followed by removal
    rng = random.Random(42)
    iterations = 0
    max_iters = 50000
    
    while (time.time() - start_time) < time_limit and iterations < max_iters:
        iterations += 1
        if current_len <= best_len:
            # Try to improve
            i = rng.randint(0, current_len - 1)
            j = rng.randint(0, current_len - 1)
            if i == j:
                continue
            
            test_word = list(current_word)
            test_word[i], test_word[j] = test_word[j], test_word[i]
            
            if not is_valid(test_word):
                continue
            
            # Try to remove characters from this new word
            improved_word = _remove_chars_fast(test_word, n, 0.5)
            
            if len(improved_word) < best_len:
                best_word = list(improved_word)
                best_len = len(improved_word)
                current_word = list(improved_word)
                current_len = best_len
            elif len(improved_word) < current_len:
                current_word = list(improved_word)
                current_len = len(improved_word)
    
    return best_word

def construct(n):
    start_time = time.time()
    time_limit = 140  # Leave 10s safety margin for 150s limit
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if remaining > 5:
        # Fast removal pass
        word = _remove_chars_fast(word, n, remaining * 0.3)
        
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        
        if remaining > 5:
            # Bidirectional removal with pair removal
            word = _remove_chars_bidirectional(word, n, remaining * 0.4)
        
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        
        if remaining > 2:
            # Local search for further improvement
            word = _local_search_improve(word, n, remaining)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
