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

def _count_windows(word, n):
    """Count occurrences of each permutation window in word."""
    counts = defaultdict(int)
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        counts[window] += 1
    return counts

def _is_valid_counts(counts, n):
    """Check if all permutations have count >= 1."""
    for perm in permutations(range(1, n + 1)):
        if counts[perm] < 1:
            return False
    return True

def _try_remove_chars_v2(word, n, time_limit):
    """More efficient: maintain window counts and only check affected windows."""
    start_time = time.time()
    word = list(word)
    L = len(word)
    
    # Build initial counts
    counts = defaultdict(int)
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        counts[w] += 1
    
    # Precompute: which permutations are "critical" (count == 1)
    critical = set()
    all_perms = list(permutations(range(1, n + 1)))
    for perm in all_perms:
        if counts[perm] <= 1:
            critical.add(perm)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(L - 1, -1, -1):
            if (time.time() - start_time) >= time_limit:
                break
            if L <= n:
                break
            
            # Windows that include position i: start positions in [i-n+1, i] ∩ [0, L-n]
            lo = max(0, i - n + 1)
            hi = min(i, L - n)
            
            # These windows will disappear. Check if any critical window disappears
            # and is not recreated by the shift
            safe = True
            for s in range(lo, hi + 1):
                w = tuple(word[s:s+n])
                if counts[w] == 1:
                    # This window appears only here. After removal, it's gone unless
                    # a new window with the same content appears due to the shift
                    safe = False
                    break
            
            if not safe:
                continue
            
            # Removal seems safe, do full verification
            test_word = word[:i] + word[i+1:]
            if len(test_word) < n:
                continue
            
            seen = set()
            for j in range(len(test_word) - n + 1):
                w = tuple(test_word[j:j+n])
                seen.add(w)
                if len(seen) == n * math.factorial(n - 1):
                    break
            
            if len(seen) == n * math.factorial(n - 1):
                valid = True
                for perm in all_perms:
                    if perm not in seen:
                        valid = False
                        break
                if valid:
                    word = test_word
                    L = len(word)
                    improved = True
                    # Rebuild counts
                    counts = defaultdict(int)
                    for j in range(L - n + 1):
                        w = tuple(word[j:j+n])
                        counts[w] += 1
                    critical = set()
                    for perm in all_perms:
                        if counts[perm] <= 1:
                            critical.add(perm)
                    break
    
    return word

def _try_remove_chars_v3(word, n, time_limit):
    """Even more efficient removal with better window tracking."""
    start_time = time.time()
    word = list(word)
    L = len(word)
    target_perms = n * math.factorial(n - 1)
    all_perms = list(permutations(range(1, n + 1)))
    
    # Build initial counts
    counts = defaultdict(int)
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        counts[w] += 1
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(L - 1, -1, -1):
            if (time.time() - start_time) >= time_limit:
                break
            if L <= n:
                break
            
            # Windows that include position i: start positions in [i-n+1, i] ∩ [0, L-n]
            lo = max(0, i - n + 1)
            hi = min(i, L - n)
            
            # Check if removal is safe by examining affected windows
            safe = True
            for s in range(lo, hi + 1):
                w = tuple(word[s:s+n])
                if counts[w] == 1:
                    # Check if this window reappears due to the shift
                    # New windows appear at positions where the gap is filled
                    # A window spanning the gap at position s' has content:
                    # word[s':i] + word[i+1:s'+n] for s' in [max(0, i-n+1), min(i, L-n-1)]
                    # Simplified: check if w appears in the new word at any position
                    # This is expensive, so we just skip removal if any critical window is affected
                    safe = False
                    break
            
            if not safe:
                continue
            
            # Removal seems safe, do full verification
            test_word = word[:i] + word[i+1:]
            if len(test_word) < n:
                continue
            
            seen = set()
            for j in range(len(test_word) - n + 1):
                w = tuple(test_word[j:j+n])
                seen.add(w)
                if len(seen) == target_perms:
                    break
            
            if len(seen) == target_perms:
                valid = True
                for perm in all_perms:
                    if perm not in seen:
                        valid = False
                        break
                if valid:
                    word = test_word
                    L = len(word)
                    improved = True
                    # Rebuild counts
                    counts = defaultdict(int)
                    for j in range(L - n + 1):
                        w = tuple(word[j:j+n])
                        counts[w] += 1
                    break
    
    return word

def _simulated_annealing_improve(word, n, time_limit):
    """Use simulated annealing to try to shorten the superpermutation."""
    start_time = time.time()
    word = list(word)
    L = len(word)
    target_perms = n * math.factorial(n - 1)
    all_perms = list(permutations(range(1, n + 1)))
    
    # Initial energy: length of word
    best_word = list(word)
    best_len = L
    current_word = list(word)
    current_len = L
    
    temperature = 100.0
    cooling_rate = 0.999
    
    while (time.time() - start_time) < time_limit and temperature > 0.1:
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.5 and L > n:
            # Try to remove a random character
            i = random.randint(0, L - 1)
            test_word = current_word[:i] + current_word[i+1:]
            if len(test_word) >= n:
                # Check if still valid
                seen = set()
                for j in range(len(test_word) - n + 1):
                    w = tuple(test_word[j:j+n])
                    seen.add(w)
                    if len(seen) == target_perms:
                        break
                if len(seen) == target_perms:
                    valid = True
                    for perm in all_perms:
                        if perm not in seen:
                            valid = False
                            break
                    if valid:
                        current_word = test_word
                        current_len = len(test_word)
                        if current_len < best_len:
                            best_word = list(current_word)
                            best_len = current_len
                else:
                    # Reject (or accept with probability)
                    delta_e = current_len - len(test_word)  # Negative means improvement
                    if delta_e < 0:
                        if random.random() < math.exp(-delta_e / temperature):
                            current_word = test_word
                            current_len = len(test_word)
        elif move_type < 0.8 and L > n + 1:
            # Try to swap two adjacent characters
            i = random.randint(0, L - 2)
            test_word = list(current_word)
            test_word[i], test_word[i+1] = test_word[i+1], test_word[i]
            # Check if still valid
            seen = set()
            for j in range(len(test_word) - n + 1):
                w = tuple(test_word[j:j+n])
                seen.add(w)
                if len(seen) == target_perms:
                    break
            if len(seen) == target_perms:
                valid = True
                for perm in all_perms:
                    if perm not in seen:
                        valid = False
                        break
                if valid:
                    current_word = test_word
        elif move_type < 0.95 and L > n + 1:
            # Try to move a character to a different position
            i = random.randint(0, L - 1)
            j = random.randint(0, L - 1)
            if i != j:
                test_word = list(current_word)
                char = test_word.pop(i)
                test_word.insert(j, char)
                # Check if still valid
                seen = set()
                for k in range(len(test_word) - n + 1):
                    w = tuple(test_word[k:k+n])
                    seen.add(w)
                    if len(seen) == target_perms:
                        break
                if len(seen) == target_perms:
                    valid = True
                    for perm in all_perms:
                        if perm not in seen:
                            valid = False
                            break
                    if valid:
                        current_word = test_word
        else:
            # No-op or other move
            pass
        
        temperature *= cooling_rate
    
    return best_word

def construct(n):
    start_time = time.time()
    time_limit = 140  # Leave 10s safety margin for 150s limit
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed
    
    if remaining > 10:
        # Try to remove redundant characters with v2
        word = _try_remove_chars_v2(word, n, remaining * 0.5)
        
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed
        
        if remaining > 5:
            # Try to remove redundant characters with v3
            word = _try_remove_chars_v3(word, n, remaining * 0.5)
            
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            
            if remaining > 5:
                # Try simulated annealing for further improvement
                word = _simulated_annealing_improve(word, n, remaining)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
