import math
import time
import random

def _classic_superpermutation(n):
    """Generates the classic superpermutation of length n! + (n-1)! + ... + 1!"""
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract all (m-1)-permutations in order of appearance
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            # Find max overlap
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _is_valid_superpermutation(word, n):
    """Checks if word contains all permutations of 1..n"""
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        seen.add(tuple(word[i:i+n]))
    return len(seen) == int(math.factorial(n))

def _fast_greedy_delete(word, n, deadline):
    """
    Aggressively deletes characters from the word if it remains a valid superpermutation.
    Uses a frequency map to check validity in O(n) per candidate.
    """
    current_word = list(word)
    L = len(current_word)
    
    # Initialize frequency map
    freq = {}
    for i in range(L - n + 1):
        p = tuple(current_word[i:i+n])
        freq[p] = freq.get(p, 0) + 1
        
    while True:
        if time.time() > deadline:
            break
            
        deleted = False
        L = len(current_word)
        
        # Iterate through all possible deletion indices
        for i in range(L):
            # Identify windows in current_word that contain index i
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            
            # Check if any of these windows has frequency 1
            can_delete = True
            for s in range(start_min, start_max + 1):
                p = tuple(current_word[s:s+n])
                if freq[p] == 1:
                    can_delete = False
                    break
            
            if can_delete:
                # Perform deletion
                for s in range(start_min, start_max + 1):
                    p = tuple(current_word[s:s+n])
                    freq[p] -= 1
                    if freq[p] == 0:
                        del freq[p]
                
                new_word = current_word[:i] + current_word[i+1:]
                
                # Increment freq for new windows
                new_L = len(new_word)
                add_start_min = max(0, i - n + 1)
                add_start_max = min(i - 1, new_L - n)
                
                for j in range(add_start_min, add_start_max + 1):
                    p = tuple(new_word[j:j+n])
                    freq[p] = freq.get(p, 0) + 1
                
                current_word = new_word
                deleted = True
                break
        
        if not deleted:
            break
            
    return current_word

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    if n <= 5:
        return "".join(map(str, _classic_superpermutation(n)))
    
    # Start with classic
    word = _classic_superpermutation(n)
    
    # If n=6, classic is optimal (873). No need to search.
    if n == 6:
        return "".join(map(str, word))
    
    # For n=7, try to shorten
    best_word = word
    
    # Pass 1: Forward greedy deletion
    w1 = _fast_greedy_delete(best_word, n, deadline)
    if len(w1) < len(best_word):
        best_word = w1
        
    # Pass 2: Backward greedy deletion (Reverse, Delete, Reverse)
    if time.time() < deadline - 5:
        rev_word = best_word[::-1]
        w2 = _fast_greedy_delete(rev_word, n, deadline)
        if len(w2) < len(best_word):
            best_word = w2[::-1]
            
    return "".join(map(str, best_word))
