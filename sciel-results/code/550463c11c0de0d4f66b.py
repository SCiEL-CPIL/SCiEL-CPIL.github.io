import time
import sys
from collections import defaultdict
import numpy as np

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def classic_construct(n):
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
            overlap = 0
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    if n == 4:
        return "".join(map(str, classic_construct(n)))
    
    word = classic_construct(n)
    target_count = factorial(n)
    L = len(word)
    
    # Use array-based counting for speed
    # Encode permutations as integers for fast hashing
    # For n symbols, each permutation can be encoded as a base-n number
    # But n! permutations, so we can use a dict
    
    # Build initial counts
    counts = defaultdict(int)
    for i in range(L - n + 1):
        w = tuple(word[i:i+n])
        counts[w] += 1
    
    if len(counts) != target_count:
        return "".join(map(str, word))
    
    # Local search: try to delete symbols
    # Use a queue of candidate positions
    # Priority: positions where deletion creates new windows that are already well-represented
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        L = len(word)
        
        # Try positions from middle to ends (arbitrary heuristic)
        # Actually, try all positions, pick the best valid deletion
        best_pos = -1
        best_gain = 0
        
        for i in range(L):
            if time.time() > deadline - 10.0:
                break
            
            min_j = max(0, i - n + 1)
            max_j = min(L - n, i)
            
            valid = True
            for j in range(min_j, max_j + 1):
                w = tuple(word[j:j+n])
                if counts[w] == 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # This deletion is valid. Calculate "gain" - how many new windows are duplicates
            # New windows are those crossing the deletion point
            new_windows = []
            for k in range(max(0, i - n + 1), min(L - n, i - 1) + 1):
                p1 = word[k:i]
                p2_len = n - len(p1)
                p2 = word[i+1:i+1+p2_len]
                if len(p1) + len(p2) == n:
                    new_w = tuple(p1 + p2)
                    new_windows.append(new_w)
            
            # Gain: number of new windows that already have count > 0 (duplicates)
            gain = 0
            for w in new_windows:
                if counts.get(w, 0) > 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_pos = i
        
        if best_pos == -1:
            # Try random deletions as fallback
            import random
            random.seed(int(time.time() * 1000) % 10000)
            for _ in range(min(100, L)):
                i = random.randint(0, L - 1)
                min_j = max(0, i - n + 1)
                max_j = min(L - n, i)
                
                valid = True
                for j in range(min_j, max_j + 1):
                    w = tuple(word[j:j+n])
                    if counts[w] == 1:
                        valid = False
                        break
                
                if not valid:
                    continue
                
                # Commit
                for j in range(min_j, max_j + 1):
                    w = tuple(word[j:j+n])
                    counts[w] -= 1
                    if counts[w] == 0:
                        del counts[w]
                
                new_windows = []
                for k in range(max(0, i - n + 1), min(L - n, i - 1) + 1):
                    p1 = word[k:i]
                    p2_len = n - len(p1)
                    p2 = word[i+1:i+1+p2_len]
                    if len(p1) + len(p2) == n:
                        new_w = tuple(p1 + p2)
                        new_windows.append(new_w)
                
                for w in new_windows:
                    counts[w] += 1
                
                word = word[:i] + word[i+1:]
                improved = True
                break
        
        if not improved and best_pos != -1:
            # Commit the best deletion
            i = best_pos
            min_j = max(0, i - n + 1)
            max_j = min(L - n, i)
            
            for j in range(min_j, max_j + 1):
                w = tuple(word[j:j+n])
                counts[w] -= 1
                if counts[w] == 0:
                    del counts[w]
            
            new_windows = []
            for k in range(max(0, i - n + 1), min(L - n, i - 1) + 1):
                p1 = word[k:i]
                p2_len = n - len(p1)
                p2 = word[i+1:i+1+p2_len]
                if len(p1) + len(p2) == n:
                    new_w = tuple(p1 + p2)
                    new_windows.append(new_w)
            
            for w in new_windows:
                counts[w] += 1
            
            word = word[:i] + word[i+1:]
            improved = True
    
    return "".join(map(str, word))
