import math
import time
import sys
import random

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Recursive construction
    word = [1]
    for m in range(2, n + 1):
        # Extract unique permutations of length m-1 from word
        # in order of appearance
        perms = []
        seen = set()
        # word length is L. windows of size m-1
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            # Check if it's a permutation (distinct elements)
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new_word = []
        for p in perms:
            # Block: p[0]...p[m-2], m, p[0]...p[m-2]
            block = list(p) + [m] + list(p)
            
            # Find max overlap with new_word
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, -1, -1):
                if o == 0:
                    break
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word.extend(block[max_overlap:])
        
        word = new_word
        
        # Time check
        if time.time() > deadline:
            break
    
    # Local search polish
    # Try to remove characters
    if time.time() < deadline - 10.0:
        # We need to check if all permutations are present
        def is_valid(w):
            if len(w) < n: return False
            perms_found = set()
            for i in range(len(w) - n + 1):
                window = tuple(w[i:i+n])
                if len(set(window)) == n:
                    perms_found.add(window)
            return len(perms_found) == int(math.factorial(n))
        
        # Optimization: Pre-compute target set?
        # For n=7, 5040 perms.
        # Checking is_valid takes ~0.01s?
        # 1000 iterations * 0.01s = 10s.
        
        # Use a faster validity check?
        # We can maintain the set incrementally?
        # Removing w[i] affects windows i-n+1 to i.
        # It's complex to maintain.
        # Let's just do full check but limit iterations.
        
        random.seed(42)
        current_word = word
        for _ in range(2000):
            if time.time() > deadline - 5.0:
                break
            if len(current_word) <= n:
                break
            # Pick a random index to remove
            # Avoid ends to keep structure?
            idx = random.randint(1, len(current_word) - 2)
            new_word = current_word[:idx] + current_word[idx+1:]
            if is_valid(new_word):
                current_word = new_word
        word = current_word

    return "".join(map(str, word))
