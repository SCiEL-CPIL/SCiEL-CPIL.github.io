import time
import random

def find_superperm_5(time_limit=10):
    """Find a short superpermutation for n=5 using greedy random restarts."""
    n = 5
    symbols = list(range(1, n + 1))
    best_len = float('inf')
    best_word = None
    start_time = time.time()
    
    # Precompute all permutations
    perms = [tuple(p) for p in __import__('itertools').permutations(symbols)]
    perm_set = set(perms)
    
    # Greedy function
    def greedy_start(start_perm):
        word = list(start_perm)
        # We need to contain all perms.
        # Track which perms are contained.
        # A perm is contained if it appears as a substring.
        # For efficiency, we can track the last n-1 symbols and check extensions.
        # But for n=5, it's small.
        
        # Simple approach: Build the word by appending one symbol at a time,
        # choosing the symbol that allows the most new permutations to be completed?
        # Or, choose the next permutation that has the best overlap.
        
        # Let's use the "next permutation" strategy.
        # Maintain a set of seen permutations.
        seen = set()
        # Add perms in current word
        for i in range(len(word) - n + 1):
            seen.add(tuple(word[i:i+n]))
        
        remaining = perm_set - seen
        if not remaining:
            return word
            
        # Heuristic: At each step, try to extend the word by 1 symbol such that
        # we complete a new permutation, or maximize overlap with a remaining perm.
        # Actually, the recursive construction appends blocks.
        # For n=5, we can just try to find a path in the de Bruijn graph of order 4?
        # No, let's just use a simple random greedy:
        # Pick a random remaining perm, try to append it with max overlap.
        
        while remaining:
            # Pick a random remaining perm? Or one with best overlap?
            # Best overlap:
            best_overlap = -1
            best_next = None
            # To speed up, don't check all remaining if time is tight.
            # For n=5, 120 perms is small.
            for p in list(remaining):
                p_list = list(p)
                # Find max overlap between suffix of word and prefix of p_list
                max_ov = 0
                for ov in range(min(len(word), len(p_list)), 0, -1):
                    if word[-ov:] == p_list[:ov]:
                        max_ov = ov
                        break
                if max_ov > best_overlap:
                    best_overlap = max_ov
                    best_next = p
            if best_next is None:
                # No overlap, just pick random
                best_next = random.choice(list(remaining))
                best_overlap = 0
                
            # Append best_next with overlap
            word.extend(best_next[best_overlap:])
            remaining.remove(best_next)
            # Update seen
            for i in range(len(word) - n + 1):
                seen.add(tuple(word[i:i+n]))
            # Remove newly seen from remaining
            for s in list(remaining):
                if s in seen:
                    remaining.remove(s)
                    
        return word

    while time.time() - start_time < time_limit:
        start_perm = random.sample(symbols, n)
        word = greedy_start(start_perm)
        if len(word) < best_len:
            best_len = len(word)
            best_word = word
            if best_len == 152: # Known optimal
                break
                
    return best_word

def construct(n):
    if n <= 4:
        # Standard recursive is optimal
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
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
        
    if n == 5:
        return "".join(map(str, find_superperm_5(10)))
        
    if n == 6 or n == 7:
        # Use best S5 as base
        s5 = find_superperm_5(10)
        word = list(s5)
        
        # Recursive extension
        for m in range(6, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))
        
    # Fallback for n > 7
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return "".join(map(str, word))
