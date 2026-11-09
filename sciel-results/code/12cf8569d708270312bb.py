import time
from itertools import permutations

def construct(n):
    # Classic recursive construction
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
        
    if n <= 6:
        return "".join(map(str, word))
        
    # For n=7, try to shorten
    if n == 7:
        # Precompute all permutations as a set of tuples for fast lookup
        all_perms = set(permutations(range(1, 8)))
        
        def is_valid(w):
            # Check if all permutations are present
            # Optimization: if len(w) < 7 + 5039, invalid? No, min length is 5040+6.
            # Just check the set of windows.
            windows = set()
            for i in range(len(w) - 6):
                windows.add(tuple(w[i:i+7]))
            # Check if all_perms is subset of windows
            # Since windows might contain duplicates or non-perms, we just check if all_perms is in windows
            # Actually, we just need to ensure every perm in all_perms appears in windows.
            # It's faster to check `if not all_perms.issubset(windows): return False`
            return all_perms.issubset(windows)

        start_time = time.time()
        current_word = list(word)
        
        # Greedy deletion loop
        improved = True
        while improved and (time.time() - start_time) < 100:
            improved = False
            L = len(current_word)
            for i in range(L):
                if (time.time() - start_time) > 110:
                    break
                # Try removing i
                candidate = current_word[:i] + current_word[i+1:]
                if is_valid(candidate):
                    current_word = candidate
                    improved = True
                    break # Restart scan from beginning after improvement
        
        return "".join(map(str, current_word))

    return "".join(map(str, word))
