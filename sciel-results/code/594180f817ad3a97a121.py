import time
import itertools

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Generate the standard recursive superpermutation
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
            # Find max overlap
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Convert to list of ints for easier manipulation
    # The seed returns string, but we work with list of ints
    
    # 2. Check validity function
    def is_valid(seq):
        if len(seq) < n:
            return False
        perms_found = set()
        for i in range(len(seq) - n + 1):
            window = tuple(seq[i:i + n])
            if len(set(window)) == n: # Only count if it's a permutation
                perms_found.add(window)
            if len(perms_found) == int(math.factorial(n)):
                return True
        return len(perms_found) == int(math.factorial(n))
        
    import math
    target_perms = int(math.factorial(n))
    
    # Optimization: Pre-compute all permutations? 
    # For n=7, 5040 perms. 
    # Checking validity by scanning the whole string is O(L * n).
    # L ~ 6000. 6000*7 = 42000 ops. Very fast.
    
    # 3. Greedy Removal
    # We try to remove each character one by one.
    # To be efficient, we can try to remove from left to right, or random order.
    # Removing a char shifts indices.
    
    # A better heuristic: 
    # The recursive construction has a lot of redundancy.
    # We can try to remove characters that are not "critical" for any unique permutation window.
    
    # Simple greedy:
    # While we have time:
    #   For each index i:
    #     Try remove S[i]
    #     If valid, remove it.
    #     Else, keep it.
    #   If no removals in a full pass, stop.
    
    # To speed up validity check, we can maintain a count of each permutation.
    # But since we are doing it in Python, and n is small, the simple check might suffice.
    
    # Let's implement a slightly smarter removal:
    # We know that in the recursive construction, the "core" permutations are essential.
    
    # We'll use a list for the word.
    w = word[:]
    
    # Helper to check if a specific permutation is still present after removal?
    # Too complex. Just re-check whole string.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing from end to start to keep indices stable? 
        # Or just iterate and if removed, restart the pass?
        # Restarting is safer.
        for i in range(len(w) - 1, -1, -1):
            if time.time() > deadline:
                break
            # Try removing w[i]
            new_w = w[:i] + w[i+1:]
            if is_valid(new_w):
                w = new_w
                improved = True
                # Restart loop? Or continue?
                # If we continue, indices are shifted. 
                # Since we are iterating backwards, removing w[i] doesn't affect indices < i.
                # So we can continue to i-1.
                pass
        
    # Final check
    if not is_valid(w):
        # Fallback to seed if something went wrong (should not happen)
        return "".join(map(str, word))
        
    return "".join(map(str, w))
