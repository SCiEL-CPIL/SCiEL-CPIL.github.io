import collections
import random

def construct(n):
    # Step 1: Generate the standard recursive superpermutation
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
            # Find maximal overlap
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # Step 2: Prune the word using randomized restarts to escape local minima
    # Target length is known to be 5905 for n=7.
    # We will try to prune as much as possible.
    
    def compute_counts(w):
        c = collections.Counter()
        for i in range(len(w) - n + 1):
            window = tuple(w[i:i+n])
            if len(set(window)) == n:
                c[window] += 1
        return c

    def prune_once(current_word, order='right'):
        """
        Performs one pass of greedy pruning.
        Returns the pruned word and a boolean indicating if any removal occurred.
        """
        if len(current_word) <= n:
            return current_word, False
        
        counts = compute_counts(current_word)
        L = len(current_word)
        to_remove = []
        
        # Determine iteration order
        if order == 'right':
            indices = range(L - 1, -1, -1)
        elif order == 'left':
            indices = range(L)
        else: # random
            indices = list(range(L))
            random.shuffle(indices)
            
        for i in indices:
            # Check if removing i is safe
            # Windows containing i are those starting at s where max(0, i-n+1) <= s <= min(i, L-n)
            s_min = max(0, i - n + 1)
            s_max = min(i, L - n)
            
            removable = True
            for s in range(s_min, s_max + 1):
                w = tuple(current_word[s:s+n])
                # Only care if it's a permutation
                if len(set(w)) == n:
                    if counts[w] <= 1:
                        removable = False
                        break
            
            if removable:
                to_remove.append(i)
                
        if not to_remove:
            return current_word, False
            
        # Remove indices in reverse sorted order to maintain index validity
        for i in sorted(to_remove, reverse=True):
            del current_word[i]
            
        return current_word, True

    # Multi-start / Randomized Restart Strategy
    # We keep the best word found so far.
    best_word = word[:]
    best_len = len(best_word)
    
    # Set a time limit or iteration limit.
    # 120s total. Pruning one pass is ~0.01s.
    # We can do ~1000 restarts easily.
    # However, we should just run until we stop improving or time runs out.
    # Given the small gap (5913 vs 5905), a few hundred restarts might suffice.
    
    import time
    start_time = time.time()
    max_time = 110.0 # Leave 10s for overhead
    
    while time.time() - start_time < max_time:
        current = word[:] # Reset to base recursive word
        # Try a few random orders or fixed orders in a cycle
        for _ in range(10): # 10 passes of random pruning
            # Randomly choose an order
            order = random.choice(['right', 'left', 'random'])
            current, changed = prune_once(current, order)
            if not changed:
                break
        
        if len(current) < best_len:
            best_len = len(current)
            best_word = current[:]
            # If we reached the known optimum (5905), we can stop early?
            # The task says "shortest known". We don't know if 5905 is the absolute minimum, 
            # but it's the target.
            if n == 7 and best_len <= 5905:
                break
                
    return "".join(map(str, best_word))
