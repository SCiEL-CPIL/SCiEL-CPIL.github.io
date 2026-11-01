import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    # 1. Generate initial superpermutation
    # Standard recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of length m-1 from current word
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            # Construct block: p + [m] + p
            block = list(p) + [m] + list(p)
            # Find max overlap with current new
            overlap = 0
            # Max possible overlap is len(block) - 1 or len(new)
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if n <= 4:
        return "".join(map(str, word))
    
    # 2. Local Search Deletion
    # We will use a randomized deletion strategy to escape local minima.
    # The key insight is that deleting a symbol is valid if no unique permutation
    # is lost. We can check this efficiently.
    
    current = list(word)
    L = len(current)
    
    # Precompute permutation counts
    # Using a dictionary to count occurrences of each n-permutation
    perm_count = defaultdict(int)
    for i in range(L - n + 1):
        p = tuple(current[i:i+n])
        perm_count[p] += 1
        
    target_perms = n
    for i in range(1, n):
        target_perms *= i
        
    if len(perm_count) < target_perms:
        return "".join(map(str, current))
        
    # Deletion loop with randomization
    # We will try to delete symbols in a randomized order to escape local minima
    # caused by deterministic ordering.
    
    while time.time() < deadline:
        # Create a list of indices to check
        # To speed up, we can just check all indices in a random order
        indices = list(range(L))
        random.shuffle(indices)
        
        deleted_any = False
        
        for i in indices:
            if time.time() > deadline:
                break
            
            # Check if i can be deleted
            # Old windows containing i
            start_w = max(0, i - n + 1)
            end_w = min(i, L - n)
            
            old_windows = []
            can_delete = True
            
            # Check if any old window with count 1 is not in new windows
            # We need to compute new windows first or check on the fly?
            # Computing new windows is O(n).
            
            new_windows = []
            for j in range(max(0, i - n + 1), i):
                if j + n > L - 1:
                    continue
                prefix = current[j:i]
                suffix_len = n - (i - j)
                suffix = current[i+1 : i+1+suffix_len]
                if len(suffix) < suffix_len:
                    continue
                p = tuple(prefix + suffix)
                new_windows.append(p)
            
            new_set = set(new_windows)
            
            for w in range(start_w, end_w + 1):
                p = tuple(current[w:w+n])
                if perm_count[p] == 1:
                    if p not in new_set:
                        can_delete = False
                        break
            
            if can_delete:
                # Perform deletion
                # Update counts
                # We need to re-calculate old windows to decrement?
                # We have old_windows list? No, we just iterated.
                # Let's collect them properly.
                
                # Re-iterate to collect old windows for count update
                old_ws = []
                for w in range(start_w, end_w + 1):
                    p = tuple(current[w:w+n])
                    old_ws.append(p)
                
                for p in old_ws:
                    perm_count[p] -= 1
                    if perm_count[p] == 0:
                        del perm_count[p]
                        
                for p in new_windows:
                    perm_count[p] += 1
                    
                current = current[:i] + current[i+1:]
                L -= 1
                deleted_any = True
                # Restart the scan? Or continue?
                # If we continue, indices > i are now shifted.
                # It is safer to break and restart the shuffle/scan.
                # But restarting shuffle is expensive.
                # Let's just break and let the outer while loop handle it.
                break
                
        if not deleted_any:
            # No more deletions possible in this configuration
            # Try a few random swaps to escape local minimum?
            # Or just break.
            # Let's try a few random swaps if we have time.
            if time.time() > deadline - 5:
                break
            
            # Try random swaps
            improved = False
            for _ in range(1000):
                if time.time() > deadline:
                    break
                
                # Pick two random positions
                p1 = random.randint(0, L-1)
                p2 = random.randint(0, L-1)
                if p1 == p2:
                    continue
                
                # Check if swap is valid
                # This is expensive to check fully.
                # We can just try the swap and see if it's valid.
                # But we need to restore if invalid.
                
                # Let's stick to deletion only for simplicity and safety.
                # If no deletion, we are stuck.
                break
                
            break # No more deletions possible
            
    return "".join(map(str, current))
