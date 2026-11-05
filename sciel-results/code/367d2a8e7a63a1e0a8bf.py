import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin before 120s limit
    
    # 1. Generate standard recursive superpermutation
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
        
    # Convert to list of ints for easier manipulation
    S = list(word)
    L = len(S)
    
    # 2. Initialize counts using a dict
    counts = {}
    for i in range(L - n + 1):
        p = tuple(S[i:i+n])
        counts[p] = counts.get(p, 0) + 1
        
    best_S = S[:]
    best_len = L
    
    # Precompute for speed: use local variables
    def try_remove(i):
        """Try to remove position i. Returns True if successful. Modifies S and counts."""
        nonlocal S, counts, best_S, best_len, L
        # Calculate affected old windows
        start_old = max(0, i - n + 1)
        end_old = min(i, L - n)
        
        # Check if removal is valid (no count drops to 0)
        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j+n])
            if counts.get(p, 0) == 1:
                return False
        
        # Perform removal - decrement old windows
        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j+n])
            c = counts[p] - 1
            if c == 0:
                del counts[p]
            else:
                counts[p] = c
        
        # Calculate new straddling windows after removal
        # After removing S[i], positions >= i shift left by 1
        # New windows that straddle the gap: starting positions from max(0, i-n+1) to min(i-1, L-1-n)
        # For a new window starting at k (in the new array), it maps to:
        #   positions k..k+n-1 in new array
        #   which correspond to k..i-1 in old (before i) and i+1..i+(k+n-1-(i-1)) in old (after i)
        # Actually, let's think differently. After removal, the new array has length L-1.
        # A window starting at position k in the new array (0-indexed) covers new[k..k+n-1].
        # In terms of old array: if k+n-1 < i, it's entirely before i (same as old).
        # If k < i <= k+n-1, it straddles.
        # If k >= i, it's entirely after i (shifted left by 1).
        
        # The windows that change are those where the removal of position i affects them.
        # Old windows containing position i: start from max(0, i-n+1) to min(i, L-n)
        # After removal, new windows at positions max(0, i-n+1) to min(i-1, L-1-n)
        # that straddle the gap.
        
        # For new window starting at k (0 <= k <= L-1-n):
        #   if k + n - 1 < i: no change (entirely before removed pos)
        #   if k >= i: no change (entirely after, but shifted - however these weren't in old affected set)
        #   if k < i <= k + n - 1: straddles
        
        # The new straddling windows start at k where max(0, i-n+1) <= k <= min(i-1, L-1-n)
        start_new = max(0, i - n + 1)
        end_new = min(i - 1, L - 1 - n)
        
        for k in range(start_new, end_new + 1):
            # Build new window from S after removal
            # But we haven't removed yet! Let's build it from current S.
            # In current S, the new window starting at k would be:
            #   S[k], S[k+1], ..., S[i-1], S[i+1], S[i+2], ...
            #   up to n elements
            window_chars = []
            for idx in range(k, k + n):
                s_idx = idx if idx < i else idx + 1
                window_chars.append(S[s_idx])
            p_new = tuple(window_chars)
            counts[p_new] = counts.get(p_new, 0) + 1
        
        S.pop(i)
        L = len(S)
        best_S = S[:]
        best_len = L
        return True
    
    def try_swap(i, j):
        """Try to swap positions i and j. Returns True if successful."""
        nonlocal S, counts
        if i == j:
            return False
        if i > j:
            i, j = j, i
        
        # Collect all affected window positions
        affected = []
        for pos in [i, j]:
            s = max(0, pos - n + 1)
            e = min(pos, L - n)
            for k in range(s, e + 1):
                if k not in affected:
                    affected.append(k)
        
        old_i, old_j = S[i], S[j]
        
        # Check validity
        for k in affected:
            old_p = tuple(S[k:k+n])
            if counts.get(old_p, 0) == 1:
                # Check if this window changes
                changed = False
                for idx in range(k, k + n):
                    if idx == i and old_j != S[i]:
                        changed = True
                        break
                    elif idx == j and old_i != S[j]:
                        changed = True
                        break
                if changed:
                    return False
        
        # Perform swap
        for k in affected:
            old_p = tuple(S[k:k+n])
            c = counts[old_p] - 1
            if c == 0:
                del counts[old_p]
            else:
                counts[old_p] = c
        
        S[i], S[j] = S[j], S[i]
        
        for k in affected:
            new_p = tuple(S[k:k+n])
            counts[new_p] = counts.get(new_p, 0) + 1
        
        return True
    
    # 3. Local Search with removal and perturbation
    # Use a more efficient scan that doesn't rebuild the full index list
    while time.time() < deadline:
        removals_in_pass = 0
        
        # Forward pass
        i = 0
        while i < L:
            if time.time() > deadline:
                break
            if try_remove(i):
                removals_in_pass += 1
                # Don't increment i; the element at i is now the next one
            else:
                i += 1
        
        # Backward pass
        i = L - 1
        while i >= 0:
            if time.time() > deadline:
                break
            if try_remove(i):
                removals_in_pass += 1
                # Don't decrement i; the element at i is now the next one
            else:
                i -= 1
        
        if removals_in_pass == 0:
            # Stuck - try perturbation via swaps
            if time.time() > deadline:
                break
            
            # Try multiple swap strategies
            max_swaps = min(200, int((deadline - time.time()) * 100))
            found_improvement = False
            
            for _ in range(max_swaps):
                if time.time() > deadline:
                    break
                
                # Try adjacent swaps and random swaps
                if random.random() < 0.5 and L > 1:
                    i = random.randint(0, L - 2)
                    j = i + 1
                else:
                    i = random.randint(0, L - 1)
                    j = random.randint(0, L - 1)
                    if i == j:
                        continue
                
                if try_swap(i, j):
                    # After swap, try a quick removal scan
                    removed = False
                    for k in range(L - 1, -1, -1):
                        if try_remove(k):
                            removed = True
                            break
                    
                    if not removed:
                        # Revert the swap
                        try_swap(i, j)
                    else:
                        found_improvement = True
                        break
            
            if found_improvement:
                continue
            
            # If no improvement from swaps, try a more aggressive perturbation
            # or accept we're stuck and try a fresh restart with different seed
            if time.time() > deadline:
                break
            
            # Try a few more targeted swaps: swap elements that are part of unique windows
            # to create duplicates
            targeted_tries = 0
            for _ in range(50):
                if time.time() > deadline:
                    break
                targeted_tries += 1
                i = random.randint(0, L - 1)
                # Find a window containing i that is unique
                s = max(0, i - n + 1)
                e = min(i, L - n)
                unique_pos = None
                for k in range(s, e + 1):
                    p = tuple(S[k:k+n])
                    if counts.get(p, 0) == 1:
                        unique_pos = k
                        break
                if unique_pos is None:
                    continue
                
                # Swap i with a nearby position to try to break the unique window
                j = i + random.randint(-2, 2)
                if j < 0 or j >= L or j == i:
                    continue
                
                if try_swap(i, j):
                    removed = False
                    for k in range(L - 1, -1, -1):
                        if try_remove(k):
                            removed = True
                            break
                    if not removed:
                        try_swap(i, j)
                    else:
                        break
    
    return "".join(map(str, best_S))
