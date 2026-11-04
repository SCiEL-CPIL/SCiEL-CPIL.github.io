import time
import random

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
    
    # 2. Initialize counts
    counts = {}
    for i in range(len(S) - n + 1):
        p = tuple(S[i:i+n])
        counts[p] = counts.get(p, 0) + 1
        
    best_S = S[:]
    best_len = len(S)
    
    def try_remove(S, counts, i, n):
        """Try to remove position i. Returns True if successful."""
        # Calculate affected old windows
        start_old = max(0, i - n + 1)
        end_old = min(i, len(S) - n)
        
        # Check if removal is valid (no count drops to 0)
        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j+n])
            if counts.get(p, 0) == 1:
                return False
        
        # Perform removal
        for j in range(start_old, end_old + 1):
            p = tuple(S[j:j+n])
            counts[p] -= 1
            if counts[p] == 0:
                del counts[p]
        
        # Calculate new straddling windows
        start_new = max(0, i - n + 1)
        end_new = min(i - 1, len(S) - 1 - n)
        
        for k in range(start_new, end_new + 1):
            window_chars = []
            for idx in range(k, k+n):
                s_idx = idx if idx < i else idx + 1
                window_chars.append(S[s_idx])
            p_new = tuple(window_chars)
            counts[p_new] = counts.get(p_new, 0) + 1
        
        S.pop(i)
        return True
    
    def try_swap(S, counts, i, j, n):
        """Try to swap positions i and j. Returns True if successful."""
        if i == j:
            return False
        if i > j:
            i, j = j, i
        
        # Check if swap is valid
        # Windows affected: those containing i or j
        # We need to check that no permutation count drops to 0
        
        # Collect all affected window positions
        affected = set()
        for pos in [i, j]:
            s = max(0, pos - n + 1)
            e = min(pos, len(S) - n)
            for k in range(s, e + 1):
                affected.add(k)
        
        # Check validity: for each affected window, if it's unique and changes, invalid
        # First, save old values
        old_i, old_j = S[i], S[j]
        
        # Check each affected window
        for k in affected:
            old_p = tuple(S[k:k+n])
            if counts.get(old_p, 0) == 1:
                # This window is unique. Check if it changes after swap
                # Reconstruct the window after swap
                new_window = []
                for idx in range(k, k+n):
                    if idx == i:
                        new_window.append(old_j)
                    elif idx == j:
                        new_window.append(old_i)
                    else:
                        new_window.append(S[idx])
                new_p = tuple(new_window)
                if new_p != old_p:
                    return False
        
        # Perform swap
        for k in affected:
            old_p = tuple(S[k:k+n])
            counts[old_p] -= 1
            if counts[old_p] == 0:
                del counts[old_p]
        
        S[i], S[j] = S[j], S[i]
        
        for k in affected:
            new_p = tuple(S[k:k+n])
            counts[new_p] = counts.get(new_p, 0) + 1
        
        return True
    
    # 3. Local Search with removal and perturbation
    def do_removal_pass(S, counts, n, direction):
        """Perform a full removal pass. Returns number of removals."""
        removals = 0
        if direction == 'forward':
            i = 0
            while i < len(S):
                if time.time() > deadline:
                    break
                if try_remove(S, counts, i, n):
                    removals += 1
                    # Don't increment i since element shifted left
                else:
                    i += 1
        else:
            i = len(S) - 1
            while i >= 0:
                if time.time() > deadline:
                    break
                if try_remove(S, counts, i, n):
                    removals += 1
                    # Don't decrement i since element shifted right
                else:
                    i -= 1
        return removals
    
    def do_swap_perturbation(S, counts, n, num_swaps):
        """Try random swaps and attempt removals after each. Returns (S, counts, improved)."""
        improved = False
        for _ in range(num_swaps):
            if time.time() > deadline:
                break
            
            i = random.randint(0, len(S) - 1)
            j = random.randint(0, len(S) - 1)
            if i == j:
                continue
            
            if try_swap(S, counts, i, j, n):
                # After swap, try removals
                removed = do_removal_pass(S, counts, n, 'backward')
                if removed > 0:
                    improved = True
                else:
                    # Revert the swap
                    try_swap(S, counts, i, j, n)
        
        return S, counts, improved
    
    # Main search loop
    while time.time() < deadline:
        # Try removals in both directions
        total_removals = 0
        for direction in ['forward', 'backward']:
            if time.time() > deadline:
                break
            removals = do_removal_pass(S, counts, n, direction)
            total_removals += removals
        
        if total_removals > 0:
            best_S = S[:]
            best_len = len(S)
        
        if total_removals == 0:
            # Stuck - try perturbation
            if time.time() > deadline:
                break
            
            # Try several random swaps
            max_swaps = min(30, int((deadline - time.time()) * 100))
            if max_swaps > 0:
                S, counts, improved = do_swap_perturbation(S, counts, n, max_swaps)
                if improved:
                    best_S = S[:]
                    best_len = len(S)
    
    return "".join(map(str, best_S))
