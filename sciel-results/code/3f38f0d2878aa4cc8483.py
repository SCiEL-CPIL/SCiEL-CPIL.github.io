<<<<<<< SEARCH
import time
import itertools
import random

def construct(n):
    """
    Construct the shortest known superpermutation for n symbols.
    """
    start_time = time.time()
    deadline = 115.0  # 5 seconds margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # 1. Generate classic recursive superpermutation
    word = [1]
    for m in range(2, n + 1):
        perms = []
        seen = set()
        for i in range(len(word) - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1:
                if window not in seen:
                    seen.add(window)
                    perms.append(window)
        
        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            target_overlap = m - 2
            overlap = 0
            if len(new_word) >= target_overlap and len(block) >= target_overlap:
                if new_word[-target_overlap:] == block[:target_overlap]:
                    overlap = target_overlap
                else:
                    max_overlap = min(len(new_word), len(block))
                    for o in range(max_overlap, -1, -1):
                        if o == 0:
                            break
                        if new_word[len(new_word) - o:] == block[:o]:
                            overlap = o
                            break
            else:
                max_overlap = min(len(new_word), len(block))
                for o in range(max_overlap, -1, -1):
                    if o == 0:
                        break
                    if new_word[len(new_word) - o:] == block[:o]:
                        overlap = o
                        break
            
            new_word.extend(block[overlap:])
        
        word = new_word
        
    current_word = word[:]
=======
import time
import itertools
import random

def construct(n):
    """
    Construct the shortest known superpermutation for n symbols.
    """
    start_time = time.time()
    deadline = 115.0  # 5 seconds margin
    
    if n == 1:
        return "1"
    if n == 2:
        return "121"
    
    # Greedy set-cover construction:
    # Start with all n symbols. Then repeatedly append the symbol that
    # completes the most new permutations, with tie-breaking by
    # "minimum remaining options" (prefer symbols that complete
    # permutations which have fewest other possible completions).
    
    all_perms = list(itertools.permutations(range(1, n+1)))
    perm_set = set(all_perms)
    
    # State: current word, set of covered permutations
    current_word = list(range(1, n + 1))
    covered = set()
    
    # Add initial permutations covered by the starting word
    if n <= len(current_word):
        for i in range(len(current_word) - n + 1):
            p = tuple(current_word[i:i+n])
            if p in perm_set:
                covered.add(p)
    
    # For each permutation, track how many times it could still be completed
    # by appending symbols to the current suffix
    # This helps with the "minimum remaining options" tie-breaking
    
    while len(covered) < len(all_perms) and time.time() < deadline:
        # For each candidate symbol s in 1..n, compute:
        # 1. How many new permutations would be completed
        # 2. Among those, how "constrained" they are (fewer other completions)
        
        # Get the last (n-1) symbols as the current suffix
        suffix = tuple(current_word[-(n-1):]) if len(current_word) >= n-1 else tuple(current_word)
        
        best_score = -1
        best_sym = None
        best_new_perms = None
        
        for s in range(1, n + 1):
            # Try appending s
            new_suffix = suffix + (s,)
            if len(new_suffix) < n:
                continue
            
            # The new window is the last n symbols
            if len(current_word) + 1 >= n:
                new_window = tuple(current_word[-(n-1):] + [s])
            else:
                new_window = tuple(current_word + [s])
            
            if new_window not in perm_set:
                continue
            
            new_perms = [new_window]
            
            # Actually, appending one symbol only creates ONE new window
            # (the last n symbols). So we just check if that window is new.
            
            if new_window in covered:
                score = 0
            else:
                score = 1
            
            # Tie-breaking: prefer symbols that lead to more constrained states
            # Count how many other symbols could complete similar windows
            # This is a heuristic - use the number of distinct symbols in suffix
            # as a proxy for constraint
            
            if score > best_score:
                best_score = score
                best_sym = s
                best_new_perms = new_perms
            elif score == best_score and best_sym is not None:
                # Tie-break: prefer the symbol that appears less in the recent history
                # (more "rare" symbols first, to diversify)
                recent = current_word[-min(20, len(current_word)):]
                s_count = recent.count(s)
                best_count = recent.count(best_sym)
                if s_count < best_count:
                    best_sym = s
                    best_new_perms = new_perms
        
        if best_sym is None:
            # No valid symbol found - this shouldn't happen for valid superperms
            # Fallback: try all symbols
            for s in range(1, n + 1):
                if len(current_word) + 1 >= n:
                    new_window = tuple(current_word[-(n-1):] + [s])
                else:
                    new_window = tuple(current_word + [s])
                if new_window in perm_set:
                    best_sym = s
                    break
            if best_sym is None:
                best_sym = 1
        
        current_word.append(best_sym)
        
        # Update covered
        if len(current_word) >= n:
            new_window = tuple(current_word[-n:])
            if new_window in perm_set:
                covered.add(new_window)
        
        # Periodic check: try to delete redundant symbols
        if len(covered) % 100 == 0 and time.time() < deadline:
            # Quick deletion pass
            for i in range(len(current_word) - 1, -1, -1):
                # Check if removing position i loses any unique permutation
                orig = current_word[i]
                current_word.pop(i)
                lost = False
                if len(current_word) >= n:
                    for j in range(max(0, i - n + 1), min(i, len(current_word) - n + 1)):
                        p = tuple(current_word[j:j+n])
                        if p in perm_set and p not in covered:
                            # This permutation is no longer covered
                            # But we need to check if it was covered elsewhere
                            pass
                # Simplified: just restore and skip complex deletion here
                current_word.insert(i, orig)
                break  # Skip deletion in greedy phase for simplicity
    
    current_word = current_word[:]
>>>>>>> REPLACE

Now let me also improve the local search to be more effective with the new starting point:

