import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = 110  # 10s margin
    
    # 1. Generate standard superpermutation
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
    
    # 2. Precompute permutation mapping
    perms_list = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(perms_list)}
    num_perms = len(perms_list)
    
    # 3. Initialize counts
    counts = [0] * num_perms
    N = len(word)
    if N < n:
        return "".join(map(str, word))
        
    for i in range(N - n + 1):
        p = tuple(word[i:i + n])
        if p in perm_to_idx:
            counts[perm_to_idx[p]] += 1
    
    # 4. Greedy deletion
    # We try to delete characters.
    # To be efficient, we scan from left to right.
    # If we delete, we restart the scan.
    # Since few deletions are expected, this is fine.
    
    while True:
        if time.time() > deadline:
            break
            
        deleted = False
        N = len(word)
        if N < n:
            break
            
        for i in range(N):
            # Check if deleting word[i] is valid
            # Windows in old word containing i:
            j_start = max(0, i - n + 1)
            j_end = min(i, N - n)
            
            # Get old windows
            old_windows = []
            for j in range(j_start, j_end + 1):
                p = tuple(word[j:j + n])
                idx = perm_to_idx.get(p)
                if idx is not None:
                    old_windows.append(idx)
            
            # Check if any old window permutation has count 1
            # If so, we must ensure it appears in new windows
            must_check = []
            for idx in old_windows:
                if counts[idx] == 1:
                    must_check.append(idx)
            
            if must_check:
                # Construct new word locally to check new windows
                # word_new = word[:i] + word[i+1:]
                # New windows spanning the deletion:
                # In word_new, the junction is at i.
                # Windows start at k in [max(0, i-n+1), min(i, N-1-n)]
                k_start = max(0, i - n + 1)
                k_end = min(i, N - 1 - n)
                
                valid = True
                for idx in must_check:
                    p_to_find = perms_list[idx]
                    found = False
                    for k in range(k_start, k_end + 1):
                        # Construct window from word with i removed
                        # Window is word_new[k:k+n]
                        # This corresponds to:
                        # If k < i: word[k:i] + word[i+1:k+n+1]
                        # If k >= i: word[k+1:k+1+n] (shifted)
                        # Actually, easier to just slice the conceptual new list
                        # But creating the list is O(N).
                        # We can just check the tuple directly.
                        
                        # Let's just build the tuple
                        w_new = []
                        for x in range(k, k + n):
                            if x < i:
                                w_new.append(word[x])
                            else:
                                w_new.append(word[x + 1])
                        if tuple(w_new) == p_to_find:
                            found = True
                            break
                    if not found:
                        valid = False
                        break
                
                if not valid:
                    continue # Try next i
            
            # If we reach here, deletion is valid
            # Update counts
            for idx in old_windows:
                counts[idx] -= 1
            
            # Add new windows
            k_start = max(0, i - n + 1)
            k_end = min(i, N - 1 - n)
            for k in range(k_start, k_end + 1):
                w_new = []
                for x in range(k, k + n):
                    if x < i:
                        w_new.append(word[x])
                    else:
                        w_new.append(word[x + 1])
                p_new = tuple(w_new)
                idx_new = perm_to_idx.get(p_new)
                if idx_new is not None:
                    counts[idx_new] += 1
            
            # Perform deletion
            word = word[:i] + word[i + 1:]
            deleted = True
            break # Restart scan
        
        if not deleted:
            break

    return "".join(map(str, word))
