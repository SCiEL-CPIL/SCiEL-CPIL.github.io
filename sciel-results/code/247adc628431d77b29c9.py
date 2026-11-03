import time
import sys
from itertools import permutations

def construct(n):
    if n == 1:
        return "1"
    
    # Standard construction
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
            overlap = 0
            max_overlap = min(len(new), len(block))
            for o in range(max_overlap, 0, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    if n <= 6:
        return "".join(map(str, word))

    # Local search for n=7
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    L = len(word)
    counts = [0] * num_perms
    for i in range(L - n + 1):
        p = tuple(word[i:i+n])
        pid = perm_to_id.get(p)
        if pid is not None:
            counts[pid] += 1
            
    if any(c == 0 for c in counts):
        return "".join(map(str, word))

    deadline = time.time() + 110  # Safety margin

    # Helper to update counts when deleting index i
    # Returns (valid, old_pids, new_pids)
    def check_deletion(word, i, n, counts, perm_to_id):
        L = len(word)
        if i < 0 or i >= L:
            return False, [], []
            
        # Old windows covering i
        # Start k in [max(0, i-n+1), min(i, L-n)]
        start_k_old = max(0, i - n + 1)
        end_k_old = min(i, L - n)
        
        old_pids = []
        for k in range(start_k_old, end_k_old + 1):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                old_pids.append(pid)
                
        # New windows in the word after deletion
        # The gap is between i-1 and i (in new word indices, the element at i is gone)
        # New word length L-1
        # Windows crossing the gap start at k_new in [max(0, i-n), min(i-1, L-1-n)]
        # In the new word, the elements are word[0..i-1] + word[i+1..L-1]
        # A window starting at k_new in the new word corresponds to:
        # If k_new < i: elements are word[k_new..k_new+n-1] but with a shift for indices >= i
        # Actually, easier to construct the tuple directly.
        
        start_k_new = max(0, i - n)
        end_k_new = min(i - 1, L - 1 - n)
        
        new_pids = []
        for k_new in range(start_k_new, end_k_new + 1):
            # Construct window
            # Part 1: word[k_new : i]
            # Part 2: word[i+1 : k_new+n+1]
            w1 = word[k_new:i]
            w2 = word[i+1:k_new+n+1]
            p = tuple(w1 + w2)
            pid = perm_to_id.get(p)
            if pid is not None:
                new_pids.append(pid)
                
        # Check validity
        # For each pid in union of old and new, check if count remains >= 1
        touched = set(old_pids) | set(new_pids)
        
        # We can optimize this by using a dictionary for counts of touched pids
        # But since |touched| is small, we can just iterate.
        
        # However, doing .count() on lists is O(N).
        # Let's use a dictionary for the deltas.
        deltas = {}
        for pid in old_pids:
            deltas[pid] = deltas.get(pid, 0) - 1
        for pid in new_pids:
            deltas[pid] = deltas.get(pid, 0) + 1
            
        for pid in touched:
            if counts[pid] + deltas.get(pid, 0) < 1:
                return False, [], []
                
        return True, old_pids, new_pids

    # Perform deletion passes
    # Try both left-to-right and right-to-left to escape local minima
    for direction in ['l2r', 'r2l']:
        while time.time() < deadline:
            improved = False
            if direction == 'l2r':
                range_fn = lambda L: range(L)
            else:
                range_fn = lambda L: range(L - 1, -1, -1)
                
            # We need to be careful with indices if we delete.
            # If we delete at i, the next index is i (which is now the old i+1).
            # If we are going l2r, we don't increment i after deletion.
            # If we are going r2l, we don't decrement i after deletion (since we are going down).
            
            L = len(word)
            if direction == 'l2r':
                i = 0
                while i < L and time.time() < deadline:
                    valid, old_pids, new_pids = check_deletion(word, i, n, counts, perm_to_id)
                    if valid:
                        word.pop(i)
                        for pid in old_pids:
                            counts[pid] -= 1
                        for pid in new_pids:
                            counts[pid] += 1
                        improved = True
                        L -= 1
                        # Do not increment i
                    else:
                        i += 1
            else:
                i = L - 1
                while i >= 0 and time.time() < deadline:
                    valid, old_pids, new_pids = check_deletion(word, i, n, counts, perm_to_id)
                    if valid:
                        word.pop(i)
                        for pid in old_pids:
                            counts[pid] -= 1
                        for pid in new_pids:
                            counts[pid] += 1
                        improved = True
                        # Do not decrement i (the next element shifts into i)
                    else:
                        i -= 1
                        
            if not improved:
                break
                
    return "".join(map(str, word))
