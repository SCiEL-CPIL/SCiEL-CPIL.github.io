import time
from itertools import permutations
from collections import defaultdict

def construct(n):
    deadline = time.time() + 140  # 10s safety margin

    # Classic recursive construction of length 1! + 2! + ... + n!
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
            best_overlap = 0
            max_check = min(len(new), len(block))
            for o in range(max_check, 0, -1):
                if new[-o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new

    if n <= 6:
        return "".join(map(str, word))

    # For n >= 7, apply local search to shorten the word
    all_perms = set(permutations(range(1, n + 1)))
    current = word[:]
    n_local = n
    L = len(current)
    
    # Build reference count map: window -> count of occurrences
    ref_count = defaultdict(int)
    for i in range(L - n_local + 1):
        ref_count[tuple(current[i:i + n_local])] += 1
    
    # Count how many required permutations are covered (ref_count >= 1)
    covered = sum(1 for p in all_perms if ref_count[p] > 0)
    total_perms = len(all_perms)
    
    # Verify classic construction is valid
    if covered < total_perms:
        return "".join(map(str, current))

    # Fast local search: try deleting each character using reference counting
    improved = True
    start_time = time.time()
    
    def try_delete(i):
        """Try to delete position i. Returns True if successful."""
        nonlocal current, L, num_windows, ref_count, improved
        L = len(current)
        num_windows = L - n_local + 1
        
        j_start = max(0, i - n_local + 1)
        j_end = min(i, num_windows - 1)
        
        can_delete = True
        old_windows = []
        for j in range(j_start, j_end + 1):
            w = tuple(current[j:j + n_local])
            old_windows.append(w)
            if ref_count.get(w, 0) == 1:
                if w in all_perms:
                    can_delete = False
                    break
        
        if not can_delete:
            return False
        
        # Perform deletion
        current.pop(i)
        L = len(current)
        num_windows = L - n_local + 1
        improved = True
        
        # Update reference counts
        for w in old_windows:
            ref_count[w] -= 1
            if ref_count[w] <= 0:
                del ref_count[w]
        
        k_start = max(0, i - n_local + 1)
        k_end = min(i - 1, L - 1 - n_local)
        
        for k in range(k_start, k_end + 1):
            new_w = tuple(current[k:k + n_local])
            ref_count[new_w] = ref_count.get(new_w, 0) + 1
        
        return True

    def try_swap(i, j):
        """Try to swap positions i and j. Returns True if successful."""
        nonlocal current, L, num_windows, ref_count, improved
        if i == j:
            return False
        if current[i] == current[j]:
            return False
        
        L = len(current)
        num_windows = L - n_local + 1
        
        # Windows affected by swapping positions i and j
        # A window starting at k is affected if i or j is in [k, k+n-1]
        # i is in window k iff k <= i <= k+n-1, i.e., i-n+1 <= k <= i
        # j is in window k iff j-n+1 <= k <= j
        
        # Collect all affected window starts
        affected_starts = set()
        for pos in [i, j]:
            ks = max(0, pos - n_local + 1)
            ke = min(pos, num_windows - 1)
            for k in range(ks, ke + 1):
                affected_starts.add(k)
        
        # Check if swap would cause any unique required permutation to be lost
        # For each affected window, check if its current value has count 1 and is required
        # After swap, that window will have a different value
        old_window_vals = {}
        for k in sorted(affected_starts):
            w = tuple(current[k:k + n_local])
            old_window_vals[k] = w
            if ref_count.get(w, 0) == 1 and w in all_perms:
                # This window would change, so the permutation w would be lost
                # unless the new window at k also equals w (impossible since we're swapping)
                # OR unless w appears elsewhere (but count is 1, so no)
                # So this swap is invalid
                return False
        
        # Perform swap
        current[i], current[j] = current[j], current[i]
        improved = True
        
        # Update reference counts for affected windows
        for k in sorted(affected_starts):
            old_w = old_window_vals[k]
            ref_count[old_w] -= 1
            if ref_count[old_w] <= 0:
                del ref_count[old_w]
            
            new_w = tuple(current[k:k + n_local])
            ref_count[new_w] = ref_count.get(new_w, 0) + 1
        
        return True

    # First pass: aggressive deletion
    L = len(current)
    while time.time() - start_time < 60:
        L = len(current)
        made_progress = False
        for i in range(L):
            if time.time() - start_time > 60:
                break
            if try_delete(i):
                made_progress = True
                L = len(current)
        if not made_progress:
            break
    
    # Second pass: swaps to enable further deletions
    while time.time() - start_time < 125:
        L = len(current)
        num_windows = L - n_local + 1
        made_progress = False
        
        # Try random swaps to break deadlocks
        import random
        random.seed(42)
        
        for _ in range(200):
            if time.time() - start_time > 125:
                break
            if L < n_local + 2:
                break
            i = random.randint(0, L - 1)
            j = random.randint(0, L - 1)
            if i == j or current[i] == current[j]:
                continue
            
            # Save state for rollback
            saved_current = current[:]
            saved_ref_count = dict(ref_count)
            saved_improved = improved
            
            if try_swap(i, j):
                # After swap, try to delete some characters
                L_after = len(current)
                deleted = 0
                for k in range(L_after):
                    if time.time() - start_time > 125:
                        break
                    if try_delete(k):
                        deleted += 1
                        L_after = len(current)
                
                if deleted > 0:
                    made_progress = True
                else:
                    # Rollback if no improvement
                    current = saved_current
                    ref_count = saved_ref_count
                    improved = saved_improved
        
        if not made_progress:
            # Try more aggressive random swaps
            any_progress = False
            for _ in range(100):
                if time.time() - start_time > 125:
                    break
                if L < n_local + 2:
                    break
                i = random.randint(0, L - 1)
                j = random.randint(0, L - 1)
                if i == j or current[i] == current[j]:
                    continue
                
                saved_current = current[:]
                saved_ref_count = dict(ref_count)
                saved_improved = improved
                
                if try_swap(i, j):
                    L_after = len(current)
                    deleted = 0
                    for k in range(L_after):
                        if time.time() - start_time > 125:
                            break
                        if try_delete(k):
                            deleted += 1
                            L_after = len(current)
                    
                    if deleted > 0:
                        any_progress = True
                    else:
                        current = saved_current
                        ref_count = saved_ref_count
                        improved = saved_improved
            
            if not any_progress:
                break
    
    return "".join(map(str, current))
