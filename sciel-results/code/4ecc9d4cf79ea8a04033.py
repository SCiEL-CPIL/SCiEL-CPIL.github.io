import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # 1. Greedy earliest-insertion superpermutation
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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    best_overlap = o
                    break
            new += block[best_overlap:]
        word = new

    S = list(word)
    L = len(S)
    
    # 2. Initialize counts
    counts = {}
    for i in range(L - n + 1):
        p = tuple(S[i:i + n])
        counts[p] = counts.get(p, 0) + 1
        
    best_S = S[:]
    best_len = len(S)
    
    # Helper to get window tuple at index i
    def get_window(i):
        return tuple(S[i:i+n])
        
    def try_remove(i):
        """Try to remove S[i]. Returns True if successful."""
        if i < 0 or i >= len(S):
            return False
            
        start = max(0, i - n + 1)
        end = min(i, len(S) - n)
        
        # Check validity: no window containing i has count 1
        for j in range(start, end + 1):
            p = get_window(j)
            if counts.get(p, 0) == 1:
                return False
        
        # Perform removal
        # 1. Decrement counts for destroyed windows
        for j in range(start, end + 1):
            p = get_window(j)
            c = counts[p] - 1
            if c == 0:
                del counts[p]
            else:
                counts[p] = c
        
        # 2. Increment counts for new straddling windows
        # New string length is L-1.
        # New windows that change are those crossing the gap at i.
        # In the new string, indices < i are same, indices >= i are shifted left by 1.
        # A window starting at k in the new string is different if it contains the gap.
        # Gap is between old i-1 and old i+1.
        # New window k uses S'[k..k+n-1].
        # S'[idx] = S[idx] if idx < i, else S[idx+1].
        # The window is changed if it includes any index >= i in the new string that maps to >= i+1 in old?
        # Actually, the content changes for windows that span the removal point.
        # These are windows k such that k < i < k+n.
        # i.e., k in [i-n+1, i-1].
        # Also, we must ensure k is a valid start index in the new string.
        # New length L' = L-1. Valid k in [0, L'-n].
        
        new_start = max(0, i - n + 1)
        new_end = min(i - 1, len(S) - 1 - n) # len(S) is still L here, so L-1-n
        
        for k in range(new_start, new_end + 1):
            # Construct new window
            wc = []
            for idx in range(k, k + n):
                if idx < i:
                    wc.append(S[idx])
                else:
                    wc.append(S[idx + 1])
            p_new = tuple(wc)
            counts[p_new] = counts.get(p_new, 0) + 1
            
        S.pop(i)
        return True

    def perform_swap(i, j):
        """Swap S[i] and S[j] and update counts incrementally."""
        if i == j:
            return
        if i > j:
            i, j = j, i
            
        # Identify affected windows
        # Windows affected by i: [i-n+1, i]
        # Windows affected by j: [j-n+1, j]
        # Union of these ranges.
        
        # Decrement counts for old windows in the union
        # We must be careful not to double-decrement if ranges overlap.
        # Let's collect unique start indices.
        starts = set()
        for k in range(max(0, i - n + 1), min(i, len(S) - n) + 1):
            starts.add(k)
        for k in range(max(0, j - n + 1), min(j, len(S) - n) + 1):
            starts.add(k)
            
        # Store old windows
        old_windows = {k: get_window(k) for k in starts}
        
        # Decrement counts
        for k, p in old_windows.items():
            counts[p] -= 1
            if counts[p] == 0:
                del counts[p]
                
        # Perform swap
        S[i], S[j] = S[j], S[i]
        
        # Increment counts for new windows
        # The set of affected start indices is the same (ranges depend on indices i,j which didn't change)
        for k in starts:
            p_new = get_window(k)
            counts[p_new] = counts.get(p_new, 0) + 1

    # ILS Loop
    stuck_rounds = 0
    max_stuck = 50 # Increased stuck tolerance for better exploration
    
    while time.time() < deadline:
        # Removal Pass
        removed_any = False
        i = len(S) - 1
        while i >= 0 and time.time() < deadline:
            if try_remove(i):
                removed_any = True
                # After removal, indices shift. We are at i.
                # The element at i is gone. The next element is at i (old i+1).
                # We should check the same index i again, as the new element there might be removable.
                # However, try_remove checks validity.
                # If we don't decrement i, we check the new S[i].
                pass
            else:
                i -= 1
                
        if removed_any:
            if len(S) < best_len:
                best_S = S[:]
                best_len = len(S)
            stuck_rounds = 0
        else:
            stuck_rounds += 1
            if stuck_rounds > max_stuck:
                break
            
            # Perturb: Swap two random elements
            if len(S) > n:
                i, j = random.sample(range(len(S)), 2)
                perform_swap(i, j)
                # No full rebuild needed!
                
    # Final Validation
    def is_valid(S):
        seen = set()
        for i in range(len(S) - n + 1):
            seen.add(tuple(S[i:i + n]))
        expected = 1
        for k in range(1, n + 1):
            expected *= k
        return len(seen) == expected

    if not is_valid(best_S):
        return "".join(map(str, word))

    return "".join(map(str, best_S))
