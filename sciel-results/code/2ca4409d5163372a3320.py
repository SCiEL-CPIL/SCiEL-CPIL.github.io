import itertools
from collections import Counter

# EVOLVE-BLOCK-START
def construct(n):
    # 1. Generate standard superpermutation
    def standard_superperm(n):
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
        return word

    W = standard_superperm(n)
    target_perms = set(itertools.permutations(range(1, n + 1)))
    
    # 2. Initialize window counts
    def get_windows(word):
        return [tuple(word[i:i+n]) for i in range(len(word) - n + 1)]
    
    windows = get_windows(W)
    window_counts = Counter(windows)
    
    # Check initial validity (should be true)
    # all(window_counts[p] > 0 for p in target_perms)
    
    max_passes = 100
    for _ in range(max_passes):
        removed_any = False
        i = 0
        L = len(W)
        while i < L:
            if L <= n:
                break
            
            # Indices of windows containing i in current W
            # Window start j: j <= i < j + n => i - n + 1 <= j <= i
            start_min = max(0, i - n + 1)
            start_max = min(i, L - n)
            
            affected_starts = list(range(start_min, start_max + 1))
            
            old_windows = [tuple(W[j:j+n]) for j in affected_starts]
            
            # Check if any old window is a target and count is 1
            # If we remove it, count becomes 0 -> invalid
            # But we might add a new window that is the same target?
            # The new windows are at the same start indices in the NEW array.
            # New window at start j (in new array) corresponds to W'[j:j+n].
            # Since W' is W with W[i] removed:
            # If j < i: W'[j:j+n] = W[j:i] + W[i+1:j+n+1]
            # If j >= i: W'[j:j+n] = W[j+1:j+n+1] (shifted)
            
            # We simulate the count change.
            # 1. Decrement old windows
            # 2. Increment new windows
            # 3. Check if any target perm count becomes 0
            
            new_windows = []
            # The new windows are at starts j in [start_min, start_max] in the NEW array.
            # Note: The range of starts in the new array is the same length.
            for j in affected_starts:
                # Construct new window content
                # In W (old), indices are 0..L-1.
                # W' index k corresponds to W index k if k < i, and k+1 if k >= i.
                # Window in W' at start j covers W' indices j .. j+n-1.
                # Corresponding W indices:
                # If j+n-1 < i: all < i. W[j..j+n-1]
                # If j >= i: all >= i. W[j+1..j+n]
                # Else: split. W[j..i-1] + W[i+1..j+n]
                
                w_new = []
                for k in range(j, j+n):
                    if k < i:
                        w_new.append(W[k])
                    else:
                        w_new.append(W[k+1])
                new_windows.append(tuple(w_new))
                
            # Check validity
            valid = True
            # Decrement old
            for w in old_windows:
                window_counts[w] -= 1
                if w in target_perms and window_counts[w] == 0:
                    # We might add it back in new_windows?
                    # So we can't say invalid yet.
                    pass
            
            # Increment new
            for w in new_windows:
                window_counts[w] += 1
                
            # Now check if any target perm is missing
            # We only need to check perms that were in old_windows or new_windows
            # Actually, just check if any target perm has count 0.
            # Optimization: Only check perms that were affected.
            affected_perms = set(old_windows + new_windows)
            affected_target_perms = [p for p in affected_perms if p in target_perms]
            
            for p in affected_target_perms:
                if window_counts[p] == 0:
                    valid = False
                    break
            
            if valid:
                # Commit
                W.pop(i)
                # Update windows list? 
                # We don't need to maintain the list, just counts and W.
                # But for next iteration, we need to recalculate windows?
                # No, the next iteration will recalculate affected windows.
                # However, the indices in W have changed.
                # The while loop uses i. If we pop i, the next element is at i.
                # So we don't increment i.
                removed_any = True
                L -= 1
            else:
                # Revert
                for w in old_windows:
                    window_counts[w] += 1
                for w in new_windows:
                    window_counts[w] -= 1
                i += 1
                
        if not removed_any:
            break
            
    return "".join(map(str, W))
# EVOLVE-BLOCK-END
