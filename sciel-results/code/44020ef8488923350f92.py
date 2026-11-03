        if safe:
            # Store old windows
            old_windows = []
            for k in range(max(0, i - n + 1), i + 1):
                old_windows.append(tuple(w[k:k+n]))
            
            del w[i]
            
            # Decrement old
            for p in old_windows:
                if len(set(p)) == n:
                    perm_counts[p] -= 1
            
            # Increment new windows
            # New windows are formed at indices j such that the window crosses the gap.
            # The gap is at index i in the new string? No, the character at i is gone.
            # The characters at i-1 and i (old) are now adjacent.
            # The new windows are those starting at j where j < i and j+n-1 >= i (old index).
            # In the new string, these start at j (same index) and end at j+n-1.
            # The range of j is [i-n+1, i-1].
            for j in range(max(0, i - n + 1), i):
                p_new = tuple(w[j:j+n])
                if len(set(p_new)) == n:
                    perm_counts[p_new] = perm_counts.get(p_new, 0) + 1
