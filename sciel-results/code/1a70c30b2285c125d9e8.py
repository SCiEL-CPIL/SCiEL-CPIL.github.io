import time
import random
import math

def construct(n):
    if n <= 1:
        return "1"
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit

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
            # Find maximum overlap
            max_check = min(len(new), len(block))
            overlap = 0
            for o in range(max_check, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
        
    S = list(word)
    L = len(S)
    
    # 2. Initialize counts
    counts = {}
    for i in range(L - n + 1):
        p = tuple(S[i:i+n])
        counts[p] = counts.get(p, 0) + 1
        
    best_S = S[:]
    best_len = L
    
    # 3. Local Search: Deletion
    # We will try to delete characters.
    # To handle index shifts and avoid bias, we can iterate from end to start.
    # If we delete S[i], indices > i shift left. Indices < i are unchanged.
    # Iterating from L-1 down to 0 allows us to continue without re-scanning shifted indices immediately?
    # Actually, if we delete i, the element at i+1 moves to i.
    # If we are iterating downwards, we have already processed i+1.
    # So we should be careful.
    # Standard approach: iterate i from 0 to L-1. If delete, L decreases, i stays same (points to next element).
    # Or iterate i from L-1 down to 0. If delete, L decreases. i-1 is next.
    
    # Let's use a random restart strategy with deletion scans.
    # Each scan goes from end to start.
    
    max_passes = 200 # Adjust based on speed
    pass_count = 0
    stuck_passes = 0
    
    while time.time() < deadline:
        pass_count += 1
        improved = False
        
        # Iterate from end to start to make index management easier
        # If we delete i, the new element at i is old i+1.
        # We continue to i-1.
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Check validity
            # Old windows affected: start = i-n+1, end = i
            # Note: window start index j must be >= 0 and j+n <= len(S)
            
            start_old = max(0, i - n + 1)
            end_old = min(i, len(S) - n)
            
            # If no old windows are affected, deletion is trivially valid?
            # No, if i is near edges, fewer windows are affected.
            # But we must check all windows that contain S[i].
            
            valid = True
            old_perms = []
            
            for j in range(start_old, end_old + 1):
                p = tuple(S[j:j+n])
                if counts.get(p, 0) == 1:
                    valid = False
                    break
                old_perms.append(p)
            
            if not valid:
                continue
            
            # If valid, calculate new windows
            # New string length is len(S) - 1
            # New windows that change are those that span the gap at i.
            # In the new string, the gap is at i.
            # Windows starting at k in [i-n+1, i-1] in the new string?
            # Let's map new index k to old indices.
            # New window k uses S_new[k:k+n].
            # S_new[m] = S[m] if m < i, else S[m+1].
            
            # The windows that are *new* (didn't exist in S) are those that combine chars from <i and >i.
            # These correspond to old windows that had S[i] in them?
            # No, the old windows are removed. The new windows replace them.
            # Specifically, for each k in [i-n+1, i-1] (in new string coordinates):
            # The window covers new indices k ... k+n-1.
            # Mapping to S:
            # If k+n-1 < i: all < i. (Unchanged content, but these are not the "new" ones, they are just shifted? No, if k+n-1 < i, the window is entirely before i. It was S[k:k+n]. It is unchanged. We don't need to update counts for it? 
            # Wait. The set of windows in the new string is different.
            # We removed windows containing S[i].
            # We added windows that bridge the gap.
            
            # Let's just re-calculate the counts for the affected region.
            # It is safer to:
            # 1. Decrement counts for old_perms.
            # 2. Increment counts for the new_perms formed by the bridge.
            
            # Bridge windows:
            # In the new string, a window is "bridging" if it includes indices < i and >= i.
            # Let new window start at k.
            # It bridges if k < i and k+n-1 >= i.
            # So k > i-n and k <= i-1.
            # Range of k: [max(0, i-n+1), min(i-1, len(S)-1-n)]
            
            start_new = max(0, i - n + 1)
            end_new = min(i - 1, len(S) - 1 - n) # Max start index in new string is (len(S)-1) - n + 1 = len(S) - n
            
            new_perms = []
            for k in range(start_new, end_new + 1):
                # Construct permutation from S
                # Indices in S:
                # m in [k, i-1] -> S[m]
                # m in [i, k+n-1] -> S[m+1]
                
                # Left part: S[k:i]
                # Right part: S[i+1 : i+1 + (k+n-1 - i + 1)] = S[i+1 : k+n]
                
                p_new = tuple(S[k:i] + S[i+1:k+n])
                new_perms.append(p_new)
            
            # Apply changes
            for p in old_perms:
                counts[p] -= 1
                if counts[p] == 0:
                    del counts[p]
            
            for p in new_perms:
                counts[p] = counts.get(p, 0) + 1
            
            # Perform deletion
            S.pop(i)
            improved = True
            
        if improved:
            stuck_passes = 0
            best_S = S[:]
            best_len = len(S)
        else:
            stuck_passes += 1
            # If stuck for a few passes, we might be in a local minimum.
            # We can try a random swap to escape, or just stop.
            # Given the time limit, simple deletion is often sufficient for the "easy" reductions.
            # If we want to be more aggressive, we can randomize the order of indices.
            if stuck_passes > 5:
                # Try a random restart of the scan order?
                # Or break.
                # Let's try a few random swaps to perturb the string, then continue deletion.
                # Swaps are risky (might make invalid), so we only accept swaps that keep validity.
                # This is complex. Let's just break if stuck.
                break

    return "".join(map(str, best_S))
