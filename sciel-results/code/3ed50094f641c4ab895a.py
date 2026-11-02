import time
import sys

def construct(n):
    # Build the classic recursive superpermutation
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
            for o in range(min(len(new), len(block)), -1, -1):
                if o == 0 or new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    # For n <= 4, classic is known optimal, return directly
    if n <= 4:
        return "".join(map(str, word))
    
    # Greedy local search: try to remove redundant characters
    # Use time limit of ~110 seconds to stay safe within 120s
    start_time = time.time()
    time_limit = 110.0
    
    # Build window counts: map from permutation tuple -> count
    from collections import defaultdict
    window_counts = defaultdict(int)
    L = len(word)
    for i in range(L - n + 1):
        w = tuple(word[i:i + n])
        window_counts[w] += 1
    
    # Total permutations of n symbols
    import math
    total_perms = math.factorial(n)
    # Verify all permutations are present
    # (should be true for valid superpermutation)
    
    while time.time() - start_time < time_limit:
        found_removal = False
        L = len(word)
        # Try each position for removal
        for i in range(L):
            if time.time() - start_time > time_limit:
                break
            
            # Simulate removal of word[i]
            # Windows affected: those that include position i
            # A window starting at position j covers positions [j, j+n-1]
            # It includes position i if j <= i <= j+n-1
            # So j in [i-n+1, i], intersect with [0, L-n]
            
            old_windows = []
            j_start = max(0, i - n + 1)
            j_end = min(i, L - n)
            for j in range(j_start, j_end + 1):
                old_windows.append(j)
            
            # Check if any window would drop to 0
            valid = True
            for j in old_windows:
                w = tuple(word[j:j + n])
                if window_counts[w] == 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check if new windows created by the removal already exist
            # After removing word[i], positions shift:
            # New window starting at j (for j in some range) corresponds to
            # old positions [j, j+n-1] but with position i removed
            # Specifically, new windows are formed at positions where the gap closes
            # New window at position j covers new positions [j, j+n-1]
            # In old indexing, this is [j, j+i-1] and [j+i, j+n] if j < i < j+n-1
            # Actually, new windows appear at positions j where j <= i-1 and j+n-1 >= i
            # i.e., j in [i-n+1, i-1], and these correspond to old windows that had the gap
            # But we need to check if the new window (formed by concatenating old[j:i] + old[i+1:j+n]) exists
            
            # New windows: for j in range(max(0, i-n+1), min(i, L-n-1) + 1) if j < i
            # Wait, after removal, length is L-1. New windows start at j in [0, L-1-n]
            # A new window at position j (in new indexing) corresponds to:
            # - If j < i: old positions [j, i-1] + old positions [i+1, j+n]
            # - If j >= i: old positions [j+1, j+n]
            
            # The new windows to check are those at positions j where the window "spans" the removed position
            # In new indexing, position j corresponds to old positions:
            # new[j] = old[j] for j < i
            # new[j] = old[j+1] for j >= i
            # So new window at j covers new positions [j, j+n-1]:
            # old positions [j, min(j+n-1, i-1)] + old positions [max(j, i+1)-1, j+n-1] adjusted
            # More precisely:
            # If j + n - 1 < i: same as old window at j
            # If j >= i: same as old window at j+1
            # If j < i <= j+n-1: new window is old[j:i] + old[i+1:j+n] (note: j+n in old = j+n-1+1 = j+n)
            
            new_windows_to_check = []
            # Positions j in new indexing where the window spans the removed position
            # j < i and j+n-1 >= i => j >= i-n+1
            for j in range(max(0, i - n + 1), min(i, L - n) + 1):
                # New window at position j: old[j:i] + old[i+1:j+n]
                # Length: (i - j) + (j + n - 1 - i) = n-1... wait
                # old[j:i] has length i-j
                # old[i+1:j+n] has length (j+n) - (i+1) = j + n - i - 1
                # Total: i - j + j + n - i - 1 = n - 1. That's not right.
                # Let me reconsider.
                # New window at position j covers new indices j, j+1, ..., j+n-1
                # New index k corresponds to old index:
                #   old[k] if k < i
                #   old[k+1] if k >= i
                # So new window at j covers old indices:
                #   j, j+1, ..., min(j+n-1, i-1), i+1, i+2, ..., j+n-1 (if j+n-1 >= i)
                # The old indices are: j to min(j+n-1, i-1) and i+1 to j+n-1
                # But j+n-1 in old indexing... if j+n-1 >= i, then old indices go up to j+n-1
                # Wait, new index j+n-1 corresponds to old index (j+n-1) if j+n-1 < i, else old index (j+n-1)+1 = j+n
                # So old indices: [j, min(j+n-1, i-1)] union [i+1, j+n] if j+n-1 >= i
                # Length: min(j+n-1, i-1) - j + 1 + (j+n - (i+1) + 1) if j+n-1 >= i
                # = (i-1-j+1) + (j+n-i-1+1) = (i-j) + (j+n-i) = n. Good.
                # If j+n-1 < i, it's just old window at j.
                
                # So for j in [i-n+1, i-1] (and j >= 0, j <= L-1-n in new indexing), 
                # the new window is formed from old[j:i] + old[i+1:j+n]
                # But we need j+n-1 >= i, i.e., j >= i-n+1
                # And in new indexing, j must satisfy j <= (L-1) - n = L - n - 1
                # In old indexing, j ranges over valid start positions
                
                # Let's just construct the new window
                if j < i and j + n - 1 >= i:
                    # New window spans the removed position
                    new_w = tuple(word[j:i] + word[i+1:i+1 + (j + n - i)])
                    # Check if this permutation is already covered
                    if window_counts.get(new_w, 0) == 0:
                        # This new window is a new permutation not yet covered
                        # But wait, we need to check if it's already in window_counts
                        # If it's 0, it means this permutation wasn't present, so removing i would make it present but it wasn't before
                        # Actually, we need all permutations to be present. If new_w was not present before, 
                        # that's fine as long as all originally present permutations remain.
                        # But actually, we started with all permutations present. After removal, some may drop to 0.
                        # We already checked that no old window drops to 0.
                        # The new windows are just combinations; they don't need to be "new" permutations.
                        # The key check is: after removal, every permutation still has count >= 1.
                        # We checked old windows don't drop to 0. New windows are formed from existing elements,
                        # so they don't introduce "missing" permutations in the sense that we started with all present.
                        # Actually, I'm overcomplicating. The check should be:
                        # After removal, for every permutation P, count_P >= 1.
                        # We verified that no window that existed with count 1 disappears.
                        # New windows are just additional occurrences, they can only increase counts.
                        # So the only risk is losing windows with count 1.
                        # Therefore, the check "no old window drops to 0" is sufficient!
                        pass
            
            # The check is simply: no old window (that includes position i) has count 1.
            # If that's the case, removal is safe.
            
            # Perform the removal
            removed = word[i]
            word.pop(i)
            L -= 1
            
            # Update window counts
            # Decrement counts for old windows that included position i
            for j in old_windows:
                # The old window at j in the old word. But after removal, some of these windows change.
                # We need to decrement the count for each old window that no longer exists as-is.
                # Old windows at positions j in old_windows: these windows included position i, so they are gone.
                # But some may reappear as new windows.
                # Simpler: rebuild affected window counts.
                # Old windows at j in old_windows: decrement their counts
                for j in old_windows:
                    # In the old word, window at j is tuple(word_old[j:j+n])
                    # But we already removed word[i], so we need to use the old word.
                    # Let's store old windows before removal.
                    pass
            
            # Let me redo this more carefully by storing old windows
            # Actually, let's revert and do it properly
            
            word.insert(i, removed)
            L = len(word)
            
            # Redo: store old windows and new windows
            old_w_tuples = []
            for j in old_windows:
                old_w_tuples.append(tuple(word[j:j + n]))
            
            # Compute new windows that replace them
            # After removal, new length is L-1
            # New windows at positions j in [max(0, i-n+1), min(i-1, L-1-n)] (in new indexing)
            # where the window spans the removed position
            new_w_tuples = []
            for j in range(max(0, i - n + 1), min(i, L - n) + 1):
                if j < i and j + n - 1 >= i:
                    # New window at new position j
                    new_w = tuple(word[j:i] + word[i+1:j+n])
                    new_w_tuples.append(new_w)
            
            # Also, windows after position i shift: old window at j+1 becomes new window at j for j >= i
            # But those don't change their content, just their position. So no count change needed.
            
            # Decrement counts for old windows
            for w in old_w_tuples:
                window_counts[w] -= 1
                if window_counts[w] <= 0:
                    # This shouldn't happen if our check was correct
                    pass
            
            # Increment counts for new windows
            for w in new_w_tuples:
                window_counts[w] += 1
            
            found_removal = True
            break
        
        if not found_removal:
            break
    
    return "".join(map(str, word))
