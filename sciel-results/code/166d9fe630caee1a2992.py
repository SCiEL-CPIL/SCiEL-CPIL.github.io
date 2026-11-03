import time
import random
import itertools

def _classic_construct(n):
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
            # Find max overlap
            overlap = 0
            for o in range(min(len(new), len(block)), -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _get_windows(word, n):
    """Yields tuples of all n-length windows."""
    for i in range(len(word) - n + 1):
        yield tuple(word[i:i+n])

def _init_counts(word, n):
    counts = {}
    for w in _get_windows(word, n):
        if len(set(w)) == n:
            counts[w] = counts.get(w, 0) + 1
    return counts

def _update_counts_after_deletion(word, index, n, counts):
    """
    Updates counts after deleting word[index].
    Returns True if all permutations are still present (no count dropped to 0).
    Modifies counts in place.
    """
    # Windows affected are those that included index 'index'.
    # Start indices j such that j <= index < j + n
    # j in [index - n + 1, index]
    
    # First, decrement counts for old windows
    # We need to know the old windows.
    # Since we are about to delete, we can reconstruct them from current word.
    
    valid = True
    
    # Old windows
    start_min = max(0, index - n + 1)
    start_max = min(index, len(word) - n)
    
    old_windows = []
    for j in range(start_min, start_max + 1):
        w = tuple(word[j:j+n])
        if len(set(w)) == n:
            old_windows.append(w)
            counts[w] -= 1
            if counts[w] == 0:
                # Check if this permutation is lost?
                # If count becomes 0, it means no other window had this permutation.
                # So it is lost.
                valid = False
                # We still need to finish updating or revert?
                # If invalid, we should revert.
                # Let's collect all changes to revert easily.
                pass
    
    # If valid so far, check new windows
    # The word shrinks by 1.
    # New windows are formed.
    # This is tricky to do in-place without copying.
    
    # Simpler approach for correctness and speed given N=5040:
    # Just recompute the set of permutations for the new word?
    # No, that's O(L*N).
    
    # Let's use a "Delta" approach carefully.
    # Actually, for n=7, L~6000.
    # Recomputing counts for the whole word takes ~10ms.
    # Trying 6000 deletions takes 60s.
    # This is too slow for a full pass.
    
    # We need the incremental update.
    
    # Let's implement incremental update properly.
    # 1. Identify old windows covering 'index'.
    # 2. Decrement their counts.
    # 3. Identify new windows in the deleted word covering the "gap".
    #    The deletion shifts elements after 'index' to the left.
    #    New windows start at j in [index - n + 1, index].
    #    Wait, the indices in the new word are different.
    
    # Let's just copy the word, delete, and check the affected windows in the new word.
    # But we need to update the global counts dict.
    
    pass

def construct(n):
    if n < 7:
        return "".join(map(str, _classic_construct(n)))
    
    start_time = time.time()
    deadline = start_time + 110
    
    word = _classic_construct(n)
    counts = _init_counts(word, n)
    best_word = list(word)
    best_len = len(word)
    
    # Optimization loop
    # We will use a greedy approach with random restarts/shuffles if stuck.
    
    # Helper to check if a specific deletion is valid
    # To speed up, we only check if the "lost" permutations are covered by "new" permutations.
    # When deleting index i:
    # Lost windows: L = {tuple(word[j:j+n]) for j in [i-n+1, i]}
    # Gained windows: G = {tuple(new_word[j:j+n]) for j in [i-n+1, i]} (indices in new word)
    # Note: new_word is word without i.
    # The windows in G are formed by elements from word.
    # Specifically, a new window starting at j in new_word corresponds to:
    # If j < i: new_word[j:j+n] = word[j:j+n+1] (shifted? No, j is same index in word, but length n)
    # Actually, new_word[k] = word[k] for k < i, and new_word[k] = word[k+1] for k >= i.
    
    # Let's implement a fast validity check.
    
    def is_deletion_valid(i):
        # 1. Get old windows covering i
        lost = []
        for j in range(max(0, i - n + 1), min(i, len(word) - n) + 1):
            w = tuple(word[j:j+n])
            if len(set(w)) == n:
                lost.append(w)
        
        # 2. Simulate deletion to get new windows
        # We don't want to actually modify word yet.
        # New word is word[:i] + word[i+1:]
        # New windows covering the "seam" at i.
        # Indices j in new_word such that the window includes the seam.
        # The seam is between new_word[i-1] and new_word[i] (which is word[i+1]).
        # Windows starting at j where j <= i-1 < j+n and j <= i < j+n?
        # In new_word, the elements that change are those involving index i.
        # Old indices: ..., i-1, i, i+1, ...
        # New indices: ..., i-1, i+1, ... (shifted left)
        # Windows in new_word starting at j:
        # If j < i: new_word[j:j+n] = word[j:i] + word[i+1:i+n+1-j]
        # If j >= i: new_word[j:j+n] = word[j+1:j+n+1] (these are old windows shifted, already counted? No, they are the same set of elements as old windows starting at j+1)
        
        # Actually, the set of windows in the new word is:
        # All old windows that did NOT include i.
        # Plus new windows formed by bridging the gap.
        
        # So, we only need to check if the permutations in `lost` are still present in `gained` or elsewhere.
        # `gained` are the new windows that include the bridged part.
        # The bridged part in new_word is around index i-1.
        # New windows starting at j in [i-n, i-1] (in new_word coordinates).
        
        gained = []
        # In new_word, the "hole" is at position i.
        # Windows starting at j such that j <= i-1 < j+n.
        # j in [i-n, i-1].
        for j in range(max(0, i - n), i):
            # Construct window from original word
            # new_word[j:j+n]
            # Elements:
            # word[j:i] (length i-j)
            # word[i+1:i+1+n-(i-j)] = word[i+1:i+n+1-j]
            w_list = word[j:i] + word[i+1:i+n+1-j]
            if len(w_list) == n:
                w = tuple(w_list)
                if len(set(w)) == n:
                    gained.append(w)
        
        # Check validity:
        # For each perm in lost:
        #   count[perm] - 1 (lost one instance)
        #   count[perm] + (number of times perm appears in gained)
        #   Must be > 0.
        
        # We need to know how many times each perm in lost appears in gained.
        gained_counts = {}
        for w in gained:
            gained_counts[w] = gained_counts.get(w, 0) + 1
            
        for w in lost:
            current_count = counts[w]
            # We are losing 1 instance of w.
            # We are gaining gained_counts.get(w, 0) instances of w.
            new_count = current_count - 1 + gained_counts.get(w, 0)
            if new_count <= 0:
                return False
        return True

    # Main Loop
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Randomize order to escape local minima?
        # For now, sequential.
        indices = list(range(len(word)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
            if is_deletion_valid(i):
                # Perform deletion
                # Update counts
                # 1. Decrement lost
                for j in range(max(0, i - n + 1), min(i, len(word) - n) + 1):
                    w = tuple(word[j:j+n])
                    if len(set(w)) == n:
                        counts[w] -= 1
                        if counts[w] == 0:
                            del counts[w]
                
                # 2. Increment gained
                for j in range(max(0, i - n), i):
                    w_list = word[j:i] + word[i+1:i+n+1-j]
                    if len(w_list) == n:
                        w = tuple(w_list)
                        if len(set(w)) == n:
                            counts[w] = counts.get(w, 0) + 1
                
                # 3. Delete from word
                word.pop(i)
                changed = True
                if len(word) < best_len:
                    best_len = len(word)
                    best_word = list(word)
                # Restart index loop because indices changed
                break # Restart the for loop by breaking and letting the while loop re-shuffle
                
    return "".join(map(str, best_word))
