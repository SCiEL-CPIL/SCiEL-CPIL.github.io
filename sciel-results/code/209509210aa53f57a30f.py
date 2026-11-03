import time
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140

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

    # 2. Setup for local search
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    # Initialize counts
    counts = [0] * num_perms
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i+n])
        if p in perm_to_id:
            counts[perm_to_id[p]] += 1
            
    # Check if valid initially
    if any(c == 0 for c in counts):
        # Should not happen for standard construction
        return "".join(map(str, word))

    # Track permutations with count == 1 (critical)
    critical_count = sum(1 for c in counts if c == 1)
    
    # Local Search: Deletion
    # We will repeatedly try to delete characters
    # To avoid O(N^2) complexity, we use the fast count update
    
    def try_delete(i):
        """
        Attempts to delete word[i]. Returns True if successful.
        Modifies word and counts if successful.
        """
        L = len(word)
        if L == n: # Cannot delete if only one window
            return False
        
        # Quick check: if deleting i would make any critical perm disappear
        # We need to check if any window containing i is the only occurrence of its perm
        
        # Affected windows in old word: start k in [i-n+1, i]
        # We need to decrement counts for these
        # And increment counts for new windows formed after deletion
        
        # Store old values to revert if fail
        old_counts = counts.copy()
        
        # 1. Decrement old windows and track which perms become 0
        old_windows = []
        for k in range(max(0, i - n + 1), min(i + 1, L - n + 1)):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] -= 1
                old_windows.append((k, pid))
        
        # 2. Simulate deletion to find new windows
        # New word is word[:i] + word[i+1:]
        # New windows affected are those starting at k in [i-n+1, i] in the NEW word.
        
        new_windows_pids = []
        for k in range(max(0, i - n + 1), min(i, L - n)): # k is start index in NEW word
            # The window is new_word[k:k+n]
            # new_word is word with i removed.
            if k < i:
                w1 = word[k:i]
                w2 = word[i+1:k+n+1]
                p = tuple(w1 + w2)
            else: # k >= i
                # new_word[k] = word[k+1]
                # window is word[k+1 : k+1+n]
                p = tuple(word[k+1:k+1+n])
                
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] += 1
                new_windows_pids.append(pid)
                
        # Check if all counts are >= 1
        valid = True
        for pid in range(num_perms):
            if counts[pid] == 0:
                valid = False
                break
                
        if valid:
            # Perform deletion
            word.pop(i)
            # Note: counts are already updated to reflect the new state
            return True
        else:
            # Revert counts
            counts[:] = old_counts
            return False

    # Iterate deletion passes
    improved = True
    while improved:
        improved = False
        # Left to right
        i = 0
        while i < len(word):
            if time.time() > deadline:
                return "".join(map(str, word))
            if try_delete(i):
                improved = True
                # Do not increment i, check same position again (new character)
            else:
                i += 1
        # Right to left
        i = len(word) - 1
        while i >= 0:
            if time.time() > deadline:
                return "".join(map(str, word))
            if try_delete(i):
                improved = True
            else:
                i -= 1

    # 3. Local Search: Swap
    # Try swapping adjacent characters if it reduces length (by enabling further deletions?)
    # No, swap doesn't change length directly.
    # But swap might enable deletions that were previously blocked.
    # Let's do a pass of swaps and then try deletions again.
    
    def try_swap(i):
        """
        Attempts to swap word[i] and word[i+1].
        Returns True if successful and it enables at least one deletion?
        Or just returns True if valid.
        We want to minimize length, so we should only swap if it leads to a shorter word.
        However, checking if it leads to a shorter word is expensive.
        Simple heuristic: Swap if valid, then immediately try to delete at i and i+1.
        If a deletion succeeds, keep the swap. Otherwise revert.
        """
        L = len(word)
        if i >= L - 1:
            return False
            
        # Check validity of swap
        # Affected windows: start k in [i-n+1, i+1]
        # We need to check if all perms are still covered.
        
        # Store old counts
        old_counts = counts.copy()
        
        # Decrement old windows
        for k in range(max(0, i - n + 1), min(i + 2, L - n + 1)):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] -= 1
        
        # Perform swap
        word[i], word[i+1] = word[i+1], word[i]
        
        # Increment new windows
        for k in range(max(0, i - n + 1), min(i + 2, L - n + 1)):
            p = tuple(word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                counts[pid] += 1
        
        # Check validity
        valid = True
        for pid in range(num_perms):
            if counts[pid] == 0:
                valid = False
                break
        
        if not valid:
            # Revert
            word[i], word[i+1] = word[i+1], word[i]
            counts[:] = old_counts
            return False
        
        # If valid, try to delete at i and i+1
        # If deletion succeeds, we have improved.
        # If not, we revert the swap to maintain the original state (since swap itself doesn't reduce length).
        # Actually, we can keep the swap if it doesn't hurt, but we want to minimize length.
        # So we only keep swap if it leads to a deletion.
        
        # Try delete i
        saved_word = word.copy()
        saved_counts = counts.copy()
        
        if try_delete(i):
            return True
        else:
            # Revert deletion attempt (which reverted itself if failed)
            # But try_delete modifies word and counts if successful.
            # If it failed, it reverted. So state is back to post-swap.
            # Now try delete i+1 (which is now index i+1 in the post-swap word? No, indices shifted if deletion happened)
            # Since try_delete(i) failed, word is still post-swap.
            # Try delete i+1
            if try_delete(i+1):
                return True
            
            # If no deletion, revert swap
            word[:] = saved_word
            counts[:] = saved_counts
            return False

    # Iterate swap passes
    improved = True
    while improved:
        improved = False
        i = 0
        while i < len(word) - 1:
            if time.time() > deadline:
                return "".join(map(str, word))
            if try_swap(i):
                improved = True
                # Don't increment i, as word changed
            else:
                i += 1

    return "".join(map(str, word))
