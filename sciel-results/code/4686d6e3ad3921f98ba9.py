import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit

    # 1. Generate initial superpermutation (Recursive Construction)
    # Length for n=7 is 5913.
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        # Extract unique permutations of length m-1
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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if o == 0:
                    break
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    if n <= 4:
        return "".join(map(str, word))
    
    current = list(word)
    L = len(current)
    
    # 2. Setup for Local Search
    # Map permutations to IDs for speed
    perm_to_id = {}
    id_to_perm = {}
    next_id = 0
    
    def get_id(perm):
        nonlocal next_id
        if perm in perm_to_id:
            return perm_to_id[perm]
        perm_to_id[perm] = next_id
        id_to_perm[next_id] = perm
        next_id += 1
        return next_id - 1

    # Initialize perm_ids and perm_count
    # perm_ids[i] is the ID of the permutation starting at index i in 'current'
    perm_ids = [0] * L
    perm_count = defaultdict(int)
    
    # Note: Windows start at 0 to L-n
    # We only care about windows of length n that are valid permutations (distinct symbols)
    # In the recursive construction, all windows of length n are permutations.
    
    for i in range(L - n + 1):
        p = tuple(current[i:i+n])
        pid = get_id(p)
        perm_ids[i] = pid
        perm_count[pid] += 1
        
    # The number of unique permutations should be n!
    target_perms = 1
    for i in range(1, n + 1):
        target_perms *= i
        
    if len(perm_count) < target_perms:
        # Should not happen, but return if invalid
        return "".join(map(str, current))
        
    # 3. Deletion Loop
    # We scan for deletable indices.
    # To speed up, we can scan in random order or reverse order.
    # Let's scan in reverse order (from L-1 down to 0) as it might be easier to delete from the end initially,
    # but actually any order works. Let's use a list of candidates and pick one.
    
    # To avoid O(L^2) complexity in the worst case (if we scan all L indices every time),
    # we can maintain a set of "potentially deletable" indices, but that's complex.
    # Given L ~ 6000, a full scan is ~6000 checks.
    # Each check is O(n) to build new windows.
    # Total work per pass: O(L * n).
    # If we delete 1 char per pass, and we can delete ~100 chars, total work O(100 * L * n) ~ 600,000 * 7 ~ 4M ops.
    # This is very fast.
    
    # The main cost is the tuple creation and hashing.
    
    while time.time() < deadline:
        deletable = []
        
        # Scan all indices
        # We can iterate backwards to handle index shifts naturally if we were doing in-place,
        # but we are collecting candidates first.
        
        for i in range(L):
            # Check if i can be deleted
            # Old windows containing i: start indices [max(0, i-n+1), min(i, L-n)]
            start_w = max(0, i - n + 1)
            end_w = min(i, L - n)
            
            if start_w > end_w:
                continue
                
            # Identify unique old windows
            # A window is "unique" if its count is 1.
            # If we delete i, these windows are destroyed.
            # We must ensure that every unique old window is "replaced" by a new window with the same ID.
            
            unique_old_ids = set()
            for w in range(start_w, end_w + 1):
                pid = perm_ids[w]
                if perm_count[pid] == 1:
                    unique_old_ids.add(pid)
            
            # If no unique old windows, deletion is safe.
            if not unique_old_ids:
                deletable.append(i)
                continue
                
            # Check if new windows cover unique_old_ids
            # New windows are formed at start positions j in [max(0, i-n+1), i-1]
            # The new window at j consists of current[j:i] + current[i+1 : i+1 + (n - (i-j))]
            
            new_ids = set()
            for j in range(max(0, i - n + 1), i):
                prefix_len = i - j
                suffix_len = n - prefix_len
                
                # Check bounds for suffix
                # The suffix ends at i + suffix_len.
                # We need i + suffix_len <= L
                # i + n - (i-j) = n + j
                # So we need n + j <= L => j <= L - n
                if j > L - n:
                    continue
                    
                # Construct new permutation
                # current[j:i] + current[i+1 : i+1+suffix_len]
                # This creates a new list, then tuple.
                # Optimization: We can avoid creating the full tuple if possible?
                # For n=7, it's small.
                new_p = tuple(current[j:i] + current[i+1 : i+1+suffix_len])
                new_pid = get_id(new_p)
                new_ids.add(new_pid)
                
                # Early exit if all unique_old_ids are covered
                if unique_old_ids.issubset(new_ids):
                    break
            
            if unique_old_ids.issubset(new_ids):
                deletable.append(i)
                
        if not deletable:
            break
            
        # Choose an index to delete
        # Random choice to avoid bias
        idx_to_delete = random.choice(deletable)
        
        # Perform deletion
        # 1. Update perm_count for old windows
        start_w = max(0, idx_to_delete - n + 1)
        end_w = min(idx_to_delete, L - n)
        
        for w in range(start_w, end_w + 1):
            pid = perm_ids[w]
            perm_count[pid] -= 1
            if perm_count[pid] == 0:
                # It might become non-zero again if it's in new windows
                pass # Don't delete key yet, or handle carefully
        
        # 2. Update perm_count for new windows
        new_window_starts = []
        for j in range(max(0, idx_to_delete - n + 1), idx_to_delete):
            prefix_len = idx_to_delete - j
            suffix_len = n - prefix_len
            if j > L - n:
                continue
                
            new_p = tuple(current[j:idx_to_delete] + current[idx_to_delete+1 : idx_to_delete+1+suffix_len])
            new_pid = get_id(new_p)
            perm_count[new_pid] += 1
            new_window_starts.append((j, new_pid))
            
        # 3. Update perm_ids
        # The perm_ids array needs to be shifted.
        # Windows starting at 0 ... idx_to_delete - n are unchanged.
        # Windows starting at idx_to_delete - n + 1 ... idx_to_delete - 1 are CHANGED (they are the new ones).
        # Windows starting at idx_to_delete ... L - n are SHIFTED LEFT by 1.
        
        # Let's rebuild the relevant part of perm_ids.
        # Actually, it's safer to just update the specific indices.
        
        # The indices in perm_ids correspond to start positions in the CURRENT (pre-deletion) word?
        # No, perm_ids[i] is the ID of the window starting at i in the CURRENT word.
        # After deletion at idx, the word length becomes L-1.
        # The new word is current[:idx] + current[idx+1:].
        
        # Let's map old indices to new indices.
        # New window starting at j (in new word):
        # If j < idx - n + 1: It corresponds to old window starting at j. Content unchanged.
        # If idx - n + 1 <= j < idx: It corresponds to the NEW window formed by the gap.
        # If j >= idx: It corresponds to old window starting at j+1. Content unchanged.
        
        # So:
        # For j in [0, idx - n]: perm_ids_new[j] = perm_ids_old[j]
        # For j in [idx - n + 1, idx - 1]: perm_ids_new[j] = new_pid calculated above
        # For j in [idx, L - n]: perm_ids_new[j] = perm_ids_old[j + 1]
        
        # We need to update perm_ids in place.
        # Since we are shifting, it's easier to create a new list or do careful shifting.
        
        # Let's do it carefully.
        # 1. Save the tail part that will be shifted.
        # The tail starts at old index idx.
        # It will move to new index idx - 1.
        
        # We need to update perm_ids for indices [0, L-n-1] (new length L-1 has L-n windows).
        
        # Let's iterate from the end to the beginning to avoid overwriting data we need?
        # Or just build a new list. L is small (~6000).
        
        new_perm_ids = [0] * (L - n)
        
        # Part 1: Unchanged prefix
        # Old indices 0 to idx - n (inclusive)
        # If idx - n < 0, this range is empty.
        end_prefix = idx - n
        for j in range(0, min(end_prefix + 1, L - n + 1)):
            if j < len(perm_ids):
                new_perm_ids[j] = perm_ids[j]
            else:
                # Should not happen if logic is correct
                pass
                
        # Part 2: New windows
        # New indices idx - n + 1 to idx - 1
        # These correspond to the new_pid values we calculated.
        # We need to map the calculated new_pids to the correct new indices.
        # The calculation loop `for j in range(max(0, idx - n + 1), idx)` used OLD index j.
        # In the new word, the window starting at OLD index j (where j < idx) starts at NEW index j.
        # So new_perm_ids[j] = new_pid for these j.
        
        for j, new_pid in new_window_starts:
            if j < len(new_perm_ids):
                new_perm_ids[j] = new_pid
                
        # Part 3: Shifted suffix
        # Old indices idx to L - n
        # New indices idx - 1 to L - n - 1
        # new_perm_ids[new_j] = perm_ids_old[old_j]
        # new_j = old_j - 1
        # old_j = new_j + 1
        
        for new_j in range(max(0, idx - 1), L - n):
            old_j = new_j + 1
            if old_j < len(perm_ids):
                new_perm_ids[new_j] = perm_ids[old_j]
                
        perm_ids = new_perm_ids
        
        # 4. Update current word
        current = current[:idx_to_delete] + current[idx_to_delete+1:]
        L -= 1
        
        # 5. Clean up perm_count if necessary?
        # The counts should be consistent now.
        
    return "".join(map(str, current))
