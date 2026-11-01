import time
import random
from collections import defaultdict

def construct(n):
    start_time = time.time()
    deadline = 145.0 # Safety margin
    
    # 1. Generate initial superpermutation
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
    
    # 2. Local Search Deletion
    current = list(word)
    L = len(current)
    
    # Map permutations to integer IDs for faster hashing/comparison
    # Since n is small, we can use a dictionary mapping tuple -> int
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

    perm_ids = [0] * L
    perm_count = defaultdict(int)
    
    for i in range(L - n + 1):
        p = tuple(current[i:i+n])
        pid = get_id(p)
        perm_ids[i] = pid
        perm_count[pid] += 1
        
    target_perms = n
    for i in range(1, n):
        target_perms *= i
        
    if len(perm_count) < target_perms:
        return "".join(map(str, current))
        
    # Deletion loop
    # We will scan the array to find deletable indices.
    # To avoid getting stuck in a local minimum due to ordering, 
    # we can randomize the start or direction occasionally, 
    # but a full scan is usually better for finding *all* candidates.
    
    while time.time() < deadline:
        deletable = []
        
        # Scan all indices
        for i in range(L):
            # Check if i can be deleted
            start_w = max(0, i - n + 1)
            end_w = min(i, L - n)
            
            if start_w > end_w:
                continue
                
            # Collect old window IDs
            old_ids = set()
            for w in range(start_w, end_w + 1):
                old_ids.add(perm_ids[w])
            
            # Check if any old window has count 1 and is not formed by deletion
            # We need to form the new windows to see if they cover the unique old ones.
            
            can_delete = True
            new_ids = set()
            
            # Form new windows
            # A new window starts at j where j in [i-n+1, i-1]
            # Length of prefix from j to i-1 is i-j
            # Length of suffix from i+1 is n - (i-j)
            for j in range(max(0, i - n + 1), i):
                prefix_len = i - j
                suffix_len = n - prefix_len
                if j + n > L - 1: # Ensure we have enough chars?
                    # Actually, if we delete i, the new length is L-1.
                    # The window starting at j in the new string corresponds to
                    # current[j:i] + current[i+1 : i+1+suffix_len]
                    # We need i+1+suffix_len <= L
                    # suffix_len = n - (i-j)
                    # i+1+n-i+j = n+1+j
                    # We need n+1+j <= L => j <= L-n-1
                    if j > L - n - 1:
                        continue
                
                # Construct the new permutation tuple
                # To speed up, we can just construct the tuple
                new_p = tuple(current[j:i] + current[i+1 : i+1+suffix_len])
                new_pid = get_id(new_p)
                new_ids.add(new_pid)
            
            # Check unique old windows
            for pid in old_ids:
                if perm_count[pid] == 1:
                    if pid not in new_ids:
                        can_delete = False
                        break
                        
            if can_delete:
                deletable.append(i)
                
        if not deletable:
            break
            
        # Choose an index to delete
        # Heuristic: Delete the one that is "most central" or just the first one?
        # Random choice from candidates might help escape local minima?
        # Let's pick the first one for determinism, or random if we want variety.
        # Given the time limit, random might be safer to explore different states.
        idx_to_delete = random.choice(deletable)
        
        # Perform deletion
        # Update perm_ids and perm_count
        # This is the expensive part.
        # We need to shift perm_ids to the right of idx_to_delete?
        # No, we remove the element.
        
        # Remove old windows containing idx_to_delete
        start_w = max(0, idx_to_delete - n + 1)
        end_w = min(idx_to_delete, L - n)
        
        for w in range(start_w, end_w + 1):
            pid = perm_ids[w]
            perm_count[pid] -= 1
            if perm_count[pid] == 0:
                del perm_count[pid]
                
        # Add new windows
        for j in range(max(0, idx_to_delete - n + 1), idx_to_delete):
            prefix_len = idx_to_delete - j
            suffix_len = n - prefix_len
            if j > L - n - 1:
                continue
            new_p = tuple(current[j:idx_to_delete] + current[idx_to_delete+1 : idx_to_delete+1+suffix_len])
            new_pid = get_id(new_p)
            perm_count[new_pid] += 1
            # Update perm_ids for the new window position j
            # The window at j in the new array is this new_p
            perm_ids[j] = new_pid
            
        # Shift perm_ids
        # perm_ids is a list of IDs for windows starting at 0, 1, ... L-n
        # After deletion at idx, the windows starting at 0..idx-1 are unchanged?
        # No.
        # Window starting at j < idx:
        # If j + n > idx, the window content changes.
        # Specifically, windows starting at j in [idx-n+1, idx-1] change.
        # Windows starting at j < idx-n+1 are unchanged.
        # Windows starting at j > idx-1 shift left by 1.
        
        # Let's just rebuild perm_ids for the affected range and shift the rest.
        # Affected range for content: [idx-n+1, idx-1]
        # Shift range for indices: [idx, L-n] -> [idx-1, L-n-1]
        
        # 1. Update perm_ids for j in [idx-n+1, idx-1] (already done above in the "Add new windows" loop? 
        #    Wait, in the loop above, I updated perm_ids[j] for the new windows.
        #    But I need to make sure I covered all j that form new windows.
        #    The loop `for j in range(max(0, idx_to_delete - n + 1), idx_to_delete)` covers exactly those.
        
        # 2. Shift perm_ids for j > idx-1
        #    The window that was at j is now at j-1.
        #    Its content is unchanged (since it doesn't include idx).
        #    So perm_ids[j-1] = perm_ids[j]
        
        # We need to be careful with the list length.
        # Old length of perm_ids: L - n + 1
        # New length: L - 1 - n + 1 = L - n
        # So we drop the last element.
        
        # Shift
        for j in range(idx_to_delete, L - n):
            perm_ids[j-1] = perm_ids[j]
            
        # Remove last element
        perm_ids.pop()
        
        # Update current
        current = current[:idx_to_delete] + current[idx_to_delete+1:]
        L -= 1
        
    return "".join(map(str, current))
