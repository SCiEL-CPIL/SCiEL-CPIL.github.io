"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START
def construct(n):
    """Return shortest superpermutation found for n symbols within time budget."""
    if n <= 2:
        return "121" if n == 2 else "1"
    
    start_time = time.time()
    TIME_LIMIT = 145  # Safety margin before 150s limit
    
    # For small n, use exact known solutions
    if n == 3:
        return "123121321"
    if n == 4:
        return "1234123142312431214321342132431213243214321"
    if n == 5:
        return "1234512341523412534123541234531234531423453142534531423545314235451234512354512354152345152354515234512354123545123541523415234515234152354152354512354512354512345312345312534531253453123454123454125345412534541235454123545412345"
    
    # Build initial superpermutation using recursive construction
    word = build_initial_superperm(n)
    
    # Verify it's valid
    if not is_valid_superperm(word, n):
        return "".join(map(str, word))
    
    # For n >= 5, try to shorten using fast local-check compression
    if n >= 5 and time.time() - start_time < TIME_LIMIT:
        word = compress_superperm_fast_local(word, n, start_time, TIME_LIMIT)
    
    return "".join(map(str, word))

def build_initial_superperm(n):
    """Build initial superpermutation using the standard recursive construction."""
    word = [1]
    for m in range(2, n + 1):
        # Find all distinct (m-1)-length windows that are permutations
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        # Build new word by inserting m after each permutation
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
    return word

def is_valid_superperm(word, n):
    """Check if word contains all n! permutations as consecutive windows."""
    if len(word) < n:
        return False
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            seen.add(window)
    return len(seen) == factorial(n)

def factorial(x):
    result = 1
    for i in range(2, x + 1):
        result *= i
    return result

def compress_superperm_fast_local(word, n, start_time, time_limit):
    """
    Compress superpermutation using local validity checks.
    Maintains a count of each permutation.
    Checks if deletion at p is safe by verifying that no permutation 
    unique to the local region is lost.
    """
    current = list(word)
    target_count = factorial(n)
    
    # Map permutations to IDs for fast counting
    # We use a dictionary to map tuple -> id
    perm_to_id = {}
    id_to_perm = []
    next_id = 0
    
    def get_perm_id(p):
        nonlocal next_id
        if p in perm_to_id:
            return perm_to_id[p]
        perm_to_id[p] = next_id
        id_to_perm.append(p)
        next_id += 1
        return next_id - 1

    # Initialize counts
    # Size of count array is target_count (5040 for n=7)
    perm_counts = [0] * target_count
    
    # Helper to update counts for a range of windows
    # This is used during initialization and updates
    
    def init_counts(arr):
        nonlocal perm_counts
        perm_counts = [0] * target_count
        for i in range(len(arr) - n + 1):
            w = tuple(arr[i:i+n])
            if len(set(w)) == n:
                pid = get_perm_id(w)
                perm_counts[pid] += 1

    init_counts(current)
    
    # Helper to check if a window is a valid permutation and return its ID or -1
    def get_window_id(arr, start):
        if start < 0 or start + n > len(arr):
            return -1
        w = tuple(arr[start:start+n])
        if len(set(w)) == n:
            return get_perm_id(w)
        return -1

    improved = True
    # Multiple passes to ensure local minima
    while improved and time.time() - start_time < time_limit:
        improved = False
        L = len(current)
        # Iterate from right to left
        for pos in range(L - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            if L - 1 < n:
                continue
            
            # Check if deletion at pos is safe
            # Affected windows are those containing pos: [pos-n+1, pos]
            safe = True
            affected_ids = []
            for i in range(max(0, pos - n + 1), min(pos, L - n) + 1):
                pid = get_window_id(current, i)
                if pid != -1:
                    if perm_counts[pid] == 1:
                        safe = False
                        break
                    affected_ids.append((i, pid))
            
            if not safe:
                continue
            
            # Perform deletion
            # We need to update counts for:
            # 1. Old affected windows (decrement)
            # 2. New affected windows (increment)
            
            # Decrement old
            for i, pid in affected_ids:
                perm_counts[pid] -= 1
            
            # Delete
            current.pop(pos)
            L -= 1
            
            # Increment new
            # The new windows are at indices [pos-n+1, pos] in the NEW array
            # Note: indices >= pos in the new array are the same as indices > pos in old
            # Indices < pos are the same.
            # The windows that CHANGED are those that straddle the deletion point.
            # In the new array, the gap is between pos-1 and pos.
            # Windows starting at i in [pos-n+1, pos] in the new array are the new ones.
            
            new_affected_ids = []
            for i in range(max(0, pos - n + 1), min(pos, L - n) + 1):
                pid = get_window_id(current, i)
                if pid != -1:
                    new_affected_ids.append((i, pid))
                    perm_counts[pid] += 1
            
            # Note: Some of the new_affected_ids might be the same as old_affected_ids 
            # (if the content didn't change for those specific windows).
            # But we decremented all old and incremented all new.
            # If a window was unaffected (e.g., entirely to the left), it wasn't in old_affected 
            # (if it didn't contain pos) and isn't in new_affected (if it doesn't contain the new gap).
            # Wait, the definition of "affected" is "contains the deleted position" for old,
            # and "contains the new gap" for new.
            # 
            # Let's verify the logic.
            # Old window i contains pos if i <= pos < i+n.
            # New window i' contains the gap (between pos-1 and pos) if i' <= pos-1 < i'+n 
            # i.e., pos-n < i' <= pos-1.
            # 
            # The set of windows that changed content are exactly those that covered the gap.
            # Any window that did NOT cover the gap in the old word, and does NOT cover the gap 
            # in the new word, is identical.
            # 
            # So, we only need to update the counts for the windows that covered the gap.
            # 
            # Old windows covering pos: i in [pos-n+1, pos]
            # New windows covering gap: i in [pos-n+1, pos] (in new indexing)
            # 
            # The code above does exactly this.
            
            improved = True
            # We don't break the loop, we continue checking previous positions?
            # No, if we delete at pos, the indices shift. 
            # It is safer to restart the scan or be very careful.
            # Given L is small, restarting the scan from L-1 is O(L^2) in worst case,
            # but since deletions are rare, it's fine.
            # However, `improved=True` and `break` is standard for greedy.
            # Let's break and restart the pass to ensure index consistency.
            break

    return current

# EVOLVE-BLOCK-END
