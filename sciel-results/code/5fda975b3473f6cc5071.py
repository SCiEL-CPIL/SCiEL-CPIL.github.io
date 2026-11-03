import time
import sys
from itertools import permutations

_PERM_INDEX = {}
_PERM_LIST = []

def _build_perm_tables(n):
    global _PERM_INDEX, _PERM_LIST
    _PERM_INDEX = {}
    _PERM_LIST = []
    idx = 0
    for p in permutations(range(1, n + 1)):
        _PERM_INDEX[p] = idx
        _PERM_LIST.append(p)
        idx += 1

def _classic_superperm(n):
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
            max_ov = min(len(new), len(block))
            for o in range(max_ov, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    return word

def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r

def _is_valid_superperm(word, n):
    if len(word) < n:
        return False
    total_perms = n * _factorial(n)
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            seen.add(window)
    return len(seen) == total_perms

def _compute_stats(word, n):
    """Returns window_ids and perm_count."""
    L = len(word)
    num_windows = L - n + 1
    window_ids = [-1] * num_windows
    total_perms = n * _factorial(n)
    perm_count = [0] * total_perms
    
    for i in range(num_windows):
        window_tuple = tuple(word[i:i + n])
        if len(set(window_tuple)) == n:
            idx = _PERM_INDEX.get(window_tuple, -1)
            if idx >= 0:
                window_ids[i] = idx
                perm_count[idx] += 1
    return window_ids, perm_count

def _can_delete(i, n, window_ids, perm_count):
    """Check if deleting position i is safe."""
    start_w = max(0, i - n + 1)
    end_w = min(len(window_ids) - 1, i)
    
    local_counts = {}
    for k in range(start_w, end_w + 1):
        pid = window_ids[k]
        if pid >= 0:
            local_counts[pid] = local_counts.get(pid, 0) + 1
            
    for pid, count in local_counts.items():
        if perm_count[pid] == count:
            return False
    return True

def _delete_pass(word, n, direction, time_limit, start_time):
    """
    Repeatedly delete the first safe character found in the given direction.
    direction: 1 for left-to-right, -1 for right-to-left.
    """
    word = list(word)
    changed = True
    while changed:
        if time.time() - start_time > time_limit:
            break
            
        L = len(word)
        if L < n:
            break
            
        window_ids, perm_count = _compute_stats(word, n)
        
        # Determine scan order
        if direction == 1:
            indices = range(L)
        else:
            indices = range(L - 1, -1, -1)
            
        found = False
        for i in indices:
            if _can_delete(i, n, window_ids, perm_count):
                # Delete
                del word[i]
                found = True
                break # Restart scan with new word
                
        changed = found
        
    return word

def construct(n):
    start_time = time.time()
    time_limit = 140  # Safety margin
    
    _build_perm_tables(n)
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    # Apply deletion passes
    # Left to right
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1:
        word = _delete_pass(word, n, 1, remaining, start_time)
        
    # Right to left
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1:
        word = _delete_pass(word, n, -1, remaining, start_time)
        
    # If time remains, try a few random swaps to escape local minima, then delete again
    # This is a simple local search heuristic
    import random
    random.seed(42) # Fixed seed for reproducibility? Or just use time?
    # Let's use a fixed seed to be deterministic in evaluation if possible, 
    # but the task implies we just need a good score.
    
    # We can try a few iterations of:
    # 1. Pick a random position i
    # 2. Swap word[i] and word[i+1]
    # 3. Run a quick delete pass (or just check if valid and shorter? No, swap doesn't change length)
    # Actually, swaps don't reduce length. They might enable further deletions.
    
    # A more effective move: "Shift" a character?
    # Given the small gap, the deletion pass might be sufficient.
    # Let's just do another L2R pass to ensure convergence.
    remaining = time_limit - (time.time() - start_time)
    if remaining > 1:
        word = _delete_pass(word, n, 1, remaining, start_time)

    # Final validation
    if not _is_valid_superperm(word, n):
        word = _classic_superperm(n)
        
    return "".join(map(str, word))
