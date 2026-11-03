# EVOLVE-BLOCK-START
import time
import sys
from itertools import permutations

# Precompute permutation index for n=7 (and smaller)
_PERM_INDEX = {}
_PERM_LIST = []
_PERM_INV = []

def _build_perm_tables(n):
    global _PERM_INDEX, _PERM_LIST, _PERM_INV
    _PERM_INDEX = {}
    _PERM_LIST = []
    _PERM_INV = []
    idx = 0
    for p in permutations(range(1, n + 1)):
        _PERM_INDEX[p] = idx
        _PERM_LIST.append(p)
        _PERM_INV.append(p)
        idx += 1

def _classic_superperm(n):
    """Classic recursive superpermutation construction."""
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
            # Find maximum overlap between end of new and start of block
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if o == 0:
                    overlap = 0
                    break
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

def _compute_window_perms(word, n):
    """Compute window_perm array: for each starting position, the perm index or -1."""
    total_perms = n * _factorial(n)
    L = len(word)
    num_windows = max(0, L - n + 1)
    window_perm = [-1] * num_windows
    for i in range(num_windows):
        w = tuple(word[i:i + n])
        if len(set(w)) == n:
            idx = _PERM_INDEX.get(w, -1)
            if idx >= 0:
                window_perm[i] = idx
    return window_perm

def _compute_perm_counts(window_perm):
    total_perms = len(_PERM_INDEX)
    perm_count = [0] * total_perms
    for idx in window_perm:
        if idx >= 0:
            perm_count[idx] += 1
    return perm_count

def _find_all_safe_deletions(word, n, window_perm, perm_count):
    """Find all positions that can be safely deleted (no unique perm window contains them)."""
    L = len(word)
    safe = [True] * L
    
    # For each position, check if any window containing it is the sole occurrence of its perm
    for i in range(L):
        s_min = max(0, i - n + 1)
        s_max = min(i, L - n)
        for s in range(s_min, s_max + 1):
            if window_perm[s] >= 0 and perm_count[window_perm[s]] == 1:
                safe[i] = False
                break
    return safe

def _apply_deletions(word, positions_to_delete):
    """Delete multiple positions from word (positions in descending order)."""
    word = list(word)
    for i in sorted(positions_to_delete, reverse=True):
        if i < len(word):
            del word[i]
    return word

def _fast_delete_pass(word, n, time_limit, start_time):
    """
    Fast deletion pass with incremental updates.
    A character at position i can be deleted if no valid permutation window
    that contains position i is the ONLY occurrence of that permutation.
    """
    word = list(word)
    L = len(word)
    
    # Initial computation
    window_perm = _compute_window_perms(word, n)
    perm_count = _compute_perm_counts(window_perm)
    
    total_deletions = 0
    max_rounds = 200
    
    for round_num in range(max_rounds):
        if time.time() - start_time > time_limit:
            break
        if len(word) < n:
            break
        
        # Find all safe deletions in this pass
        safe = _find_all_safe_deletions(word, n, window_perm, perm_count)
        
        deletions = [i for i, s in enumerate(safe) if s]
        if not deletions:
            break
        
        # Apply all safe deletions at once
        word = _apply_deletions(word, deletions)
        total_deletions += len(deletions)
        
        # Recompute window_perm and perm_count for the new word
        window_perm = _compute_window_perms(word, n)
        perm_count = _compute_perm_counts(window_perm)
        
        # Early exit if no more progress possible
        if total_deletions > 0 and round_num > 5:
            # Check if we're making diminishing returns
            pass
    
    return word

def _try_swap_pass(word, n, time_limit, start_time):
    """
    Try swapping adjacent characters to potentially enable more deletions.
    """
    word = list(word)
    total_perms = n * _factorial(n)
    
    # Track best word found
    best_word = list(word)
    best_len = len(word)
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
        
        if len(word) < 2:
            break
        
        # Try a deterministic swap based on attempt number for reproducibility
        i = (attempt * 7 + 3) % (len(word) - 1)
        
        # Check if swap is valid
        word[i], word[i + 1] = word[i + 1], word[i]
        
        # Quick check: does the word still contain all permutations?
        seen = set()
        for s in range(len(word) - n + 1):
            window = tuple(word[s:s + n])
            if len(set(window)) == n:
                seen.add(window)
        
        if len(seen) == total_perms:
            # Keep the swap - this might enable further deletions
            if len(word) < best_len:
                best_word = list(word)
                best_len = len(word)
        else:
            # Revert
            word[i], word[i + 1] = word[i + 1], word[i]
    
    return best_word

def _local_search_delete(word, n, time_limit, start_time):
    """
    More aggressive local search: try inserting and deleting characters
    to find shorter valid superpermutations.
    """
    word = list(word)
    total_perms = n * _factorial(n)
    best_word = list(word)
    best_len = len(word)
    
    # First do standard deletion
    window_perm = _compute_window_perms(word, n)
    perm_count = _compute_perm_counts(window_perm)
    
    # Iterative deletion with single-position focus
    for round_num in range(100):
        if time.time() - start_time > time_limit:
            break
        if len(word) < n:
            break
        
        found_deletion = False
        # Try each position from right to left
        for i in range(len(word) - 1, -1, -1):
            if time.time() - start_time > time_limit:
                break
            
            # Check if position i is in any unique perm window
            s_min = max(0, i - n + 1)
            s_max = min(i, len(word) - n)
            critical = False
            for s in range(s_min, s_max + 1):
                if s < len(window_perm) and window_perm[s] >= 0:
                    if perm_count[window_perm[s]] == 1:
                        critical = True
                        break
            
            if not critical:
                # Delete position i
                del word[i]
                
                # Recompute
                window_perm = _compute_window_perms(word, n)
                perm_count = _compute_perm_counts(window_perm)
                
                found_deletion = True
                if len(word) < best_len:
                    best_word = list(word)
                    best_len = len(word)
                break  # Restart scan after a removal
        
        if not found_deletion:
            break
    
    return best_word

def _two_char_swap_search(word, n, time_limit, start_time):
    """
    Try swapping two non-adjacent characters to potentially enable more deletions.
    """
    word = list(word)
    total_perms = n * _factorial(n)
    best_word = list(word)
    best_len = len(word)
    L = len(word)
    
    # Try random pairs of positions to swap
    import random
    rng = random.Random(42)
    
    max_attempts = 200
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
        
        if L < 2:
            break
        
        i = rng.randint(0, L - 2)
        j = rng.randint(i + 1, L - 1)
        
        # Swap
        word[i], word[j] = word[j], word[i]
        
        # Check validity
        seen = set()
        for s in range(len(word) - n + 1):
            window = tuple(word[s:s + n])
            if len(set(window)) == n:
                seen.add(window)
        
        if len(seen) == total_perms:
            # Valid - keep it
            if len(word) < best_len:
                best_word = list(word)
                best_len = len(word)
        else:
            # Revert
            word[i], word[j] = word[j], word[i]
    
    return best_word

def _insert_delete_search(word, n, time_limit, start_time):
    """
    Try inserting a character at a random position, then doing deletion passes.
    This can sometimes lead to shorter final results.
    """
    word = list(word)
    total_perms = n * _factorial(n)
    best_word = list(word)
    best_len = len(word)
    
    import random
    rng = random.Random(123)
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
        
        # Try inserting a random character at a random position
        pos = rng.randint(0, len(word))
        char = rng.randint(1, n)
        
        test_word = list(word)
        test_word.insert(pos, char)
        
        # Do a quick deletion pass on the test word
        test_word = _fast_delete_pass(test_word, n, time_limit, start_time)
        
        # Check validity
        if _is_valid_superperm(test_word, n):
            if len(test_word) < best_len:
                best_word = list(test_word)
                best_len = len(test_word)
    
    return best_word

def construct(n):
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, we use 135s to be safe
    time_limit = 135.0
    
    # Build permutation tables
    _build_perm_tables(n)
    
    # Get classic superpermutation
    word = _classic_superperm(n)
    
    # Phase 1: Fast batch deletion
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        word = _fast_delete_pass(word, n, remaining_time, start_time)
    
    # Phase 2: Local search deletion (more careful, single-position)
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        word = _local_search_delete(word, n, remaining_time, start_time)
    
    # Phase 3: Try swaps to enable more deletions
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5:
        word = _try_swap_pass(word, n, remaining_time, start_time)
        
        # After swaps, try deletion again
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5:
            word = _fast_delete_pass(word, n, remaining_time, start_time)
    
    # Phase 4: Two-char swap search
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        word = _two_char_swap_search(word, n, remaining_time, start_time)
        
        # After swaps, try deletion again
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time > 5:
            word = _fast_delete_pass(word, n, remaining_time, start_time)
    
    # Phase 5: Insert-delete search
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10:
        word = _insert_delete_search(word, n, remaining_time, start_time)
    
    # Final validation
    if not _is_valid_superperm(word, n):
        # Fallback to classic
        word = _classic_superperm(n)
    
    # Final safety check
    if time.time() - start_time > time_limit - 5:
        # Emergency: just validate and return
        if not _is_valid_superperm(word, n):
            word = _classic_superperm(n)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
