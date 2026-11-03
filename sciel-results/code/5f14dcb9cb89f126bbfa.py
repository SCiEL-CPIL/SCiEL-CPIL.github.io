"""Superpermutation search: classic construction + fast local search with count tracking."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations


def _classic_superperm(n):
    """Classic recursive construction of length 1! + 2! + ... + n!."""
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
    return word


def _factorial(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def _build_perm_index(n):
    """Build a dict mapping permutation tuples to unique IDs, and return list of all perms."""
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    return all_perms, perm_to_id


def _init_counts(word, n, perm_to_id):
    """Initialize count array: for each permutation, how many windows cover it."""
    num_perms = _factorial(n)
    counts = [0] * num_perms
    for i in range(len(word) - n + 1):
        p = tuple(word[i:i + n])
        pid = perm_to_id.get(p)
        if pid is not None:
            counts[pid] += 1
    return counts


def _is_all_covered(counts):
    """Check if all permutations have count >= 1."""
    for c in counts:
        if c == 0:
            return False
    return True


def _try_remove_chars_fast(word, n, time_limit):
    """
    Shorten superpermutation using fast count-tracking deletion.
    Each deletion attempt is O(n) instead of O(n * L).
    """
    word = list(word)
    target_perms = _factorial(n)
    deadline = time.time() + time_limit
    
    # Build permutation index
    all_perms, perm_to_id = _build_perm_index(n)
    num_perms = target_perms
    
    # Initialize counts
    counts = _init_counts(word, n, perm_to_id)
    
    if not _is_all_covered(counts):
        return word
    
    best_len = len(word)
    
    # Fast deletion: try removing each position, use incremental count updates
    improved = True
    while improved:
        if time.time() > deadline - 2:
            break
        improved = False
        
        for i in range(len(word) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            if len(word) - 1 < n:
                break
            
            L = len(word)
            
            # Save state for rollback
            old_counts = []
            
            # Step 1: Decrement counts for old windows that include position i
            # Old windows containing position i: start k where k <= i < k+n
            # i.e., k in [max(0, i-n+1), min(i, L-n)]
            k_start = max(0, i - n + 1)
            k_end = min(i, L - n)
            
            for k in range(k_start, k_end + 1):
                p = tuple(word[k:k + n])
                pid = perm_to_id.get(p)
                if pid is not None:
                    old_counts.append((k, pid))
                    counts[pid] -= 1
            
            # Step 2: Compute new windows formed after deletion
            # After removing position i, new word has length L-1
            # New windows affected: start k in [max(0, i-n+1), min(i-1, L-1-n)] in new word
            # Map to old word indices
            new_k_start = max(0, i - n + 1)
            new_k_end = min(i - 1, L - 1 - n)
            
            new_pids = []
            for k in range(new_k_start, new_k_end + 1):
                # In new word, window starting at k
                # If k < i: new_word[k:k+n] = word[k:i] + word[i+1:k+n+1]
                # If k >= i: new_word[k:k+n] = word[k+1:k+1+n]
                if k < i:
                    w1 = word[k:i]
                    w2 = word[i + 1:k + n + 1]
                    p = tuple(w1 + w2)
                else:
                    p = tuple(word[k + 1:k + 1 + n])
                
                pid = perm_to_id.get(p)
                if pid is not None:
                    counts[pid] += 1
                    new_pids.append(pid)
            
            # Step 3: Check if all permutations still covered
            valid = True
            for pid in range(num_perms):
                if counts[pid] == 0:
                    valid = False
                    break
            
            if valid:
                # Accept deletion
                del word[i]
                improved = True
                best_len = len(word)
            else:
                # Rollback
                # Restore old windows
                for k, pid in old_counts:
                    counts[pid] += 1
                # Remove new windows
                for pid in new_pids:
                    counts[pid] -= 1
    
    return word


def _try_swap_chars_fast(word, n, time_limit, perm_to_id, counts):
    """
    Try swapping characters to shorten (though swap doesn't shorten, it may enable
    subsequent deletions). Actually, let's focus on more aggressive local search.
    """
    # For now, this is a placeholder for future improvements
    return word


def construct(n):
    """
    Construct a superpermutation on n symbols.
    Returns a string of digits 1..n.
    """
    start_time = time.time()
    time_limit = 120  # evaluator limit
    
    if n <= 1:
        return "1" if n == 1 else ""
    
    # Step 1: Build classic superpermutation
    word = _classic_superperm(n)
    
    # Step 2: Try to shorten it with fast count-tracking
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 15  # safety margin
    
    if remaining > 5:
        word = _try_remove_chars_fast(word, n, remaining)
    
    return "".join(map(str, word))
# EVOLVE-BLOCK-END
