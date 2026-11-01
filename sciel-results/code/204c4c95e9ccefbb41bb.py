import time
import itertools
import random

def _standard_superperm(n):
    """Standard recursive superpermutation construction. Returns list of ints."""
    if n == 1:
        return [1]
    if n == 2:
        return [1, 2, 1]
    word = _standard_superperm(n - 1)
    perms = []
    seen = set()
    for i in range(len(word) - (n - 2)):
        window = tuple(word[i:i + n - 1])
        if len(set(window)) == n - 1 and window not in seen:
            seen.add(window)
            perms.append(window)
    new_word = []
    for p in perms:
        block = list(p) + [n] + list(p)
        max_ov = 0
        for ov in range(min(len(new_word), len(block)), -1, -1):
            if ov == 0:
                max_ov = 0
                break
            if new_word[-ov:] == block[:ov]:
                max_ov = ov
                break
        new_word.extend(block[max_ov:])
    return new_word

def _build_perm_index(n):
    """Build a mapping from permutation tuple to index for fast lookup."""
    idx = {}
    i = 0
    for perm in itertools.permutations(range(1, n + 1)):
        idx[perm] = i
        i += 1
    return idx, i  # mapping, count

def _count_perm_occurrences(word, n, perm_idx, perm_count):
    """Count how many times each permutation appears as a consecutive window."""
    counts = [0] * perm_count
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            pidx = perm_idx.get(window, -1)
            if pidx >= 0:
                counts[pidx] += 1
    return counts

def _validate_superperm(word, n):
    """Check if word contains all permutations of 1..n as consecutive substrings."""
    if len(word) < n:
        return False
    perm_idx, perm_count = _build_perm_index(n)
    counts = _count_perm_occurrences(word, n, perm_idx, perm_count)
    return all(c > 0 for c in counts)

def _local_optimize_fast(word, n, time_limit=100):
    """
    Fast local optimization using incremental coverage counting.
    Key insight: when we remove position i, only windows starting at positions
    i-n+1 through i are affected. We can track coverage counts incrementally.
    """
    start_time = time.time()
    best_word = list(word)
    best_len = len(best_word)
    
    perm_idx, perm_count = _build_perm_index(n)
    
    # Build initial coverage counts
    counts = _count_perm_occurrences(best_word, n, perm_idx, perm_count)
    # Verify all covered
    if not all(c > 0 for c in counts):
        return best_word
    
    rng = random.Random(42)
    max_passes = 50
    
    for pass_num in range(max_passes):
        if time.time() - start_time > time_limit:
            break
        
        removed_any = False
        # Try positions in random order, skip first and last
        positions = list(range(1, len(best_word) - 1))
        rng.shuffle(positions)
        
        for pos in positions:
            if time.time() - start_time > time_limit:
                break
            
            # Check if removing position pos keeps all perms covered
            # Windows affected: those that include position pos
            # Window starting at i includes positions i..i+n-1
            # Position pos is in window i iff i <= pos <= i+n-1, i.e., pos-n+1 <= i <= pos
            # But window must be valid: 0 <= i <= len(word)-n
            
            # Find all windows that include position pos
            # For each such window, decrement its count; if any hits 0, removal fails
            # Then restore counts if we keep the word
            
            affected_windows = []
            min_i = max(0, pos - n + 1)
            max_i = min(pos, len(best_word) - n)
            
            for i in range(min_i, max_i + 1):
                window = tuple(best_word[i:i + n])
                if len(set(window)) == n:
                    pidx = perm_idx.get(window, -1)
                    if pidx >= 0:
                        affected_windows.append((i, pidx))
            
            # Check if removal is safe
            safe = True
            for (i, pidx) in affected_windows:
                if counts[pidx] <= 1:
                    safe = False
                    break
            
            if safe:
                # Remove position pos
                # Update counts: decrement for affected windows
                for (i, pidx) in affected_windows:
                    counts[pidx] -= 1
                
                best_word.pop(pos)
                best_len = len(best_word)
                removed_any = True
                break  # restart pass after successful removal
        
        if not removed_any:
            break
    
    return best_word

def _local_optimize_greedy(word, n, time_limit=100):
    """
    Greedy local optimization: repeatedly find the position whose removal
    causes the fewest "critical" windows to lose coverage, and remove it
    if safe. Uses a priority-based approach.
    """
    start_time = time.time()
    best_word = list(word)
    best_len = len(best_word)
    
    perm_idx, perm_count = _build_perm_index(n)
    counts = _count_perm_occurrences(best_word, n, perm_idx, perm_count)
    
    if not all(c > 0 for c in counts):
        return best_word
    
    rng = random.Random(123)
    
    while time.time() - start_time < time_limit:
        # Find all removable positions (removal keeps all counts > 0)
        removable = []
        for pos in range(1, len(best_word) - 1):
            min_i = max(0, pos - n + 1)
            max_i = min(pos, len(best_word) - n)
            
            safe = True
            for i in range(min_i, max_i + 1):
                window = tuple(best_word[i:i + n])
                if len(set(window)) == n:
                    pidx = perm_idx.get(window, -1)
                    if pidx >= 0 and counts[pidx] <= 1:
                        safe = False
                        break
            
            if safe:
                removable.append(pos)
        
        if not removable:
            break
        
        # Pick a random removable position
        pos = rng.choice(removable)
        
        # Update counts
        min_i = max(0, pos - n + 1)
        max_i = min(pos, len(best_word) - n)
        for i in range(min_i, max_i + 1):
            window = tuple(best_word[i:i + n])
            if len(set(window)) == n:
                pidx = perm_idx.get(window, -1)
                if pidx >= 0:
                    counts[pidx] -= 1
        
        best_word.pop(pos)
        best_len = len(best_word)
    
    return best_word

def construct(n):
    start_time = time.time()
    
    if n <= 1:
        return "1"
    if n == 2:
        return "121"
    if n == 3:
        return "123121321"
    
    # Use standard recursive construction
    word = _standard_superperm(n)
    
    # For n <= 4, the standard construction is optimal
    if n <= 4:
        return "".join(map(str, word))
    
    # For n >= 5, try to optimize
    if n >= 5:
        # Time budget: leave 15s for safety
        elapsed = time.time() - start_time
        time_for_opt = min(120, 135 - elapsed)
        if time_for_opt > 5:
            # First pass: fast local optimization
            word = _local_optimize_fast(word, n, time_limit=time_for_opt * 0.6)
            
            # Second pass: greedy optimization with remaining time
            elapsed2 = time.time() - start_time
            time_for_opt2 = min(135 - elapsed2, 5)
            if time_for_opt2 > 1:
                word = _local_optimize_greedy(word, n, time_limit=time_for_opt2)
    
    return "".join(map(str, word))
