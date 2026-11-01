import time
import itertools

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

def _validate_superperm(word, n):
    """Check if word contains all permutations of 1..n as consecutive substrings."""
    if len(word) < n:
        return False
    perms = set(itertools.permutations(range(1, n + 1)))
    found = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            found.add(window)
    return found == perms

def _local_optimize(word, n, time_limit=100):
    """Try to shorten the superpermutation via local moves."""
    start_time = time.time()
    best_word = list(word)
    best_len = len(best_word)
    
    # Try removing individual symbols and checking validity
    # This is O(n * len) per check, but we can be smarter
    # For n=7, len~5913, so ~41k checks per pass, each O(5040) to verify
    # That's ~200M operations, too slow. Need smarter approach.
    
    # Alternative: try swapping adjacent symbols, or removing a symbol
    # and checking if the resulting word still contains all perms.
    
    # A faster check: precompute which windows are "critical"
    # Actually, let's just try a few random removals and validate.
    
    import random
    rng = random.Random(42)
    
    # Try to find removable positions
    # A position i is removable if word[i] == word[i-1] or word[i] == word[i+1]
    # No, that's not sufficient. We need to check if all perms still appear.
    
    # Let's try a different approach: look for patterns like "aba" where we can remove the middle 'b'
    # if the surrounding context still allows all perms.
    
    # Actually, the most effective local move for superpermutations is:
    # If we have ...X Y Z... and X==Z, we might be able to remove Y.
    # But this is complex to verify.
    
    # Given time constraints, let's just do a few passes of:
    # 1. Try removing each symbol one at a time (in random order)
    # 2. Validate the result
    # 3. If valid, keep it and continue
    
    max_passes = 5
    for pass_num in range(max_passes):
        if time.time() - start_time > time_limit:
            break
        removed_any = False
        positions = list(range(1, len(best_word) - 1))  # don't remove first or last
        rng.shuffle(positions)
        
        for pos in positions:
            if time.time() - start_time > time_limit:
                break
            if len(best_word) <= best_len - 8:  # we want to reach ~5905
                pass
            
            # Try removing position pos
            trial = best_word[:pos] + best_word[pos+1:]
            if len(trial) < best_len:
                # Quick check: does it still have enough length?
                if len(trial) >= n:
                    # Full validation
                    if _validate_superperm(trial, n):
                        best_word = trial
                        best_len = len(best_word)
                        removed_any = True
                        break  # restart the pass after a removal
        
        if not removed_any:
            break
    
    return best_word

def construct(n):
    start_time = time.time()
    
    if n <= 1:
        return "1"
    
    # Use standard recursive construction
    word = _standard_superperm(n)
    
    # For n <= 4, the standard construction is optimal
    if n <= 4:
        return "".join(map(str, word))
    
    # For n >= 5, try to optimize
    if n >= 5:
        # Time budget: leave 10s for safety
        time_for_opt = min(100, 140 - (time.time() - start_time))
        if time_for_opt > 5:
            word = _local_optimize(word, n, time_limit=time_for_opt)
    
    return "".join(map(str, word))
