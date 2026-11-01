"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from math import factorial

# EVOLVE-BLOCK-START

def _standard_superperm(n):
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
            # find max overlap
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
    return "".join(map(str, word))


def _validate(word, n):
    """Check if word contains all n! permutations of 1..n as n-consecutive symbols."""
    if n <= 1:
        return len(word) == n
    if len(word) < n:
        return False
    required = set(itertools.permutations(range(1, n + 1)))
    found = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n:
            found.add(window)
    return found == required


def _greedy_superperm(n, time_limit=100):
    """
    Greedy construction: maintain a set of permutations still needed.
    At each step, pick the next symbol that maximizes the number of new permutations
    completed (those that end at the current position). Tie-break by frequency in
    remaining permutations to keep options open.
    """
    if n <= 1:
        return "".join(str(i) for i in range(1, n + 1))
    
    all_perms = list(itertools.permutations(range(1, n + 1)))
    perm_to_idx = {p: i for i, p in enumerate(all_perms)}
    remaining = [True] * len(all_perms)
    remaining_count = len(all_perms)
    
    # For each position in the window (0..n-1), maintain counts of how many remaining
    # permutations have symbol s at that position. This helps with tie-breaking.
    # pos_counts[pos][symbol] = count of remaining perms with symbol at pos
    pos_counts = [[0] * (n + 1) for _ in range(n)]
    for p in all_perms:
        for pos in range(n):
            pos_counts[pos][p[pos]] += 1
    
    # Build the word greedily
    word = []
    # Start with a random-ish or fixed prefix
    # Use the first permutation as starting point
    current = list(all_perms[0])
    word.extend(current)
    
    # Mark the first permutation as found
    remaining[perm_to_idx[tuple(current)]] = False
    remaining_count -= 1
    for pos in range(n):
        pos_counts[pos][current[pos]] -= 1
    
    start_time = time.time()
    
    while remaining_count > 0:
        if time.time() - start_time > time_limit:
            break
        
        # Current window is word[-n:]
        if len(word) < n:
            # shouldn't happen since we started with n
            break
        current_window = tuple(word[-n:])
        
        # For each candidate symbol s (1..n), compute:
        # 1. How many new permutations would be completed (the window ending with s)
        # 2. Tie-break: sum of pos_counts for the shifted window
        best_s = None
        best_score = -1
        best_tiebreak = -1
        
        current_window_list = list(current_window)
        
        for s in range(1, n + 1):
            # The new window would be current_window[1:] + (s,)
            new_window = tuple(current_window_list[1:] + [s])
            
            # Check if this window is a valid permutation (all distinct)
            if len(set(new_window)) != n:
                continue
            
            idx = perm_to_idx.get(new_window, -1)
            if idx == -1:
                continue
            
            # Score: 1 if this permutation is still remaining (we complete it)
            score = 1 if remaining[idx] else 0
            
            # Tie-break: sum of pos_counts for the new window positions
            # This estimates how "central" this window is among remaining perms
            tiebreak = 0
            for pos in range(n):
                sym = new_window[pos]
                tiebreak += pos_counts[pos][sym]
            
            if score > best_score or (score == best_score and tiebreak > best_tiebreak):
                best_score = score
                best_tiebreak = tiebreak
                best_s = s
        
        if best_s is None:
            # No valid next symbol found; this shouldn't happen in a valid construction
            # but if it does, we're stuck
            break
        
        word.append(best_s)
        
        # Update remaining and pos_counts
        new_window = tuple(word[-n:])
        if len(set(new_window)) == n:
            idx = perm_to_idx.get(new_window, -1)
            if idx != -1 and remaining[idx]:
                remaining[idx] = False
                remaining_count -= 1
                for pos in range(n):
                    pos_counts[pos][new_window[pos]] -= 1
    
    return "".join(map(str, word))


def _try_improve_with_local_search(word, n, time_limit=50):
    """
    Try to shorten the word by removing symbols where possible.
    A symbol at position i can be removed if no permutation is lost,
    i.e., all n-windows that include position i are still present elsewhere.
    """
    if n <= 1:
        return word
    
    word_list = list(word)
    start_time = time.time()
    improved = True
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Compute which permutations are present
        all_perms = set(itertools.permutations(range(1, n + 1)))
        
        def get_perm_set(wl):
            ps = set()
            for i in range(len(wl) - n + 1):
                window = tuple(wl[i:i + n])
                if len(set(window)) == n:
                    ps.add(window)
            return ps
        
        current_perms = get_perm_set(word_list)
        
        # Try removing each position
        for i in range(len(word_list)):
            if (time.time() - start_time) > time_limit:
                break
            
            # Remove position i
            new_list = word_list[:i] + word_list[i+1:]
            new_perms = get_perm_set(new_list)
            
            if new_perms == current_perms:
                word_list = new_list
                improved = True
                break  # restart from beginning since indices shifted
    
    return "".join(map(str, word_list))


def construct(n):
    if n <= 1:
        return "".join(str(i) for i in range(1, n + 1))
    
    start_time = time.time()
    deadline = 110  # safety margin before 120s limit
    
    # For small n, use the exact optimal construction
    if n <= 4:
        return _standard_superperm(n)
    
    # For n=5,6,7: use greedy + local search
    # The greedy approach should produce a valid superpermutation
    # and local search can trim it.
    
    # Try greedy construction
    greedy_word = _greedy_superperm(n, time_limit=60)
    
    # Validate
    if not _validate(greedy_word, n):
        # Fallback to standard if greedy failed
        return _standard_superperm(n)
    
    # Try to improve with local search
    if time.time() - start_time < deadline - 30:
        improved_word = _try_improve_with_local_search(greedy_word, n, time_limit=30)
        if _validate(improved_word, n):
            return improved_word
    
    return greedy_word

# EVOLVE-BLOCK-END
