import itertools
import time

def _standard_superperm(n):
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
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    return word

def _greedy_constrained_superperm(n, deadline):
    """
    Greedy superpermutation: build a word over symbols 1..n such that every
    permutation of 1..n appears as a consecutive window of length n.
    
    Strategy: maintain the current suffix of length n-1. At each step, choose
    the next symbol that (a) is not already in the suffix, and (b) among valid
    choices, prefer the one that leads to a suffix with the fewest remaining
    valid extensions (most-constrained-first). This minimizes dead-ends.
    
    We track which n-permutations have been "seen" (completed). A permutation
    is completed when it appears as a window. We need all n! permutations.
    """
    total_perms = 1
    for k in range(2, n + 1):
        total_perms *= k
    
    seen_perms = set()
    # Start with the first n-1 symbols: 1, 2, ..., n-1
    suffix = list(range(1, n))  # length n-1
    word = list(suffix)
    seen_perms.add(tuple(suffix + [n]))  # the first permutation
    
    # Precompute for each (n-1)-tuple suffix, the set of valid next symbols
    # A valid next symbol x: x not in suffix, and tuple(suffix + [x]) is a valid perm
    # Actually every (n-1)-tuple with distinct elements from 1..n has exactly one
    # valid completion (the missing symbol). So "most constrained" is trivial here.
    
    # Better strategy: think in terms of which permutations are "reachable" next.
    # From suffix s (length n-1), the only valid next symbol is the missing one.
    # So the walk is forced! Each step is determined by the current suffix.
    # 
    # Wait - that means the greedy walk is deterministic and will just cycle
    # through permutations. The issue is that we might revisit a permutation
    # before seeing all of them.
    #
    # The key insight: when we have a suffix of length n-1, the next symbol is
    # forced. But we can choose WHICH suffix to be in by controlling the path.
    # 
    # Actually, let's think differently. The word is built one symbol at a time.
    # At position i, the suffix of length n-1 determines the next symbol.
    # So the entire walk is determined by the initial n-1 symbols.
    #
    # This means we need to find a good initial sequence and the walk will
    # either cover all permutations or not. For n=7, the standard construction
    # already does this. The greedy approach needs a different framing.
    #
    # Alternative: Build the word greedily, but at each step consider ALL possible
    # next symbols (not just the forced one) and pick the one that maximizes
    # new permutations seen while minimizing "wasted" extensions.
    #
    # Let's use: at each step, for each candidate next symbol x (1..n, x not in
    # last n-1 symbols to form a valid window), check if it creates a new permutation.
    # Prefer symbols that create new permutations. Among those, prefer the one
    # whose resulting suffix has the fewest "useful" future extensions.
    
    if n < 4:
        return _standard_superperm(n)
    
    # Track: for each possible suffix (n-1)-tuple, how many unseen permutations
    # can be reached from it (including the immediate one).
    # This is expensive to compute exactly, so use a heuristic:
    # "constrained" = the number of distinct permutations that share this suffix
    # as their first n-1 symbols... but each suffix has exactly one completion.
    
    # Simpler heuristic that works: greedy with backtracking on a priority basis.
    # Use the standard construction as a base, then try local improvements.
    
    # For now, return standard construction (this function is a placeholder
    # for the greedy approach; the real improvement comes from the pruning pass)
    return _standard_superperm(n)

def construct(n):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate standard superpermutation
    word = _standard_superperm(n)
    
    # If n is small, return immediately
    if n < 7:
        return "".join(map(str, word))
        
    # For n=7, try to optimize
    # Convert to list of ints for easy manipulation
    w = word[:]
    
    # Pre-compute all permutations
    all_perms = set(itertools.permutations(range(1, n+1)))
    total_perms = len(all_perms)
    
    # Build a helper to get the window at a given start index
    def get_window(start, w_list, n_val):
        if start < 0 or start + n_val > len(w_list):
            return None
        return tuple(w_list[start:start+n_val])
    
    # Count occurrences of each permutation in the current word
    perm_counts = {}
    for i in range(len(w) - n + 1):
        p = tuple(w[i:i+n])
        if p in all_perms:
            perm_counts[p] = perm_counts.get(p, 0) + 1
            
    # Check if all are present (should be)
    if len(perm_counts) < total_perms:
        return "".join(map(str, w))
        
    # Strategy: Try to remove characters from the end first (safest), 
    # then try removing from various positions.
    # Key insight: When we remove a character at position i, the windows affected are:
    # - Old windows starting at i-n+1 through i (these are destroyed/changed)
    # - New windows may form at the boundary
    # 
    # For simplicity and correctness, let's use a two-pass approach:
    # Pass 1: Remove from the end as much as possible
    # Pass 2: Try removing from middle positions (left to right)
    
    # Pass 1: Remove from end
    while len(w) > n:
        if time.time() > deadline:
            break
        last_start = len(w) - n
        p = tuple(w[last_start:])
        if p in all_perms and perm_counts.get(p, 0) == 1:
            break
        else:
            if p in all_perms:
                perm_counts[p] -= 1
                if perm_counts[p] == 0:
                    del perm_counts[p]
            w.pop()
    
    # Pass 2: Try removing from left to right
    # For each position i from 0 to len(w)-n, try to remove w[i]
    # We need to be careful about maintaining perm_counts consistency
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(w) - n + 1):
            if time.time() > deadline:
                break
            
            # Check if removing w[i] is safe
            # Windows that include w[i]: start indices from max(0, i-n+1) to min(i, len(w)-n)
            removable = True
            affected_starts = list(range(max(0, i-n+1), min(i, len(w)-n) + 1))
            
            for start in affected_starts:
                p = tuple(w[start:start+n])
                if p in all_perms:
                    if perm_counts.get(p, 0) == 1:
                        removable = False
                        break
            
            if removable:
                # Remove w[i] and update counts
                # First, decrement counts for old affected windows
                for start in affected_starts:
                    p = tuple(w[start:start+n])
                    if p in all_perms:
                        perm_counts[p] -= 1
                        if perm_counts[p] == 0:
                            del perm_counts[p]
                
                # Remove the character
                w.pop(i)
                
                # After removal, new windows may form. The windows starting at 
                # positions affected_starts (in the new string) need to be added back.
                # In the new string, the positions are the same for starts < i, 
                # but for starts >= i, they shift. Actually, since we removed position i,
                # windows starting at positions in affected_starts (which are all < i or == i)
                # now have different content. We need to add the new windows.
                
                # The new affected windows in the new string are at the same start positions
                # (since we removed position i, and all affected starts are <= i, 
                # the new windows at those starts are different)
                for start in affected_starts:
                    if start + n <= len(w):  # Make sure the window fits
                        p = tuple(w[start:start+n])
                        if p in all_perms:
                            perm_counts[p] = perm_counts.get(p, 0) + 1
                
                changed = True
                break  # Restart the scan from the beginning
    
    # Pass 3: Multi-pass removal with wider affected window check
    # Sometimes a single removal enables another removal that was previously blocked.
    # Do several full passes.
    for _pass in range(5):
        if time.time() > deadline:
            break
        pass_changed = True
        while pass_changed and time.time() < deadline:
            pass_changed = False
            for i in range(len(w) - n + 1):
                if time.time() > deadline:
                    break
                removable = True
                affected_starts = list(range(max(0, i-n+1), min(i, len(w)-n) + 1))
                for start in affected_starts:
                    p = tuple(w[start:start+n])
                    if p in all_perms and perm_counts.get(p, 0) == 1:
                        removable = False
                        break
                if removable:
                    for start in affected_starts:
                        p = tuple(w[start:start+n])
                        if p in all_perms:
                            perm_counts[p] -= 1
                            if perm_counts[p] == 0:
                                del perm_counts[p]
                    w.pop(i)
                    for start in affected_starts:
                        if start + n <= len(w):
                            p = tuple(w[start:start+n])
                            if p in all_perms:
                                perm_counts[p] = perm_counts.get(p, 0) + 1
                    pass_changed = True
                    break
    
    return "".join(map(str, w))
