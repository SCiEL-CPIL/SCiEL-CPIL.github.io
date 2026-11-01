import time
import random
import sys

def construct(n):
    start_time = time.time()
    deadline = 110  # Leave 10s margin
    
    if n == 1:
        return "1"
    
    # Precompute factorial numbers or just use tuples
    # We will use integers 0..n!-1 for permutations?
    # Or just tuples of ints 1..n.
    # Tuples are hashable.
    
    # Generate all permutations
    # For n=7, 5040 permutations.
    # We can generate them on the fly or precompute.
    # Precomputation of next_permutation logic might be faster?
    # Let's just generate all perms as tuples.
    
    from itertools import permutations
    
    all_perms = list(permutations(range(1, n+1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    total_perms = len(all_perms)
    
    # Precompute for each suffix (tuple of n-1 elements)
    # 1. The forced symbol
    # 2. The resulting permutation ID
    # Suffix is a tuple of n-1 distinct elements.
    # There are n! such suffixes.
    
    # Map suffix to (forced_symbol, perm_id)
    suffix_info = {}
    for p in all_perms:
        suffix = p[:-1]
        forced_sym = p[-1]
        # The permutation is p itself.
        suffix_info[suffix] = (forced_sym, perm_to_id[p])
        
    # Note: Multiple permutations can have the same suffix?
    # No, suffix is p[:-1]. Since p is a permutation, p[:-1] is a sequence of n-1 distinct elements.
    # Is the mapping suffix -> p unique?
    # If suffix is (1,2,3) for n=4.
    # Perm 1234 has suffix 123.
    # Perm 1234... wait.
    # Suffix is the first n-1 elements.
    # For a fixed suffix, the last element is forced to be the missing one.
    # So yes, each suffix corresponds to exactly ONE permutation.
    
    best_word = None
    best_len = float('inf')
    
    # Run multiple greedy attempts
    max_attempts = 200
    
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
            
        random.shuffle(all_perms) # Randomize start? Or just pick random start perm.
        start_perm = random.choice(all_perms)
        
        word = list(start_perm)
        visited = set([perm_to_id[start_perm]])
        current_suffix = tuple(start_perm[:-1])
        
        # We need to track the actual word to return it.
        # We iterate.
        
        success = False
        while len(visited) < total_perms:
            if time.time() > deadline:
                break
                
            # Check forced move
            forced_sym, next_perm_id = suffix_info[current_suffix]
            
            if next_perm_id not in visited:
                # Take the free move
                word.append(forced_sym)
                visited.add(next_perm_id)
                # Update suffix
                # New suffix is word[-(n-1):]
                # Which is current_suffix[1:] + (forced_sym,)
                current_suffix = current_suffix[1:] + (forced_sym,)
            else:
                # Stuck. Must escape.
                # Find alternative symbols
                # The set of symbols in current_suffix
                suffix_set = set(current_suffix)
                # The forced symbol is the one NOT in suffix_set
                # Alternatives are all symbols in suffix_set?
                # No, we can append ANY symbol 1..n.
                # But appending forced_sym leads to visited perm.
                # Appending other symbols creates non-perm windows.
                
                candidates = []
                for c in range(1, n+1):
                    if c == forced_sym:
                        continue
                    new_suffix = current_suffix[1:] + (c,)
                    # Check if this new suffix leads to an unvisited perm in next step
                    # We need to check if new_suffix is in suffix_info
                    # It should be, as it's a valid sequence of n-1 distinct elements?
                    # Wait. current_suffix has distinct elements.
                    # new_suffix = current_suffix[1:] + (c,).
                    # Are the elements in new_suffix distinct?
                    # current_suffix[1:] has n-2 distinct elements.
                    # c is added.
                    # If c is in current_suffix[1:], then new_suffix has duplicates.
                    # A suffix with duplicates CANNOT be the prefix of a permutation.
                    # So if c is in current_suffix[1:], the next window (after next step) will NOT be a permutation.
                    # This might be a "double waste".
                    # We prefer c such that new_suffix has distinct elements.
                    
                    if c in current_suffix[1:]:
                        # This leads to a suffix with duplicates.
                        # The next forced step will fail (suffix not in suffix_info).
                        # This is a "deep" escape.
                        score = 0
                    else:
                        # New suffix is valid.
                        if new_suffix in suffix_info:
                            _, next_next_perm_id = suffix_info[new_suffix]
                            if next_next_perm_id not in visited:
                                score = 1
                            else:
                                score = 0
                        else:
                            score = 0
                            
                    candidates.append((score, c))
                
                # Sort by score desc, then random
                candidates.sort(key=lambda x: -x[0])
                # Pick from top candidates (score=1) if any, else random from rest
                top = [c for s, c in candidates if s == 1]
                if not top:
                    top = [c for _, c in candidates]
                    
                c = random.choice(top)
                word.append(c)
                current_suffix = current_suffix[1:] + (c,)
                
        # Check if we succeeded
        if len(visited) == total_perms:
            if len(word) < best_len:
                best_len = len(word)
                best_word = word.copy()
        
    if best_word:
        return "".join(map(str, best_word))
    else:
        # Fallback to recursive construction if greedy failed
        # (Should not happen if we run enough attempts, but safe to have)
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
        return "".join(map(str, word))
