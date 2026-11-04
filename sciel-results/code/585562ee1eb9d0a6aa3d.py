import itertools
import time

def construct(n):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Generate permutations
    perms = list(itertools.permutations(range(1, n+1)))
    N = len(perms)
    
    # Baseline recursive construction
    def recursive_superperm(n):
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

    baseline_word = recursive_superperm(n)
    baseline_len = len(baseline_word)
    
    # Greedy approach
    # Precompute prefixes for all permutations?
    # We need to maintain sets of unused indices for each prefix length.
    
    # maps[k] is a dict: prefix_tuple -> set of indices
    maps = [dict() for _ in range(n)] # k=0 to n-1
    
    for i, p in enumerate(perms):
        for k in range(n):
            pref = p[:k]
            if pref not in maps[k]:
                maps[k][pref] = set()
            maps[k][pref].add(i)
            
    used = [False] * N
    # Start with first permutation
    current_idx = 0
    used[0] = True
    current_word = list(perms[0])
    result = current_word[:]
    
    # Remove first permutation from all maps
    p0 = perms[0]
    for k in range(n):
        pref = p0[:k]
        maps[k][pref].discard(0)
        
    remaining = N - 1
    
    while remaining > 0:
        if time.time() > deadline:
            break
            
        best_k = -1
        best_idx = -1
        
        # Try to find max overlap
        # Current suffix of length k
        # We check k from n-1 down to 0
        for k in range(n-1, -1, -1):
            if k == 0:
                # Any unused permutation
                # Find any non-empty set in maps[0]
                # Actually maps[0][()] contains all unused
                if maps[0].get((), set()):
                    # Pick the first one
                    # Sets don't have order, but we can just pop one
                    # To be deterministic, maybe pick min index?
                    # Popping from a set is O(1) but arbitrary.
                    # Let's just pick one.
                    best_idx = next(iter(maps[0][()]))
                    best_k = 0
                    break
            else:
                suffix = tuple(result[-k:])
                if suffix in maps[k]:
                    s = maps[k][suffix]
                    if s:
                        # Pick one
                        best_idx = next(iter(s))
                        best_k = k
                        break
                        
        if best_idx == -1:
            # Should not happen if remaining > 0
            break
            
        # Use best_idx
        p_next = perms[best_idx]
        
        # Append non-overlapping part
        if best_k > 0:
            result.extend(p_next[best_k:])
        else:
            result.extend(p_next)
            
        # Update state
        used[best_idx] = True
        remaining -= 1
        
        # Remove best_idx from all maps
        for k in range(n):
            pref = p_next[:k]
            if pref in maps[k]:
                maps[k][pref].discard(best_idx)
                
        current_word = list(p_next) # Not really needed for logic, just for clarity
        
    greedy_len = len(result)
    
    if greedy_len < baseline_len:
        return "".join(map(str, result))
    else:
        return "".join(map(str, baseline_word))
