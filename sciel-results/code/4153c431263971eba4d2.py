import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 256
    
    # Precompute conflicts using bitsets for O(1) intersection checks
    # conflict_bits[i] is an integer where bit j is set if i and j conflict
    conflict_bits = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = math.gcd(i, j)
            x = i // g
            y = j // g
            if g % (x + y) == 0:
                conflict_bits[i] |= (1 << j)
                conflict_bits[j] |= (1 << i)
    
    # Helper: check if element e conflicts with any element in set S
    def has_conflict(e, S):
        bits = conflict_bits[e]
        for s in S:
            if bits & (1 << s):
                return True
        return False
    
    # Helper: check if element e conflicts with any element in set S using bitset
    def has_conflict_bits(e, S_bits):
        return (conflict_bits[e] & S_bits) != 0
    
    # Build S_bits from a set
    def set_to_bits(S):
        bits = 0
        for x in S:
            bits |= (1 << x)
        return bits
    
    # Initial set: all odds
    S = set(range(1, N + 1, 2))
    S_bits = set_to_bits(S)
    
    # List of evens not in S, sorted by conflict count (fewer conflicts first)
    evens = [i for i in range(2, N + 1, 2) if i not in S]
    evens.sort(key=lambda x: bin(conflict_bits[x]).count('1'))
    
    # Greedy addition of non-conflicting evens
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_evens = []
        for e in evens:
            if time.time() > deadline:
                break
            if not has_conflict_bits(e, S_bits):
                S.add(e)
                S_bits |= (1 << e)
                changed = True
            else:
                new_evens.append(e)
        evens = new_evens
    
    # Local Search: try to improve by swapping elements
    def try_local_search(S, S_bits, evens, deadline):
        """Try to improve S by local search (swaps)."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try each element in S for removal
            for s_elem in list(S):
                if time.time() > deadline:
                    return S, S_bits, evens
                
                # If we remove s_elem, which evens become available?
                removable_bits = S_bits ^ (1 << s_elem)
                
                # Find evens that are now free (no conflict with S \ {s_elem})
                free_evens = []
                for e in evens:
                    if not has_conflict_bits(e, removable_bits):
                        free_evens.append(e)
                
                if not free_evens:
                    continue
                
                # Try to greedily add as many free evens as possible
                # Sort by number of conflicts to prioritize "easy" ones
                free_evens.sort(key=lambda x: bin(conflict_bits[x]).count('1'))
                
                new_S_bits = removable_bits
                added = 0
                for e in free_evens:
                    if time.time() > deadline:
                        break
                    if not has_conflict_bits(e, new_S_bits):
                        new_S_bits |= (1 << e)
                        added += 1
                
                # If we added more elements than we removed (1), it's an improvement
                if added > 1:
                    # Reconstruct S from new_S_bits
                    new_S = set()
                    for i in range(1, N + 1):
                        if new_S_bits & (1 << i):
                            new_S.add(i)
                    S = new_S
                    S_bits = new_S_bits
                    evens = [e for e in evens if e not in new_S]
                    improved = True
                    break  # Restart the loop with the new S
        
        return S, S_bits, evens
    
    S, S_bits, evens = try_local_search(S, S_bits, evens, deadline)
    
    # Second pass: try removing pairs of elements to free up more evens
    def try_pair_removal(S, S_bits, evens, deadline):
        """Try removing pairs of elements to free up more evens."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each pair of elements
            S_list = list(S)
            for i in range(len(S_list)):
                if time.time() > deadline:
                    return S, S_bits, evens
                for j in range(i + 1, len(S_list)):
                    if time.time() > deadline:
                        return S, S_bits, evens
                    
                    s1, s2 = S_list[i], S_list[j]
                    removable_bits = S_bits ^ (1 << s1) ^ (1 << s2)
                    
                    # Find evens that are now free
                    free_evens = []
                    for e in evens:
                        if not has_conflict_bits(e, removable_bits):
                            free_evens.append(e)
                    
                    if not free_evens:
                        continue
                    
                    # Try to greedily add as many free evens as possible
                    free_evens.sort(key=lambda x: bin(conflict_bits[x]).count('1'))
                    
                    new_S_bits = removable_bits
                    added = 0
                    for e in free_evens:
                        if time.time() > deadline:
                            break
                        if not has_conflict_bits(e, new_S_bits):
                            new_S_bits |= (1 << e)
                            added += 1
                    
                    # If we added more elements than we removed (2), it's an improvement
                    if added > 2:
                        # Reconstruct S from new_S_bits
                        new_S = set()
                        for x in range(1, N + 1):
                            if new_S_bits & (1 << x):
                                new_S.add(x)
                        S = new_S
                        S_bits = new_S_bits
                        evens = [e for e in evens if e not in new_S]
                        improved = True
                        break
                if improved:
                    break
        
        return S, S_bits, evens
    
    S, S_bits, evens = try_pair_removal(S, S_bits, evens, deadline)
    
    # Third pass: try removing triples of elements
    def try_triple_removal(S, S_bits, evens, deadline):
        """Try removing triples of elements to free up more evens."""
        improved = True
        while improved and time.time() < deadline:
            improved = False
            S_list = list(S)
            n = len(S_list)
            # Only try if S is not too large (to keep runtime manageable)
            if n > 140:
                break
            
            for i in range(n):
                if time.time() > deadline:
                    return S, S_bits, evens
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        return S, S_bits, evens
                    for k in range(j + 1, n):
                        if time.time() > deadline:
                            return S, S_bits, evens
                        
                        s1, s2, s3 = S_list[i], S_list[j], S_list[k]
                        removable_bits = S_bits ^ (1 << s1) ^ (1 << s2) ^ (1 << s3)
                        
                        # Find evens that are now free
                        free_evens = []
                        for e in evens:
                            if not has_conflict_bits(e, removable_bits):
                                free_evens.append(e)
                        
                        if len(free_evens) < 4:
                            continue
                        
                        # Try to greedily add as many free evens as possible
                        free_evens.sort(key=lambda x: bin(conflict_bits[x]).count('1'))
                        
                        new_S_bits = removable_bits
                        added = 0
                        for e in free_evens:
                            if time.time() > deadline:
                                break
                            if not has_conflict_bits(e, new_S_bits):
                                new_S_bits |= (1 << e)
                                added += 1
                        
                        # If we added more elements than we removed (3), it's an improvement
                        if added > 3:
                            # Reconstruct S from new_S_bits
                            new_S = set()
                            for x in range(1, N + 1):
                                if new_S_bits & (1 << x):
                                    new_S.add(x)
                            S = new_S
                            S_bits = new_S_bits
                            evens = [e for e in evens if e not in new_S]
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
        
        return S, S_bits, evens
    
    S, S_bits, evens = try_triple_removal(S, S_bits, evens, deadline)
    
    # Final greedy pass after local search
    changed = True
    while changed and time.time() < deadline:
        changed = False
        new_evens = []
        for e in evens:
            if time.time() > deadline:
                break
            if not has_conflict_bits(e, S_bits):
                S.add(e)
                S_bits |= (1 << e)
                changed = True
            else:
                new_evens.append(e)
        evens = new_evens
    
    return list(S)
# EVOLVE-BLOCK-END
