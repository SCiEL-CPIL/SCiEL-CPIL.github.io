import time

def construct():
    n = 256
    deadline = time.time() + 80  # 80s safety margin
    
    def is_valid(S, candidate):
        """Check if adding candidate to set S maintains the property."""
        # S is a sorted list
        if candidate in S:
            return False
        # Insert candidate in sorted position
        # For each a in S (and candidate if a < candidate), check all b <= c
        # where a < b <= c
        # We need: for all a < b <= c in S ∪ {candidate}, a ∤ (b+c)
        
        # Only new triples involving candidate need checking
        # Case 1: candidate is a (smallest)
        #   For all b, c in S with candidate < b <= c: candidate ∤ (b+c)
        # Case 2: candidate is b or c (larger)
        #   For all a in S with a < candidate, and for all b, c in S ∪ {candidate}
        #   with a < b <= c and at least one of b,c is candidate: a ∤ (b+c)
        
        S_set = set(S)
        
        # Case 1: candidate is the smallest element a
        for i in range(len(S)):
            b = S[i]
            if b <= candidate:
                continue
            for j in range(i, len(S)):
                c = S[j]
                if c < b:
                    continue
                if (b + c) % candidate == 0:
                    return False
        
        # Case 2: candidate is b or c
        # Subcase 2a: candidate is c (largest), b is in S, a is in S, a < b <= candidate
        for i in range(len(S)):
            a = S[i]
            if a >= candidate:
                continue
            for j in range(i + 1, len(S)):
                b = S[j]
                if b <= a:
                    continue
                if b > candidate:
                    continue
                # a < b <= candidate, c = candidate
                if (b + candidate) % a == 0:
                    return False
        
        # Subcase 2b: candidate is b, c is in S, a is in S, a < candidate <= c
        for i in range(len(S)):
            a = S[i]
            if a >= candidate:
                continue
            for j in range(i + 1, len(S)):
                c = S[j]
                if c < candidate:
                    continue
                # a < candidate <= c, b = candidate
                if (candidate + c) % a == 0:
                    return False
        
        # Subcase 2c: candidate is b=c
        for i in range(len(S)):
            a = S[i]
            if a >= candidate:
                continue
            if (2 * candidate) % a == 0:
                return False
        
        return True
    
    # Start with the known good interval [171, 256]
    S = list(range(171, n + 1))
    S_set = set(S)
    
    # Try to add elements from below 171, in descending order (larger first, fewer constraints)
    best_S = list(S)
    best_size = len(S)
    
    # Greedy: try adding each candidate from 170 down to 1
    for candidate in range(170, 0, -1):
        if time.time() > deadline:
            break
        if is_valid(S, candidate):
            S.append(candidate)
            S_set.add(candidate)
            S.sort()
            if len(S) > best_size:
                best_size = len(S)
                best_S = list(S)
    
    # Try local search: remove one element, try to add two
    if time.time() < deadline - 10:
        S = list(best_S)
        for remove_idx in range(len(S)):
            if time.time() > deadline - 5:
                break
            removed = S[remove_idx]
            S_temp = S[:remove_idx] + S[remove_idx+1:]
            S_temp_set = set(S_temp)
            
            added_count = 0
            new_elements = []
            for candidate in range(1, n + 1):
                if time.time() > deadline - 5:
                    break
                if candidate in S_temp_set:
                    continue
                if is_valid(S_temp, candidate):
                    S_temp.append(candidate)
                    S_temp_set.add(candidate)
                    S_temp.sort()
                    added_count += 1
                    new_elements.append(candidate)
            
            if len(S_temp) > len(best_S):
                best_S = list(S_temp)
                best_size = len(S_temp)
    
    return best_S
