import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    # Precompute conflicts
    # Conflicts[a] is a list of tuples (b, c) such that {b,c} conflicts with a
    # We assume b < c for uniqueness
    N = 180
    conflicts = {i: [] for i in range(1, N + 1)}
    for a in range(1, N + 1):
        a2 = a * a
        # Find divisors u, v such that u * v = a2
        # b = a - u, c = a - v OR b = a + u, c = a + v
        # (a-b)(a-c) = a2
        # Let x = a-b, y = a-c. xy = a2.
        # b = a-x, c = a-y.
        # b, c must be in [1, N] and distinct from a and each other.
        for x in range(1, int(a2**0.5) + 1):
            if a2 % x == 0:
                y = a2 // x
                # Case 1: x, y positive
                b1 = a - x
                c1 = a - y
                if 1 <= b1 <= N and 1 <= c1 <= N and b1 != c1 and b1 != a and c1 != a:
                    pair = (min(b1, c1), max(b1, c1))
                    conflicts[a].append(pair)
                
                # Case 2: x, y negative (impossible as x,y >= 1 in loop)
                # But b, c can be > a.
                # If b > a, then a-b < 0. Let x' = b-a > 0.
                # Then (a-b) = -x'.
                # So (-x')(-y') = a2 => x'y' = a2.
                # b = a + x', c = a + y'.
                b2 = a + x
                c2 = a + y
                if 1 <= b2 <= N and 1 <= c2 <= N and b2 != c2 and b2 != a and c2 != a:
                    pair = (min(b2, c2), max(b2, c2))
                    conflicts[a].append(pair)
    
    # Remove duplicate pairs in conflicts[a]
    for a in conflicts:
        conflicts[a] = list(set(conflicts[a]))

    def is_valid(S):
        S_set = set(S)
        for a in S_set:
            for b, c in conflicts[a]:
                if b in S_set and c in S_set:
                    return False
        return True

    def greedy_solve(random_seed):
        random.seed(random_seed)
        # Start with the safe large set
        S = list(range(91, 181))
        S_set = set(S)
        
        # Try to add small numbers 1..90
        # Order them randomly? Or sorted?
        # Sorted might be more predictable.
        # Let's try adding in increasing order.
        for k in range(1, 91):
            # Check if k can be added
            # We need to check if S U {k} is valid.
            # If not, we need to remove some element.
            # We try to add k.
            new_S = S_set | {k}
            if is_valid(new_S):
                S_set = new_S
            else:
                # Find conflicts introduced by k
                # Conflict means there exists a in new_S such that a pair in conflicts[a] is in new_S.
                # Since S was valid, the only new conflicts must involve k.
                # So either:
                # 1. k is 'a' in a conflict: pair {b,c} in S such that (k-b)(k-c)=k^2.
                # 2. k is 'b' or 'c' in a conflict for some a in S.
                
                # We identify all elements in S that are part of a conflict with k.
                # If we remove any one of them, the conflict is resolved.
                # We pick one to remove.
                
                candidates_to_remove = set()
                
                # Case 1: k is 'a'
                for b, c in conflicts[k]:
                    if b in S_set and c in S_set:
                        candidates_to_remove.add(b)
                        candidates_to_remove.add(c)
                        
                # Case 2: k is 'b' or 'c'
                # For each a in S, check if k is part of a conflict pair for a.
                # We can just check all a in S_set.
                # To speed up, we can precompute for each k, which a's have k in their conflict pairs.
                # But S changes.
                # Just iterate a in S_set.
                for a in S_set:
                    for b, c in conflicts[a]:
                        if b == k and c in S_set:
                            candidates_to_remove.add(c)
                            candidates_to_remove.add(a) # Removing a also resolves
                        elif c == k and b in S_set:
                            candidates_to_remove.add(b)
                            candidates_to_remove.add(a)
                
                if candidates_to_remove:
                    # Choose one to remove
                    # Heuristic: Remove the largest number? 
                    # Large numbers are "safer" to keep? 
                    # Actually, removing a large number might allow keeping more small numbers.
                    # Let's remove the largest.
                    rem = max(candidates_to_remove)
                    S_set.remove(rem)
                    S_set.add(k)
                else:
                    # Should not happen if is_valid failed
                    pass
        
        return len(S_set)

    best_size = 90
    best_set = list(range(91, 181))
    
    # Run a few restarts with different strategies for removal
    # Strategy 1: Remove largest
    # Strategy 2: Remove smallest
    # Strategy 3: Random
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        random.seed(attempt)
        S_set = set(range(91, 181))
        
        # Shuffle the order of small numbers to try?
        # Or just use different removal heuristics.
        
        for k in range(1, 91):
            new_S = S_set | {k}
            if is_valid(new_S):
                S_set = new_S
            else:
                candidates_to_remove = set()
                
                for b, c in conflicts[k]:
                    if b in S_set and c in S_set:
                        candidates_to_remove.add(b)
                        candidates_to_remove.add(c)
                        
                for a in S_set:
                    for b, c in conflicts[a]:
                        if b == k and c in S_set:
                            candidates_to_remove.add(c)
                            candidates_to_remove.add(a)
                        elif c == k and b in S_set:
                            candidates_to_remove.add(b)
                            candidates_to_remove.add(a)
                
                if candidates_to_remove:
                    # Random removal
                    rem = random.choice(list(candidates_to_remove))
                    S_set.remove(rem)
                    S_set.add(k)
                    
        if len(S_set) > best_size:
            best_size = len(S_set)
            best_set = list(S_set)
            
    return best_set
