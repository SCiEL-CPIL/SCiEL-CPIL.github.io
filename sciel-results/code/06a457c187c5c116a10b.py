import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = 180
    
    # Precompute triples (a,b,c) such that 1/a = 1/b + 1/c
    # a < b < c
    # (b-a)(c-a) = a^2
    triples = []
    for a in range(1, n + 1):
        a2 = a * a
        # Find divisors d of a2
        # d and e such that d*e = a2
        # b = a + d, c = a + e
        # We need b < c => d < e
        # b, c <= n => a + d <= n, a + e <= n
        # So d <= n - a, e <= n - a
        d = 1
        while d * d <= a2:
            if a2 % d == 0:
                e = a2 // d
                if d < e:
                    b = a + d
                    c = a + e
                    if c <= n:
                        triples.append((a, b, c))
            d += 1
            
    # Build conflict graph or list of pairs for each vertex
    # For each a, list of (b,c)
    # For each b, list of (a,c)
    # For each c, list of (a,b)
    # To check if adding x creates a violation, we need to know if there are
    # any triples (x, y, z) or (y, x, z) or (y, z, x) in the current set.
    # It's easier to just check all triples involving x.
    
    # Incidence list: for each index i (0 to 179, value i+1),
    # store list of tuples (j, k) such that (i+1, j+1, k+1) is a triple.
    # Note: if i is the 'a', pairs are (b-1, c-1).
    # If i is the 'b', pairs are (a-1, c-1).
    # If i is the 'c', pairs are (a-1, b-1).
    
    conflicts = [[] for _ in range(n)]
    for (a, b, c) in triples:
        # a is smallest
        conflicts[a - 1].append((b - 1, c - 1))
        # b is middle
        conflicts[b - 1].append((a - 1, c - 1))
        # c is largest
        conflicts[c - 1].append((a - 1, b - 1))
        
    def is_valid(S):
        # S is a boolean list or set of indices
        # Check all triples
        for (a, b, c) in triples:
            if S[a - 1] and S[b - 1] and S[c - 1]:
                return False
        return True

    best_set = None
    best_size = 0
    
    def try_add(S, a):
        """Try to add element a (0-indexed) to set S. Returns True if successful."""
        for (u, v) in conflicts[a]:
            if S[u] and S[v]:
                return False
        S[a] = True
        return True

    def local_search(S, max_iters=5000):
        """Improve set S using greedy local search: try adding any element, then remove conflicting ones."""
        cur_size = sum(S)
        improved = True
        while improved and cur_size > 0:
            improved = False
            # Try to add each missing element in random order
            missing = [i for i in range(n) if not S[i]]
            random.shuffle(missing)
            for a in missing:
                if try_add(S, a):
                    cur_size += 1
                    improved = True
            # Try to remove and re-add elements to escape local optima
            present = [i for i in range(n) if S[i]]
            random.shuffle(present)
            for a in present:
                # Try removing a and see if we can add more
                S[a] = False
                cur_size -= 1
                # Try to add all missing elements
                missing2 = [i for i in range(n) if not S[i]]
                random.shuffle(missing2)
                added = 0
                for b in missing2:
                    if try_add(S, b):
                        added += 1
                # If we added more than we removed, keep the change
                if added > 1:  # net gain
                    cur_size += added
                    improved = True
                else:
                    # Restore a
                    try_add(S, a)
                    cur_size += 1
        return cur_size

    # Strategy 1: Suffix + Greedy Add
    # Try different starting K
    for K in range(1, n + 1):
        if time.time() - t0 > DEADLINE - 15:
            break
            
        # Initialize S with {K, ..., n}
        S = [False] * n
        for i in range(K - 1, n):
            S[i] = True
            
        # Check if this suffix is valid on its own
        valid_suffix = True
        for (a, b, c) in triples:
            if a >= K and b >= K and c >= K:
                valid_suffix = False
                break
        
        if not valid_suffix:
            continue
            
        cur_size = n - K + 1
        if cur_size > best_size:
            best_size = cur_size
            best_set = S[:]
            
        # Greedy add smaller numbers
        # Iterate a from K-1 down to 1
        for a in range(K - 1, 0, -1):
            if S[a - 1]:
                continue
            # Check if adding a creates a violation
            can_add = True
            for (u, v) in conflicts[a - 1]:
                if S[u] and S[v]:
                    can_add = False
                    break
                    
            if can_add:
                S[a - 1] = True
                cur_size += 1
                if cur_size > best_size:
                    best_size = cur_size
                    best_set = S[:]
                    
    # Strategy 2: Local search from best set found
    if best_set is not None and time.time() - t0 < DEADLINE - 5:
        S2 = best_set[:]
        cur_size = local_search(S2)
        if cur_size > best_size:
            best_size = cur_size
            best_set = S2[:]

    # Strategy 3: Multiple random restarts with local search
    if time.time() - t0 < DEADLINE - 10:
        for restart in range(20):
            if time.time() - t0 > DEADLINE - 5:
                break
            
            # Random starting point: pick a random threshold
            K = random.randint(30, 120)
            S = [False] * n
            for i in range(K - 1, n):
                S[i] = True
                
            # Check validity
            valid = True
            for (a, b, c) in triples:
                if a >= K and b >= K and c >= K:
                    valid = False
                    break
            
            if not valid:
                continue
            
            cur_size = n - K + 1
            
            # Greedy add
            for a in range(K - 1, 0, -1):
                if S[a - 1]:
                    continue
                can_add = True
                for (u, v) in conflicts[a - 1]:
                    if S[u] and S[v]:
                        can_add = False
                        break
                if can_add:
                    S[a - 1] = True
                    cur_size += 1
            
            # Local search
            cur_size = local_search(S)
            
            if cur_size > best_size:
                best_size = cur_size
                best_set = S[:]

    if best_set is None:
        best_set = [False] * n
        best_set[n-1] = True # [180]
        best_size = 1

    # Convert to list of values
    out = [i + 1 for i in range(n) if best_set[i]]
    
    return out
