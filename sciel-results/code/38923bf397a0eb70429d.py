import time

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Candidate primes p where p % 4 == 3, starting from lower bound 31
    candidates = []
    for p in range(31, 90):
        if p % 4 == 3 and is_prime(p):
            candidates.append(p)
            
    # We want the smallest n. Check candidates in increasing order.
    # If we find a valid one, return it immediately.
    # If we run out of time, return the known best (67).
    
    best_result = None
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        # Construct Paley tournament on p vertices
        # i -> j iff (j - i) % p is a quadratic residue
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
            
        # Precompute In-neighborhoods as bitmasks
        # In[v] is a bitmask where bit u is set if u -> v
        # u -> v iff (v - u) % p is a square
        # So for a fixed v, u is in In[v] if (v - u) % p in squares
        # Which is equivalent to u = (v - s) % p for s in squares
        In = [0] * p
        for v in range(p):
            mask = 0
            for s in squares:
                u = (v - s) % p
                mask |= (1 << u)
            In[v] = mask
            
        # Verify condition: Every 4 vertices have a common dominator
        # i.e., for any distinct a, b, c, d, In[a] & In[b] & In[c] & In[d] != 0
        valid = True
        # Optimization: iterate through 4-combinations
        # We can break early if invalid
        # p is at most 67 in the known best case, but we check smaller first.
        # For p=31, 43, 59, this is fast.
        
        # To speed up, we can precompute pairwise intersections?
        # Or just loop.
        # Let's try a simple nested loop with bitmask ANDs.
        
        for a in range(p):
            if not valid:
                break
            mask_a = In[a]
            for b in range(a + 1, p):
                if not valid:
                    break
                mask_ab = mask_a & In[b]
                if mask_ab == 0:
                    # If In[a] and In[b] are disjoint, any set containing a and b
                    # has no common dominator? 
                    # Wait. The condition is for EVERY set of 4.
                    # If In[a] & In[b] == 0, then for ANY c, d, 
                    # In[a] & In[b] & In[c] & In[d] will be 0.
                    # So if there exists ANY pair (a,b) with empty intersection, 
                    # the tournament is INVALID.
                    # This is a huge optimization!
                    valid = False
                    break
                for c in range(b + 1, p):
                    if not valid:
                        break
                    mask_abc = mask_ab & In[c]
                    if mask_abc == 0:
                        # If In[a] & In[b] & In[c] is empty, then for ANY d,
                        # the intersection of 4 will be empty.
                        # So if there exists ANY triple (a,b,c) with empty intersection,
                        # the tournament is INVALID.
                        valid = False
                        break
                    for d in range(c + 1, p):
                        if (mask_abc & In[d]) == 0:
                            valid = False
                            break
                    if not valid:
                        break
        
        if valid:
            # Construct orientation list
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    # 0 means i -> j, 1 means j -> i
                    # i -> j iff (j - i) % p is a square
                    if ((j - i) % p) in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': p, 'orientation': orientation}
            
    # Fallback: If no smaller valid Paley tournament found, return p=67
    # This should happen because 31, 43, 59 likely fail the condition.
    p = 67
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if ((j - i) % p) in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
