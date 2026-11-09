import time
import math
import random

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Try Paley tournaments
    # Optimized verification
    for p in range(5, 91):
        if time.time() > deadline:
            break
        if p % 4 != 3:
            continue
        if not is_prime(p):
            continue
            
        squares = set()
        for x in range(1, p):
            squares.add((x * x) % p)
            
        in_neigh = [0] * p
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                diff = (j - i) % p
                if diff in squares:
                    # i -> j, so i is in-neighbor of j
                    in_neigh[j] |= (1 << i)
                else:
                    # j -> i, so j is in-neighbor of i
                    in_neigh[i] |= (1 << j)
                    
        valid = True
        # Fast check: for all a<b<c<d, check intersection
        # We can speed this up by precomputing pair intersections?
        # But p is small enough.
        for a in range(p - 3):
            ia = in_neigh[a]
            for b in range(a + 1, p - 2):
                ab = ia & in_neigh[b]
                if ab == 0: continue
                for c in range(b + 1, p - 1):
                    abc = ab & in_neigh[c]
                    if abc == 0: continue
                    for d in range(c + 1, p):
                        if (abc & in_neigh[d]) == 0:
                            valid = False
                            break
                    if not valid: break
                if not valid: break
            if not valid: break
                
        if valid:
            orientation = []
            for i in range(p):
                for j in range(i + 1, p):
                    if (j - i) % p in squares:
                        orientation.append(0)
                    else:
                        orientation.append(1)
            return {'n': p, 'orientation': orientation}
            
    # 2. Fallback: Randomized Local Search for n < 83
    # Try to find a valid tournament for n starting from 31 up to 82
    # We use a greedy repair heuristic.
    
    def check_validity(n, in_neigh):
        for a in range(n - 3):
            ia = in_neigh[a]
            for b in range(a + 1, n - 2):
                ab = ia & in_neigh[b]
                if ab == 0: continue
                for c in range(b + 1, n - 1):
                    abc = ab & in_neigh[c]
                    if abc == 0: continue
                    for d in range(c + 1, n):
                        if (abc & in_neigh[d]) == 0:
                            return False
        return True

    def get_orientation(n, in_neigh):
        # in_neigh[i] has bit j set if j -> i
        # We need orientation[i,j] for i<j.
        # 0 means i->j (i is in-neighbor of j, so bit i in in_neigh[j] is 1)
        # 1 means j->i (j is in-neighbor of i, so bit j in in_neigh[i] is 1)
        orientation = []
        for i in range(n):
            for j in range(i + 1, n):
                if (in_neigh[j] >> i) & 1:
                    orientation.append(0)
                else:
                    orientation.append(1)
        return orientation

    # Try n from 31 to 82
    for n in range(31, 83):
        if time.time() > deadline:
            break
            
        # Random initialization
        in_neigh = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    # i -> j
                    in_neigh[j] |= (1 << i)
                else:
                    # j -> i
                    in_neigh[i] |= (1 << j)
                    
        # Local search
        max_iters = 5000
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            # Find a bad 4-set
            bad_set = None
            found = False
            for a in range(n - 3):
                ia = in_neigh[a]
                for b in range(a + 1, n - 2):
                    ab = ia & in_neigh[b]
                    if ab == 0: continue
                    for c in range(b + 1, n - 1):
                        abc = ab & in_neigh[c]
                        if abc == 0: continue
                        for d in range(c + 1, n):
                            if (abc & in_neigh[d]) == 0:
                                bad_set = (a, b, c, d)
                                found = True
                                break
                        if found: break
                    if found: break
                if found: break
                
            if not found:
                # Valid!
                orientation = get_orientation(n, in_neigh)
                return {'n': n, 'orientation': orientation}
                
            # Try to fix by flipping an edge
            a, b, c, d = bad_set
            # Pick a random pair from the set and flip
            # Flipping (x, y) changes in_neigh[x] and in_neigh[y]
            # If we flip x->y to y->x:
            #   in_neigh[x] loses y (if y->x was false, now y->x is true? No.)
            #   Let's define: bit k in in_neigh[i] is 1 if k -> i.
            #   Current state of (x, y):
            #   If bit y in in_neigh[x] is 1, then y -> x.
            #   If we flip, x -> y.
            #   So in_neigh[x] should lose bit y. in_neigh[y] should gain bit x.
            
            pairs = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            random.shuffle(pairs)
            
            improved = False
            for x, y in pairs:
                # Check current direction
                # If bit y in in_neigh[x] is 1, y->x. Flip to x->y.
                # If bit x in in_neigh[y] is 1, x->y. Flip to y->x.
                
                # We just try flipping and see if it reduces "badness" or satisfies this set.
                # Simple heuristic: Flip if it makes the specific bad set valid.
                
                # Calculate if flipping (x,y) helps the set (a,b,c,d)
                # The set is bad if intersection of in_neigh[a,b,c,d] is 0.
                # Flipping an edge changes in_neigh for x and y.
                
                # Try flip x->y (if currently y->x)
                if (in_neigh[x] >> y) & 1:
                    # Currently y -> x. Flip to x -> y.
                    # in_neigh[x] loses y. in_neigh[y] gains x.
                    old_x = in_neigh[x]
                    old_y = in_neigh[y]
                    
                    new_x = old_x & ~(1 << y)
                    new_y = old_y | (1 << x)
                    
                    # Check if new state makes (a,b,c,d) valid
                    # We need to recompute intersection for a,b,c,d with new in_neigh
                    # But only x and y changed.
                    # If x,y are in {a,b,c,d}, it affects the intersection.
                    
                    # To be safe and simple, just flip and check the whole set?
                    # Or just flip and check if THIS set is now valid.
                    
                    # Let's just apply the flip and check the specific set.
                    # We need a helper to get in_neigh for a vertex, considering potential flips.
                    # This is getting complex. 
                    
                    # Simpler: Just flip a random edge in the set and re-check the whole graph?
                    # No, too slow.
                    
                    # Let's just flip and break. The next iteration will check validity.
                    in_neigh[x] = new_x
                    in_neigh[y] = new_y
                    improved = True
                    break
                else:
                    # Currently x -> y. Flip to y -> x.
                    # in_neigh[x] gains y. in_neigh[y] loses x.
                    old_x = in_neigh[x]
                    old_y = in_neigh[y]
                    
                    new_x = old_x | (1 << y)
                    new_y = old_y & ~(1 << x)
                    
                    in_neigh[x] = new_x
                    in_neigh[y] = new_y
                    improved = True
                    break
            if not improved:
                break
                
    # Fallback: Return Paley 83
    p = 83
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                orientation.append(0)
            else:
                orientation.append(1)
    return {'n': p, 'orientation': orientation}
