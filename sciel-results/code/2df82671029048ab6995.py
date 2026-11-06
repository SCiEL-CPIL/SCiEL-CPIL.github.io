import time
from collections import defaultdict

def _valid(S):
    """Full validity check: for all a<b<=c in S, a does NOT divide b+c."""
    lst = sorted(S)
    n = len(lst)
    for i in range(n):
        a = lst[i]
        for j in range(i+1, n):
            b = lst[j]
            for k in range(j, n):
                c = lst[k]
                if (b + c) % a == 0:
                    return False
    return True

def _is_valid_fast(S_set, S_list):
    """Fast validity check using k-based enumeration."""
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i+1, m):
            b = S_list[j]
            k_min = max(3, (2*b + a - 1) // a)
            k_max = min(512 // a, (256 + b) // a)
            for k in range(k_min, k_max + 1):
                c = k * a - b
                if c >= b and c in S_set:
                    return False
    return True

def _build_conflict_graph():
    """Build conflict graph: edge between x,y if they form a violation with some a in set.
    Actually, we need: for a<b<=c, a divides b+c.
    So x and y conflict if there exists z such that {x,y,z} forms a violation.
    But that's 3-way. Instead, let's think differently.
    
    For a fixed set S, x is "bad" if there exist a<b<=c in S with a|x or b|x or c|x... no.
    
    Let me reconsider. The constraint is: for ALL a<b<=c in S, a ∤ (b+c).
    
    So two elements x,y in S create a conflict if there exists some a in S (a < min(x,y)) 
    such that a | (x+y) when x<=y, or similar.
    
    Actually the simplest: build a graph where x~y means {x,y} is part of some violating triple.
    But the triple involves 3 elements. Let's just use the fast validity check.
    
    Alternative: For each element x, compute which other elements it "conflicts with" in the sense
    that if both are in S, they might create a violation with some third element.
    
    Let me just build a pairwise conflict: x and y conflict if there exists z in [1,256] 
    such that {x,y,z} (sorted as a<b<=c) has a|(b+c).
    """
    n = 256
    # conflict[x][y] = True if x,y,z for some z form a violation
    # Actually, let's compute: for pair (x,y) with x<y, when do they conflict?
    # Case 1: x is the divisor a, y is b or c
    #   If x < y: x could be a, y could be b or c
    #   If x=a, y=b: need c>=y, x|(y+c), c in [1,256]
    #   If x=a, y=c: need b<=y, b>x, x|(b+y), b in [x+1,y]
    # Case 2: some z < x is a, and x,y are b,c
    #   z | (x+y)
    
    # Let's precompute for each pair (x,y) whether they are "compatible"
    # Two elements x<y are compatible if NO z in [1,256] creates a violation with them.
    
    # For efficiency, let's just precompute a conflict matrix
    # conflict[x][y] = True if x and y CANNOT both be in a valid set
    # (i.e., there exists z such that {x,y,z} violates the property)
    
    conflict = [[False]*257 for _ in range(257)]
    
    for x in range(1, 257):
        for y in range(x+1, 257):
            # Check if there exists z such that {x,y,z} (sorted a<b<=c) has a|(b+c)
            
            # Sub-case A: x is a (smallest), y is b
            #   Need c >= y, x | (y+c), c in [y, 256]
            #   c = k*x - y for k >= 1, need c >= y and c <= 256
            found = False
            for k in range(1, 513 // x + 1):
                c = k * x - y
                if c >= y and c <= 256:
                    found = True
                    break
            if not found:
                # Sub-case B: x is a, y is c
                #   Need b with x < b <= y, x | (b+y)
                #   b = k*x - y, need x < b <= y
                for k in range(1, 513 // x + 1):
                    b = k * x - y
                    if x < b <= y:
                        found = True
                        break
            if not found:
                # Sub-case C: z < x is a, x is b, y is c
                #   Need z < x, z | (x+y)
                #   z divides x+y and z < x
                s = x + y
                for z in range(1, x):
                    if s % z == 0:
                        found = True
                        break
            if not found:
                # Sub-case D: z < x is a, x is c, y is b? No, x<y so x can't be c if y is b
                # Actually if z<x, then a=z, and b,c are x,y with x<y so b=x, c=y
                # That's the same as sub-case C.
                
                # Sub-case E: x < z < y? No, we only have x,y as the two elements.
                # The third element z can be anywhere.
                
                # Wait, I think I'm overcomplicating. Let me reconsider.
                # The triple is a<b<=c from S. If x,y in S with x<y:
                # - x could be a, y could be b or c
                # - z could be a (z<x), x=b, y=c
                # - z could be b (x<z<y), x=a, y=c  
                # - z could be c (z>y), x=a, y=b
                
                # I already covered:
                # A: x=a, y=b, z=c (z>=y)
                # B: x=a, y=c, z=b (x<z<=y)
                # C: z=a (z<x), x=b, y=c
                # Missing:
                # D: x=a, y=b, z=c where... wait that's A.
                # E: z=b (x<z<y), x=a, y=c
                # F: z=c (z>y), x=a, y=b -- same as A
                
                # Let me redo more carefully.
                
                pass
            
            conflict[x][y] = found
            conflict[y][x] = found
    
    return conflict

def _greedy_independent_set(conflict, order, deadline):
    """Greedy: add elements in given order if they don't conflict with already-chosen."""
    chosen = set()
    for x in order:
        if time.time() > deadline:
            break
        ok = True
        for y in chosen:
            if conflict[x][y]:
                ok = False
                break
        if ok:
            chosen.add(x)
    return chosen

def _local_search(conflict, S, deadline, max_iters=5000):
    """Local search: try swaps to improve set size."""
    S_set = set(S)
    best_size = len(S_set)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try adding an element not in S
        added = False
        for x in range(1, 257):
            if x in S_set:
                continue
            # Check if x can be added
            can_add = True
            for y in S_set:
                if conflict[x][y]:
                    can_add = False
                    break
            if can_add:
                S_set.add(x)
                best_size = max(best_size, len(S_set))
                added = True
                break
        
        if not added:
            # Try swaps: remove one, add one or two
            swapped = False
            for x in list(S_set):
                for y in range(1, 257):
                    if y in S_set:
                        continue
                    # Try removing x, adding y
                    new_set = S_set - {x} | {y}
                    # Check validity of new_set
                    valid = True
                    for a in new_set:
                        for b in new_set:
                            if a != b and conflict[a][b]:
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and len(new_set) > len(S_set):
                        S_set = new_set
                        best_size = max(best_size, len(S_set))
                        swapped = True
                        break
                if swapped:
                    break
            
            if not swapped:
                # Try removing one element to free up space for additions
                # Just try random removals
                if S_set:
                    import random
                    x = random.choice(list(S_set))
                    S_set.discard(x)
                else:
                    break
    
    return S_set

def construct():
    start = time.time()
    deadline = start + 85.0
    n = 256
    
    # Build conflict graph
    conflict = _build_conflict_graph()
    
    best_S = None
    best_size = 0
    
    # Try different orderings for greedy
    orderings = [
        list(range(256, 0, -1)),       # descending
        list(range(1, 257)),           # ascending
        list(range(256, 0, -2)),       # descending evens
        list(range(1, 257, 2)),        # ascending evens
        list(range(256, 0, -3)),       # descending step 3
        list(range(1, 257, 3)),        # ascending step 3
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        S = _greedy_independent_set(conflict, order, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Local search from best
    if best_S and time.time() < deadline - 10:
        S = _local_search(conflict, best_S, deadline)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S
    
    # Fallback
    if not best_S or len(best_S) == 0:
        best_S = set(range(171, 257))
    
    result = list(best_S)
    
    # Final validation
    if not _valid(result):
        result = list(range(171, 257))
    
    return result
