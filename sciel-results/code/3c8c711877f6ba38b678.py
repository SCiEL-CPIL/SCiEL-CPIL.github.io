import time
import random

def _can_add_bitmask(mask, x):
    """
    Check if x can be added to the set represented by bitmask without creating a forbidden triple.
    Forbidden triple: distinct a, b, c in S_new such that a+b, a+c, b+c in S_new.
    If x is new, any new triple must involve x.
    """
    # Check triples involving x: (a, b, x)
    # Sums: a+b, a+x, b+x must be in S_set U {x}.
    # a+x > x, b+x > x (since a,b >= 1), so a+x, b+x must be in S_set.
    # a+b in S_set or a+b == x.
    
    # Precompute which elements a in S have a+x also in S
    # For each a in S where a+x in S, check all b > a in S where b+x in S
    # Then check if a+b in S or a+b == x
    
    # Build list of elements in S
    elems = []
    for i in range(1, 129):
        if (mask >> i) & 1:
            elems.append(i)
    
    n = len(elems)
    for i in range(n):
        a = elems[i]
        xa = x + a
        if xa > 128:
            continue
        if not ((mask >> xa) & 1):
            continue
        for j in range(i + 1, n):
            b = elems[j]
            xb = x + b
            if xb > 128:
                continue
            if not ((mask >> xb) & 1):
                continue
            ab = a + b
            if ab == x:
                return False
            if ab <= 128 and ((mask >> ab) & 1):
                return False
    
    # Check triples in S that become forbidden because x completes a sum
    # Triple (a, b, c) in S. One sum must be x.
    # If a+b == x, need a+c in S and b+c in S for some c != a,b
    # Find pairs (a, b) in S such that a + b == x
    for i in range(n):
        a = elems[i]
        b = x - a
        if b <= a or b > 128:
            continue
        if not ((mask >> b) & 1):
            continue
        # Check if there exists c in S, c != a, c != b, such that a+c in S and b+c in S
        for c in elems:
            if c == a or c == b:
                continue
            ac = a + c
            bc = b + c
            if ac > 128 or bc > 128:
                continue
            if ((mask >> ac) & 1) and ((mask >> bc) & 1):
                return False
    
    return True

def _is_valid_bitmask(mask):
    """Check if the set represented by bitmask is valid (no forbidden triple)."""
    elems = []
    for i in range(1, 129):
        if (mask >> i) & 1:
            elems.append(i)
    
    n = len(elems)
    for i in range(n):
        a = elems[i]
        for j in range(i + 1, n):
            b = elems[j]
            ab = a + b
            if ab > 128:
                continue
            if not ((mask >> ab) & 1):
                continue
            for k in range(j + 1, n):
                c = elems[k]
                ac = a + c
                bc = b + c
                if ac > 128 or bc > 128:
                    continue
                if ((mask >> ac) & 1) and ((mask >> bc) & 1):
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 82  # Safety margin before 90s limit
    
    def time_left():
        return deadline - time.time()
    
    best_mask = 0
    best_size = 0
    rng = random.Random(42)
    
    def mask_size(mask):
        return bin(mask).count('1')
    
    def try_improve(mask, deadline, max_iterations=100000):
        """Local search: try to add elements, then try swaps."""
        current_mask = mask
        current_size = mask_size(current_mask)
        improved = True
        iterations = 0
        
        while improved and time_left() > 5 and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # 1. Try to add an element
            for x in range(1, 129):
                if (current_mask >> x) & 1:
                    continue
                if _can_add_bitmask(current_mask, x):
                    current_mask |= (1 << x)
                    current_size += 1
                    improved = True
                    break
            
            if improved:
                continue
            
            # 2. Try single swaps: remove r, add a
            elems = [i for i in range(1, 129) if (current_mask >> i) & 1]
            for r in elems:
                if time_left() < 5:
                    break
                without_mask = current_mask & ~(1 << r)
                for a in range(1, 129):
                    if (without_mask >> a) & 1:
                        continue
                    if _can_add_bitmask(without_mask, a):
                        new_mask = without_mask | (1 << a)
                        new_size = mask_size(new_mask)
                        if new_size > current_size:
                            current_mask = new_mask
                            current_size = new_size
                            improved = True
                            break
                if improved:
                    break
        
        return current_mask, current_size
    
    # Strategy 1: Start with upper half, greedily add lower elements
    base_mask = 0
    for i in range(65, 129):
        base_mask |= (1 << i)
    
    # Try multiple orderings for adding lower elements
    lower_elements = list(range(1, 65))
    
    # Natural order
    cur_mask = base_mask
    for x in lower_elements:
        if time_left() < 10:
            break
        if _can_add_bitmask(cur_mask, x):
            cur_mask |= (1 << x)
    
    cur_size = mask_size(cur_mask)
    if cur_size > best_size:
        best_size = cur_size
        best_mask = cur_mask
    
    # Local search from this result
    if time_left() > 20:
        cur_mask, cur_size = try_improve(cur_mask, deadline)
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
    
    # Random orderings
    for trial in range(30):
        if time_left() < 15:
            break
        order = lower_elements[:]
        rng.shuffle(order)
        cur_mask = base_mask
        for x in order:
            if time_left() < 10:
                break
            if _can_add_bitmask(cur_mask, x):
                cur_mask |= (1 << x)
        
        cur_size = mask_size(cur_mask)
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
        
        # Local search
        if time_left() > 20:
            cur_mask, cur_size = try_improve(cur_mask, deadline, max_iterations=50000)
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
    
    # Strategy 2: Try different base ranges
    for start in range(55, 65):
        if time_left() < 15:
            break
        base_mask = 0
        for i in range(start, 129):
            base_mask |= (1 << i)
        
        candidates = list(range(1, start))
        rng.shuffle(candidates)
        
        cur_mask = base_mask
        for x in candidates:
            if time_left() < 10:
                break
            if _can_add_bitmask(cur_mask, x):
                cur_mask |= (1 << x)
        
        cur_size = mask_size(cur_mask)
        if cur_size > best_size:
            best_size = cur_size
            best_mask = cur_mask
        
        if time_left() > 20:
            cur_mask, cur_size = try_improve(cur_mask, deadline, max_iterations=50000)
            if cur_size > best_size:
                best_size = cur_size
                best_mask = cur_mask
    
    # Strategy 3: Try odds/evens patterns
    # Odds only
    cur_mask = 0
    for i in range(1, 129, 2):
        cur_mask |= (1 << i)
    cur_size = mask_size(cur_mask)
    if cur_size > best_size:
        best_size = cur_size
        best_mask = cur_mask
    
    # Evens only
    cur_mask = 0
    for i in range(2, 129, 2):
        cur_mask |= (1 << i)
    cur_size = mask_size(cur_mask)
    if cur_size > best_size:
        best_size = cur_size
        best_mask = cur_mask
    
    # Strategy 4: More aggressive local search with removals
    if time_left() > 30:
        current_mask = best_mask
        current_size = best_size
        improved = True
        while improved and time_left() > 10:
            improved = False
            # Try removing one element and adding two
            elems = [i for i in range(1, 129) if (current_mask >> i) & 1]
            for r in elems:
                if time_left() < 10:
                    break
                without_mask = current_mask & ~(1 << r)
                # Try adding one
                for a in range(1, 129):
                    if (without_mask >> a) & 1:
                        continue
                    if _can_add_bitmask(without_mask, a):
                        new_mask = without_mask | (1 << a)
                        if mask_size(new_mask) > current_size:
                            current_mask = new_mask
                            current_size = mask_size(current_mask)
                            improved = True
                            break
                if improved:
                    break
        
        if current_size > best_size:
            best_size = current_size
            best_mask = current_mask
    
    # Final verification
    result = [i for i in range(1, 129) if (best_mask >> i) & 1]
    
    # Verify validity
    if not _is_valid_bitmask(best_mask):
        # Fallback to upper half
        result = list(range(65, 129))
    
    return [int(x) for x in result]
