import time
import random

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for safety
    
    N = 512
    # 1-indexed for problem statement, 0-indexed internally for arrays? 
    # Let's use 0..N-1 internally, map to 1..N at the end.
    
    # Precompute 4-APs
    # An AP is (a, b, c, d) with b=a+d, c=a+2d, d=a+3d
    # Indices 0..N-1
    aps_by_element = [[] for _ in range(N)]
    
    # Generate all 4-APs
    # d can range from 1 to (N-1)//3
    max_d = (N - 1) // 3
    for d in range(1, max_d + 1):
        # a can range from 0 to N - 3*d - 1
        for a in range(N - 3 * d):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            ap = (a, b, c, e)
            aps_by_element[a].append(ap)
            aps_by_element[b].append(ap)
            aps_by_element[c].append(ap)
            aps_by_element[e].append(ap)
            
    # Helper to check if x can be added to in_set (bytearray of 0/1)
    def can_add(x, in_set):
        for ap in aps_by_element[x]:
            # Check if the other 3 are in in_set
            # ap contains x. 
            # We can just check all 4, but we know x is not in in_set yet (we are adding it)
            # So we check if the other 3 are 1.
            i1, i2, i3, i4 = ap
            # If x is one of them, the others must be 1
            # This is slightly inefficient but N is small.
            # Optimization: store the other 3 indices separately?
            # For now, this is fine.
            if i1 == x:
                if in_set[i2] and in_set[i3] and in_set[i4]:
                    return False
            elif i2 == x:
                if in_set[i1] and in_set[i3] and in_set[i4]:
                    return False
            elif i3 == x:
                if in_set[i1] and in_set[i2] and in_set[i4]:
                    return False
            else: # i4 == x
                if in_set[i1] and in_set[i2] and in_set[i3]:
                    return False
        return True

    def greedy_init():
        in_set = bytearray(N)
        elements = list(range(N))
        random.shuffle(elements)
        count = 0
        for x in elements:
            if can_add(x, in_set):
                in_set[x] = 1
                count += 1
        return in_set, count

    def local_search(in_set, count):
        # Find missing elements
        missing = [i for i in range(N) if not in_set[i]]
        
        # Try to add elements until no more can be added
        # We shuffle missing to avoid bias
        random.shuffle(missing)
        
        changed = True
        while changed:
            changed = False
            new_missing = []
            for x in missing:
                if can_add(x, in_set):
                    in_set[x] = 1
                    count += 1
                    changed = True
                else:
                    new_missing.append(x)
            missing = new_missing
            if not missing:
                break
            # If we added some, we might be able to add others that were previously blocked?
            # Actually, adding elements only creates more constraints, so it never makes a previously invalid element valid.
            # So we only need to scan once.
            break 

        return in_set, count

    def perturb(in_set, count, k):
        # Remove k random elements
        current_elements = [i for i in range(N) if in_set[i]]
        random.shuffle(current_elements)
        for i in range(min(k, len(current_elements))):
            in_set[current_elements[i]] = 0
            count -= 1
        return in_set, count

    best_count = 0
    best_in_set = bytearray(N)
    
    # Initial random greedy
    in_set, count = greedy_init()
    in_set, count = local_search(in_set, count)
    if count > best_count:
        best_count = count
        best_in_set = in_set.copy()
        
    while time.time() - start_time < time_limit:
        # Perturb the current best or a random state?
        # Strategy: Start from best, perturb, local search.
        # Also try fresh random starts occasionally.
        
        if random.random() < 0.5 and best_count > 0:
            # Perturb best
            in_set = best_in_set.copy()
            count = best_count
            in_set, count = perturb(in_set, count, 5)
        else:
            # Fresh start
            in_set, count = greedy_init()
        
        in_set, count = local_search(in_set, count)
        
        if count > best_count:
            best_count = count
            best_in_set = in_set.copy()
            
    # Convert to 1-indexed list
    result = [i + 1 for i in range(N) if best_in_set[i]]
    
    # Ensure non-empty (should always be true)
    if not result:
        return [1]
        
    return result

# EVOLVE-BLOCK-END
