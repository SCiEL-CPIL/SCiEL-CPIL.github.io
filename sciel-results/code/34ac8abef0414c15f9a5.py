def construct():
    n = 256
    
    # The base set [171, 256] is valid because for any a in the set,
    # 3a >= 3*171 = 513 > 2*256 = 512 (max possible sum b+c).
    # Thus a cannot divide b+c since b+c > 2a (as b>a, c>=b) and b+c < 3a.
    base = list(range(171, 257))
    best_set = set(base)
    best_size = len(base)
    
    # Try to improve by adding elements below 171 and removing minimal conflicts
    # For each candidate x < 171, check if it can be added
    # If not, find if removing 1 element allows adding 2, etc.
    
    def is_valid(s):
        """Check if set s is valid"""
        s_list = sorted(s)
        for i, a in enumerate(s_list):
            for j in range(i+1, len(s_list)):
                b = s_list[j]
                for k in range(j, len(s_list)):
                    c = s_list[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def conflicts_of(x, current_set):
        """Find pairs (b,c) in current_set such that x divides b+c"""
        confs = []
        cs = sorted(current_set)
        for i in range(len(cs)):
            for j in range(i, len(cs)):
                b = cs[i]
                c = cs[j]
                if x < b or (x == b and x <= c):
                    # a=x, need x < b <= c
                    if (b + c) % x == 0:
                        confs.append((b, c))
        return confs
    
    # Try adding each element from 1 to 170
    # For each, find minimal removals needed
    candidates = list(range(1, 171))
    
    # Precompute for speed: for each x, which pairs in base set conflict
    # Then try local search
    
    # Greedy: try to add each candidate, removing minimal conflicts
    import time
    start_time = time.time()
    deadline = start_time + 50  # 50 second safety margin
    
    current = set(base)
    
    # Try all candidates, see which ones can be added with minimal removal
    for x in sorted(candidates, reverse=True):
        if time.time() > deadline:
            break
        
        confs = conflicts_of(x, current)
        if not confs:
            # Can add x directly
            current.add(x)
            if len(current) > best_size:
                best_size = len(current)
                best_set = set(current)
        else:
            # Need to remove elements to resolve conflicts
            # Try removing 1 element at a time
            elements_to_remove = set()
            for b, c in confs:
                elements_to_remove.add(b)
                elements_to_remove.add(c)
            
            # Try removing each single element from elements_to_remove
            added = False
            for rem in sorted(elements_to_remove):
                if time.time() > deadline:
                    break
                new_set = current - {rem}
                new_set.add(x)
                # Check if new_set is valid
                if is_valid(new_set):
                    if len(new_set) > best_size:
                        best_size = len(new_set)
                        best_set = set(new_set)
                        current = set(new_set)
                        added = True
                        break
            
            # If single removal didn't work, try pairs (expensive, so limit)
            if not added and len(elements_to_remove) <= 8:
                elems = sorted(elements_to_remove)
                for i in range(len(elems)):
                    if time.time() > deadline:
                        break
                    for j in range(i+1, len(elems)):
                        new_set = current - {elems[i], elems[j]}
                        new_set.add(x)
                        if is_valid(new_set):
                            if len(new_set) > best_size:
                                best_size = len(new_set)
                                best_set = set(new_set)
                                current = set(new_set)
                            break
                    if time.time() > deadline:
                        break
    
    return sorted(best_set)
