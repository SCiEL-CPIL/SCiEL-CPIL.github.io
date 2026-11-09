import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    N = 128
    # Precompute gcd table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g

    # Precompute bad triples bitmask B[i][j]
    # B[i][j] is a mask of k such that {i, j, k} is a bad triple (all pairwise gcds equal)
    B = [[0] * (N + 1) for _ in range(N + 1)]
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g_ij = gcd_table[i][j]
            mask = 0
            for k in range(1, N + 1):
                if k == i or k == k == j:
                    continue
                g_ik = gcd_table[i][k]
                g_jk = gcd_table[j][k]
                if g_ij == g_ik == g_jk:
                    mask |= (1 << k)
            B[i][j] = mask
            B[j][i] = mask

    def is_valid_addition(x, current_set_mask):
        """
        Check if x can be added to the set represented by current_set_mask.
        x is an integer 1..N.
        current_set_mask is a bitmask of the current set.
        """
        if (current_set_mask >> x) & 1:
            return False
            
        temp = current_set_mask
        while temp:
            lsb = temp & -temp
            y = lsb.bit_length() - 1
            
            if (B[x][y] & current_set_mask) != 0:
                return False
                
            temp ^= lsb
            
        return True

    def get_conflict_count(x, current_set_mask):
        """Count how many elements in current_set conflict with x"""
        count = 0
        temp = current_set_mask
        while temp:
            lsb = temp & -temp
            y = lsb.bit_length() - 1
            if (B[x][y] & current_set_mask) != 0:
                count += 1
            temp ^= lsb
        return count

    def greedy_build(order):
        s = 0
        for x in order:
            if is_valid_addition(x, s):
                s |= (1 << x)
        return s

    def local_search(s, time_limit):
        """
        Try to improve s by various swap strategies.
        Returns improved s.
        """
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            elements_in_s = [i for i in range(1, N + 1) if (s >> i) & 1]
            elements_not_in_s = [i for i in range(1, N + 1) if not ((s >> i) & 1)]
            
            # Strategy 0: Greedy add - try to add any element that fits
            if not improved and time.time() < time_limit:
                # Sort candidates by how many elements they conflict with
                candidates = sorted(elements_not_in_s, key=lambda x: get_conflict_count(x, s))
                for x in candidates:
                    if is_valid_addition(x, s):
                        s |= (1 << x)
                        improved = True
                        break
            
            # Strategy 1: 1-for-2 swap (remove u, add v and w)
            if not improved and time.time() < time_limit:
                random.shuffle(elements_in_s)
                for u in elements_in_s:
                    if time.time() > time_limit:
                        break
                    
                    s_temp = s & ~(1 << u)
                    
                    # Sort candidates by conflict count (fewer conflicts first)
                    candidates = sorted(elements_not_in_s, key=lambda x: get_conflict_count(x, s_temp))
                    
                    for v in candidates:
                        if time.time() > time_limit:
                            break
                        if is_valid_addition(v, s_temp):
                            s_temp_with_v = s_temp | (1 << v)
                            
                            # Try to add another element w
                            for w in elements_not_in_s:
                                if w == v:
                                    continue
                                if is_valid_addition(w, s_temp_with_v):
                                    s = s_temp_with_v | (1 << w)
                                    improved = True
                                    break
                            if improved:
                                break
                                
                    if improved:
                        break
            
            # Strategy 2: 2-for-3 swap (remove u,v, add w,x,y)
            if not improved and time.time() < time_limit:
                random.shuffle(elements_in_s)
                for i in range(len(elements_in_s)):
                    if time.time() > time_limit:
                        break
                    u = elements_in_s[i]
                    for j in range(i + 1, len(elements_in_s)):
                        if time.time() > time_limit:
                            break
                        v = elements_in_s[j]
                        
                        s_temp = s & ~(1 << u) & ~(1 << v)
                        
                        candidates = sorted(elements_not_in_s, key=lambda x: get_conflict_count(x, s_temp))
                        
                        added = []
                        for w in candidates:
                            if len(added) >= 3:
                                break
                            if is_valid_addition(w, s_temp | sum([1 << a for a in added])):
                                added.append(w)
                        
                        if len(added) >= 3:
                            new_s = s_temp
                            for a in added:
                                new_s |= (1 << a)
                            s = new_s
                            improved = True
                            break
                    if improved:
                        break
            
            # Strategy 3: Remove all blockers for a specific candidate
            if not improved and time.time() < time_limit:
                # For each element not in s, find how many elements in s block it
                # Then try removing those blockers to add the candidate
                candidates = sorted(elements_not_in_s, key=lambda x: get_conflict_count(x, s))
                
                for x in candidates[:10]:  # Limit to keep fast
                    if time.time() > time_limit:
                        break
                    
                    # Find which elements in s block x
                    blockers = []
                    temp = s
                    while temp:
                        lsb = temp & -temp
                        y = lsb.bit_length() - 1
                        if (B[x][y] & s) != 0:
                            blockers.append(y)
                        temp ^= lsb
                    
                    if len(blockers) <= 2:
                        # Try removing all blockers and adding x
                        s_temp = s
                        for b in blockers:
                            s_temp &= ~(1 << b)
                        
                        if is_valid_addition(x, s_temp):
                            new_s = s_temp | (1 << x)
                            # Try to add more elements
                            elements_not_in_new = [i for i in range(1, N + 1) if not ((new_s >> i) & 1)]
                            elements_not_in_new = sorted(elements_not_in_new, key=lambda z: get_conflict_count(z, new_s))
                            
                            for z in elements_not_in_new:
                                if is_valid_addition(z, new_s):
                                    new_s |= (1 << z)
                                    break
                            
                            if bin(new_s).count("1") > bin(s).count("1"):
                                s = new_s
                                improved = True
                                break
            
            # Strategy 4: 1-for-1 swap to change structure (maintain size, enable future growth)
            if not improved and time.time() < time_limit:
                random.shuffle(elements_in_s)
                random.shuffle(elements_not_in_s)
                
                for u in elements_in_s[:20]:  # Limit to keep fast
                    if time.time() > time_limit:
                        break
                    
                    s_temp = s & ~(1 << u)
                    
                    # Try to find a replacement that reduces conflicts
                    best_v = -1
                    best_conflicts = get_conflict_count(u, s)
                    
                    for v in elements_not_in_s[:30]:
                        if is_valid_addition(v, s_temp):
                            v_conflicts = get_conflict_count(v, s_temp)
                            if v_conflicts < best_conflicts:
                                best_conflicts = v_conflicts
                                best_v = v
                    
                    if best_v != -1:
                        s = s_temp | (1 << best_v)
                        improved = True
                        break
        
        return s

    best_size = 0
    best_set = 0
    
    random.seed(42)
    
    # Precompute conflict density for ordering
    conflict_density = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            count = bin(B[i][j]).count("1")
            conflict_density[i] += count
            conflict_density[j] += count
    
    # Multiple restarts with different orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: conflict_density[x]),  # Low conflict first
        sorted(range(1, N + 1), key=lambda x: -conflict_density[x]),  # High conflict first
    ]
    
    for _ in range(20):
        shuffled = list(range(1, N + 1))
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > deadline:
            break
            
        s = greedy_build(order)
        s = local_search(s, deadline)
        
        size = bin(s).count("1")
        if size > best_size:
            best_size = size
            best_set = s
    
    # Final polish: try to add more elements to the best set found
    if time.time() < deadline - 5:
        elements_not_in_best = [i for i in range(1, N + 1) if not ((best_set >> i) & 1)]
        elements_not_in_best = sorted(elements_not_in_best, key=lambda x: get_conflict_count(x, best_set))
        
        for x in elements_not_in_best:
            if time.time() > deadline - 5:
                break
            if is_valid_addition(x, best_set):
                best_set |= (1 << x)
                best_size = bin(best_set).count("1")

    # Convert bitmask to list
    result = [i for i in range(1, N + 1) if (best_set >> i) & 1]
    
    if not result:
        return [1]
        
    return result
