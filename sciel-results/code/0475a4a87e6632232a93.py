import time
import random

def construct():
    """
    Construct a set of distinct integers in 1..256 such that subset sums of
    different cardinalities never collide.
    
    Strategy:
    - Use bitset representation for subset sums per cardinality
    - Greedy initialization with multiple orderings
    - Simulated annealing with swap moves to escape local optima
    - Fast validity check using bitset intersection
    """
    deadline = time.time() + 55.0
    
    def bitset_add(T, x, current_size):
        """Add x to bitset representation T. current_size is len before adding."""
        new_T = T[:]
        for k in range(current_size, -1, -1):
            if new_T[k]:
                new_T[k + 1] |= (new_T[k] << x)
        return new_T
    
    def is_valid_bitset(T, x, current_size):
        """Check if adding x to set of size current_size maintains disjointness."""
        for k in range(1, current_size + 2):
            if k - 1 > 60 or not T[k - 1]:
                continue
            new_sums = T[k - 1] << x
            for j in range(0, current_size + 1):
                if j == k:
                    continue
                if j <= 60 and T[j] and (new_sums & T[j]):
                    return False
        return True
    
    def rebuild_T(s):
        """Rebuild T from scratch given a list s."""
        T = [0] * 61
        T[0] = 1
        size = 0
        for x in s:
            T = bitset_add(T, x, size)
            size += 1
        return T
    
    def greedy_construct(order):
        """Build a set using the given order of candidates."""
        T = [0] * 61
        T[0] = 1
        current_set = []
        used = set()
        
        for x in order:
            if time.time() > deadline:
                break
            if len(current_set) >= 60:
                break
            if x in used:
                continue
            if is_valid_bitset(T, x, len(current_set)):
                T = bitset_add(T, x, len(current_set))
                current_set.append(x)
                used.add(x)
        
        return current_set, T, used
    
    # Phase 1: Try several different orderings for greedy initialization
    best_set = []
    best_T = None
    
    order1 = list(range(1, 257))
    order2 = list(range(256, 0, -1))
    order3 = list(range(1, 257))
    random.shuffle(order3)
    smalls = list(range(1, 129))
    larges = list(range(256, 128, -1))
    order4 = []
    si, li = 0, 0
    while si < len(smalls) or li < len(larges):
        if si < len(smalls):
            order4.append(smalls[si])
            si += 1
        if li < len(larges):
            order4.append(larges[li])
            li += 1
    upper = list(range(129, 257))
    lower = list(range(1, 129))
    random.shuffle(upper)
    random.shuffle(lower)
    order6 = upper + lower
    
    for order in [order1, order2, order3, order4, order6]:
        if time.time() > deadline:
            break
        s, T, used = greedy_construct(order)
        if len(s) > len(best_set):
            best_set = s
            best_T = T
    
    # Phase 2: Simulated annealing with swap moves
    if len(best_set) < 60 and best_T is not None:
        current_set = best_set[:]
        T = best_T[:]
        used = set(current_set)
        excluded = [x for x in range(1, 257) if x not in used]
        
        T_start = 10.0
        T_end = 0.1
        n_iter = 3000
        best_sa_set = current_set[:]
        best_sa_T = T[:]
        
        for it in range(n_iter):
            if time.time() > deadline - 5:
                break
            
            frac = it / max(n_iter, 1)
            temp = T_start * (T_end / T_start) ** frac
            
            if not excluded:
                break
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Try pure addition
                x = random.choice(excluded)
                if is_valid_bitset(T, x, len(current_set)):
                    T = bitset_add(T, x, len(current_set))
                    current_set.append(x)
                    used.add(x)
                    excluded = [e for e in excluded if e != x]
                    
                    if len(current_set) > len(best_sa_set):
                        best_sa_set = current_set[:]
                        best_sa_T = T[:]
                continue
            
            elif move_type < 0.8:
                # Swap: remove one, add one
                if not current_set or not excluded:
                    continue
                idx = random.randint(0, len(current_set) - 1)
                remove_x = current_set[idx]
                add_x = random.choice(excluded)
                
                if add_x == remove_x:
                    continue
                
                new_set = [x for i, x in enumerate(current_set) if i != idx]
                new_T = rebuild_T(new_set)
                
                if is_valid_bitset(new_T, add_x, len(new_set)):
                    new_T2 = bitset_add(new_T, add_x, len(new_set))
                    final_set = new_set + [add_x]
                    
                    old_energy = -len(current_set)
                    new_energy = -len(final_set)
                    delta_e = new_energy - old_energy
                    
                    if delta_e <= 0 or random.random() < pow(2.718281828, -delta_e / max(temp, 0.01)):
                        current_set = final_set
                        T = new_T2
                        used = set(final_set)
                        excluded = [e for e in excluded if e != add_x]
                        
                        if len(current_set) > len(best_sa_set):
                            best_sa_set = current_set[:]
                            best_sa_T = T[:]
                continue
            
            else:
                x = random.choice(excluded)
                if is_valid_bitset(T, x, len(current_set)):
                    T = bitset_add(T, x, len(current_set))
                    current_set.append(x)
                    used.add(x)
                    excluded = [e for e in excluded if e != x]
                    
                    if len(current_set) > len(best_sa_set):
                        best_sa_set = current_set[:]
                        best_sa_T = T[:]
                continue
        
        if len(best_sa_set) > len(best_set):
            best_set = best_sa_set
            best_T = best_sa_T
    
    # Phase 3: Final greedy pass
    if len(best_set) < 60 and best_T is not None:
        current_set = best_set[:]
        T = best_T[:]
        used = set(current_set)
        
        for x in range(1, 257):
            if time.time() > deadline - 2:
                break
            if x in used:
                continue
            if is_valid_bitset(T, x, len(current_set)):
                T = bitset_add(T, x, len(current_set))
                current_set.append(x)
                used.add(x)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            best_T = T
    
    if not best_set:
        best_set = [1]
    
    return [int(x) for x in best_set]
