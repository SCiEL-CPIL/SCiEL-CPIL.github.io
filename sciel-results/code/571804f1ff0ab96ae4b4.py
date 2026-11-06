"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85.0  # 5s safety margin
    n = 256
    
    def get_sums(S):
        """Compute the set of all sums b+c for b,c in S with b<=c."""
        S_list = sorted(S)
        sums = set()
        L = len(S_list)
        for i in range(L):
            b = S_list[i]
            for j in range(i, L):
                c = S_list[j]
                sums.add(b + c)
        return sums

    def can_add_small(x, S, sums_S):
        """
        Check if x can be added to S, assuming x < min(S).
        Condition: x must not divide any sum in sums_S.
        """
        for s in sums_S:
            if s % x == 0:
                return False
        return True

    def is_valid_set(S):
        """Full validity check for arbitrary S."""
        S_list = sorted(S)
        L = len(S_list)
        for i in range(L):
            a = S_list[i]
            for j in range(i + 1, L):
                b = S_list[j]
                for k in range(j, L):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add_general(x, current_set):
        """Check if x can be added to current_set without violating constraints."""
        if x in current_set:
            return False
        
        # Case 1: x is 'a' (smallest) - need b,c from current_set with x < b <= c
        for b in current_set:
            if b <= x:
                continue
            for c in current_set:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b' (middle) - need a < x <= c, a from current_set
        for a in current_set:
            if a >= x:
                continue
            for c in current_set:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is 'c' (largest) - need a < b <= x, a,b from current_set
        for a in current_set:
            for b in current_set:
                if b <= a:
                    continue
                if b > x:
                    continue
                if (b + x) % a == 0:
                    return False
        
        return True

    # Phase 1: Find the best starting interval [L, 256]
    best_start_L = 256
    for L in range(1, n + 1):
        if time.time() > deadline:
            break
        S_interval = set(range(L, n + 1))
        if is_valid_set(S_interval):
            best_start_L = L
            break

    # Start with the largest valid interval
    current_set = set(range(best_start_L, n + 1))
    
    # Phase 2: Greedy Extension
    sums_S = get_sums(current_set)
    
    for x in range(best_start_L - 1, 0, -1):
        if time.time() > deadline:
            break
        if x in current_set:
            continue
        if can_add_small(x, current_set, sums_S):
            current_set.add(x)
            sums_S = get_sums(current_set)

    best_set = set(current_set)
    best_size = len(best_set)

    # Phase 3: Local Search
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        current_set = set(best_set)
        sums_S = get_sums(current_set)
        
        # Step 3a: Try to Add 1
        added_any = False
        for x in range(1, n + 1):
            if x in current_set:
                continue
            if x < min(current_set):
                if can_add_small(x, current_set, sums_S):
                    current_set.add(x)
                    added_any = True
                    break
            else:
                if can_add_general(x, current_set):
                    current_set.add(x)
                    added_any = True
                    break
        
        if added_any:
            best_set = current_set
            best_size = len(best_set)
            improved = True
            continue

        # Step 3b: Try to Add 2, Remove 1
        candidates_x = [x for x in range(1, n + 1) if x not in current_set]
        random.shuffle(candidates_x)
        
        for x in candidates_x:
            if time.time() > deadline - 5:
                break
            
            y_candidates = list(current_set)
            random.shuffle(y_candidates)
            
            for y in y_candidates:
                if time.time() > deadline - 5:
                    break
                
                S_minus_y = current_set - {y}
                
                if x < min(S_minus_y):
                    sums_minus_y = get_sums(S_minus_y)
                    if can_add_small(x, S_minus_y, sums_minus_y):
                        S_temp = S_minus_y | {x}
                        sums_temp = get_sums(S_temp)
                        
                        for z in range(1, n + 1):
                            if z in S_temp:
                                continue
                            if z < min(S_temp):
                                if can_add_small(z, S_temp, sums_temp):
                                    best_set = S_temp | {z}
                                    best_size = len(best_set)
                                    improved = True
                                    break
                            else:
                                if can_add_general(z, S_temp):
                                    best_set = S_temp | {z}
                                    best_size = len(best_set)
                                    improved = True
                                    break
                        
                        if improved:
                            break
                else:
                    if can_add_general(x, S_minus_y):
                        S_temp = S_minus_y | {x}
                        sums_temp = get_sums(S_temp)
                        
                        for z in range(1, n + 1):
                            if z in S_temp:
                                continue
                            if z < min(S_temp):
                                if can_add_small(z, S_temp, sums_temp):
                                    best_set = S_temp | {z}
                                    best_size = len(best_set)
                                    improved = True
                                    break
                            else:
                                if can_add_general(z, S_temp):
                                    best_set = S_temp | {z}
                                    best_size = len(best_set)
                                    improved = True
                                    break
                        
                        if improved:
                            break

    if best_set is None:
        best_set = set(range(192, 257))
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
