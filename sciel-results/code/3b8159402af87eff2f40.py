import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s limit
    
    n = 256
    
    # Precompute conflict relationships: for each pair (a, b) with a < b,
    # the three-term APs they form:
    #   midpoint m = (a+b)/2 if integer
    #   extension e = a - (b-a) = 2a - b if in [1,n]
    #   extension e = b + (b-a) = 2b - a if in [1,n]
    
    # For a set S, x cannot be added if:
    # 1. There exist a, b in S such that {a, b, x} is a 3-AP
    #    This means x = (a+b)/2 (midpoint), or x = 2a-b (a is midpoint), or x = 2b-a (b is midpoint)
    #    with distinct elements.
    
    # Fast check using a boolean array
    def can_add_bool(arr, x, S):
        """Check if adding x to S (represented by bool array and set) maintains AP-free property."""
        # x is the middle: need a, b in S with a+b=2x, a!=b, a!=x, b!=x
        # For each a in S, b = 2x - a must be in S, b != a, b != x
        for a in S:
            b = 2 * x - a
            if b != a and b != x and 1 <= b <= n and arr[b]:
                return False
        # x is an end: need a, b in S with a+x=2b or b+x=2a
        # Case: x is first term: x, b, c with x+c=2b => c=2b-x, need b,c in S
        # Case: x is last term: a, b, x with a+x=2b => a=2b-x, need a,b in S
        # Both: for each b in S, a = 2b-x must be in S, a != b, a != x
        for b in S:
            a = 2 * b - x
            if a != b and a != x and 1 <= a <= n and arr[a]:
                return False
        return True
    
    # Greedy initialization: add smallest elements first (maximizes reciprocal sum)
    S = set()
    arr = [False] * (n + 1)
    for x in range(1, n + 1):
        if can_add_bool(arr, x, S):
            S.add(x)
            arr[x] = True
    
    best_S = set(S)
    best_arr = arr[:]
    best_frac = sum(Fraction(1, x) for x in S)
    
    # Local search with multiple phases
    while time.time() < deadline:
        improved = False
        S = set(best_S)
        arr = best_arr[:]
        frac = best_frac
        
        excluded = [x for x in range(1, n + 1) if x not in S]
        excluded.sort(key=lambda v: -1.0 / v)  # prioritize small values
        
        # Phase 1: Try adding each excluded element
        for x in excluded:
            if time.time() > deadline:
                break
            if can_add_bool(arr, x, S):
                new_frac = frac + Fraction(1, x)
                if new_frac > best_frac:
                    S.add(x)
                    arr[x] = True
                    frac = new_frac
                    best_S = set(S)
                    best_arr = arr[:]
                    best_frac = new_frac
                    improved = True
        
        if time.time() > deadline:
            break
        
        # Phase 2: Try single swaps (remove u, add v where 1/v > 1/u)
        if not improved:
            S = set(best_S)
            arr = best_arr[:]
            frac = best_frac
            excluded = [x for x in range(1, n + 1) if x not in S]
            excluded.sort(key=lambda v: -1.0 / v)
            
            S_list = sorted(S, reverse=True)  # Try removing largest first
            
            for u in S_list:
                if time.time() > deadline:
                    break
                # Temporarily remove u
                S_temp = S - {u}
                arr_temp = arr[:]
                arr_temp[u] = False
                frac_temp = frac - Fraction(1, u)
                
                for v in excluded:
                    if v == u:
                        continue
                    if 1.0 / v <= 1.0 / u:
                        break  # No improvement possible
                    if can_add_bool(arr_temp, v, S_temp):
                        new_frac = frac_temp + Fraction(1, v)
                        if new_frac > best_frac:
                            S_temp.add(v)
                            arr_temp[v] = True
                            S = S_temp
                            arr = arr_temp
                            frac = new_frac
                            best_S = set(S)
                            best_arr = arr[:]
                            best_frac = new_frac
                            improved = True
                            break
                
                if improved:
                    break
        
        if time.time() > deadline:
            break
        
        # Phase 3: Try pair swaps (remove u1, u2, add v1, v2)
        if not improved:
            S = set(best_S)
            arr = best_arr[:]
            frac = best_frac
            excluded = [x for x in range(1, n + 1) if x not in S]
            excluded.sort(key=lambda v: -1.0 / v)
            
            S_list = sorted(S, reverse=True)
            
            if len(S_list) >= 2:
                found = False
                # Try removing pairs of large elements
                for i in range(len(S_list)):
                    if time.time() > deadline - 5.0:
                        break
                    u1 = S_list[i]
                    for j in range(i + 1, len(S_list)):
                        u2 = S_list[j]
                        
                        # Try adding pairs of small excluded elements
                        S_temp = S - {u1} - {u2}
                        arr_temp = arr[:]
                        arr_temp[u1] = False
                        arr_temp[u2] = False
                        frac_temp = frac - Fraction(1, u1) - Fraction(1, u2)
                        
                        # Try adding pairs of small excluded elements
                        for k in range(len(excluded)):
                            if time.time() > deadline - 5.0:
                                break
                            v1 = excluded[k]
                            if v1 == u1 or v1 == u2:
                                continue
                            
                            arr_temp2 = arr_temp[:]
                            S_temp2 = set(S_temp)
                            S_temp2.add(v1)
                            arr_temp2[v1] = True
                            
                            if not can_add_bool(arr_temp, v1, S_temp):
                                continue
                            
                            for l in range(k + 1, len(excluded)):
                                v2 = excluded[l]
                                if v2 == u1 or v2 == u2 or v2 == v1:
                                    continue
                                if 1.0 / v2 <= 1.0 / max(u1, u2):
                                    break
                                
                                if can_add_bool(arr_temp2, v2, S_temp2):
                                    new_frac = frac_temp + Fraction(1, v1) + Fraction(1, v2)
                                    if new_frac > best_frac:
                                        S_final = S_temp2 | {v2}
                                        arr_final = arr_temp2[:]
                                        arr_final[v2] = True
                                        S = S_final
                                        arr = arr_final
                                        frac = new_frac
                                        best_S = set(S)
                                        best_arr = arr[:]
                                        best_frac = new_frac
                                        improved = True
                                        found = True
                                        break
                            
                            if found:
                                break
                        
                        if found:
                            break
                    
                    if found:
                        break
        
        if not improved:
            break  # No improvement found, stop
    
    return list(best_S)
