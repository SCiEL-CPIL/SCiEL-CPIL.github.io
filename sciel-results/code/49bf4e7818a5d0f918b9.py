import time
import random

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 85  # safety margin for 90s limit

    # Start with the "safe" upper set
    # Condition 3*min(S) > 2*max(S) guarantees validity for S subset [K, N]
    # 3*K > 2*N => K > 2*256/3 = 170.67 => K = 171
    K = 171
    S = set(range(K, n + 1))

    def can_add(x, S_set):
        """Check if x can be added to S_set."""
        if x in S_set:
            return False
        
        lst = sorted(S_set)
        m = len(lst)
        
        # 1. x is smallest: x < b <= c
        # Check x !| (b+c) for all b, c in S_set with x < b <= c
        # b, c are elements of S_set. If b=c, we use the same element twice.
        for i in range(m):
            b = lst[i]
            if b <= x:
                continue
            # b > x
            # Check b+c for c >= b
            # c can be b (reuse) or c > b
            # 1. c = b. Check x !| 2b.
            if (2 * b) % x == 0:
                return False
            # 2. c > b.
            for j in range(i + 1, m):
                c = lst[j]
                if (b + c) % x == 0:
                    return False
        
        # 2. x is middle: a < x <= c
        # Check a !| (x+c) for all a, c in S_set with a < x <= c
        # Since x is not in S_set, c cannot be x. So c > x.
        for i in range(m):
            a = lst[i]
            if a >= x:
                break
            # a < x
            # c must be >= x. Since x not in S_set, c > x.
            for j in range(m):
                c = lst[j]
                if c <= x:
                    continue
                # c > x
                if (x + c) % a == 0:
                    return False
        
        # 3. x is largest: a < b < x
        # Check a !| (b+x) for all a, b in S_set with a < b < x
        for i in range(m):
            a = lst[i]
            if a >= x:
                break
            # a < x
            for j in range(i + 1, m):
                b = lst[j]
                if b >= x:
                    break
                # a < b < x
                if (b + x) % a == 0:
                    return False

        return True

    def is_valid(S_set):
        """Check if the entire set S_set is valid."""
        lst = sorted(S_set)
        m = len(lst)
        for i in range(m):
            a = lst[i]
            for j in range(i + 1, m):
                b = lst[j]
                # b > a
                # Check a !| (b+c) for c >= b
                # c can be b (reuse)
                if (2 * b) % a == 0:
                    return False
                for k in range(j + 1, m):
                    c = lst[k]
                    if (b + c) % a == 0:
                        return False
        return True

    # Greedy add smaller numbers
    for x in range(K - 1, 0, -1):
        if time.time() > deadline - 15:
            break
        if can_add(x, S):
            S.add(x)

    best_S = set(S)
    best_size = len(S)

    # Local Search: Simulated Annealing style or Hill Climbing with restarts
    # We will try to improve by removing one element and adding another, or just adding.
    
    # Phase 1: Hill Climbing - try to add any element
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Shuffle candidates to avoid local minima bias
        candidates = [x for x in range(1, n + 1) if x not in S]
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline - 5:
                break
            if can_add(x, S):
                S.add(x)
                improved = True
                best_S = set(S)
                best_size = len(S)
                break

    # Phase 2: Local Search - Swap/Remove-Add
    # Try removing each element u and then greedily adding elements
    # This can escape local minima where a single element blocks many others.
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            current_size = len(S)
            
            # Shuffle S to vary the order of removal attempts
            s_list = list(S)
            random.shuffle(s_list)
            
            for u in s_list:
                if time.time() > deadline - 5:
                    break
                
                S_try = S - {u}
                
                # Greedy add to S_try
                # Try adding in increasing order? Or decreasing?
                # Decreasing (larger first) is often safer for divisibility constraints
                # because smaller numbers have more multiples.
                # Let's try decreasing first, then increasing if that fails.
                # Actually, the initial greedy was decreasing. Let's stick with that or try both.
                
                # Try adding candidates in decreasing order
                candidates = [x for x in range(n, 0, -1) if x not in S_try]
                for x in candidates:
                    if can_add(x, S_try):
                        S_try.add(x)
                
                if len(S_try) > current_size:
                    S = S_try
                    improved = True
                    best_S = set(S)
                    best_size = len(S)
                    break

    # Phase 3: If we have time, try a few random restarts from the best set
    # Remove 2 random elements and greedily refill
    if time.time() < deadline - 10:
        restarts = 0
        while time.time() < deadline - 5 and restarts < 20:
            restarts += 1
            S_try = set(best_S)
            
            # Remove 2 random elements
            if len(S_try) > 2:
                elems = list(S_try)
                random.shuffle(elems)
                S_try.remove(elems[0])
                S_try.remove(elems[1])
            
            # Greedy refill
            candidates = [x for x in range(n, 0, -1) if x not in S_try]
            for x in candidates:
                if time.time() > deadline - 5:
                    break
                if can_add(x, S_try):
                    S_try.add(x)
            
            if len(S_try) > best_size:
                best_S = S_try
                best_size = len(S_try)

    # Final check and return
    # Ensure the returned set is valid (paranoia check, though can_add should guarantee it)
    if not is_valid(best_S):
        # Fallback to a known valid set if something went wrong
        best_S = set(range(K, n + 1))

    return list(best_S)
