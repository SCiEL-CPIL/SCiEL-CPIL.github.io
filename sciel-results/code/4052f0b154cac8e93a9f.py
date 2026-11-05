import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    n = 256

    def is_valid_set(S):
        """Check if S satisfies the property: for all a<b<=c in S, a does NOT divide b+c."""
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                if (b + b) % a == 0:
                    return False
                for k in range(j + 1, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add_fast(S, x):
        """Fast check: can x be added to S?
        Uses the insight that when x is the smallest new element, we only need to check
        if x divides any pair sum in S (for x as 'a'), and for existing elements as 'a',
        check if they divide sums involving x.
        """
        if x in S:
            return False
        S_list = sorted(S)
        m = len(S_list)
        
        # For each existing element a in S:
        # If a < x: check if a divides (b+x) for any b in S with a < b <= x, 
        #           or a divides (x+c) for any c in S with c >= x
        # If a > x: a cannot be 'a' in a triple where x is involved as b or c
        #           (since we need a < b <= c, and if a > x, then a can't be < x)
        #           BUT x could be 'a' in a triple with b,c in S where x < b <= c
        
        # Case 1: x is the 'a' (smallest). Check: for all b,c in S with x < b <= c, x ∤ (b+c)
        # Efficient: for each b in S with b > x, check if any c in S with c >= b has (b+c) % x == 0
        # Use set for O(1) lookup
        S_set = S
        for i in range(m):
            b = S_list[i]
            if b <= x:
                continue
            # Check b+b
            if (b + b) % x == 0:
                return False
            # Check b+c for c > b
            for j in range(i + 1, m):
                c = S_list[j]
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b' or 'c', and some a in S is the smallest
        # For each a in S with a < x:
        #   Check if a divides (b+x) for any b in S with a < b <= x (x is c)
        #   Check if a divides (x+c) for any c in S with c >= x (x is b)
        for i in range(m):
            a = S_list[i]
            if a >= x:
                continue
            # x is 'c': need a < b <= x, check (b+x) % a != 0
            for j in range(m):
                b = S_list[j]
                if a < b <= x:
                    if (b + x) % a == 0:
                        return False
            # x is 'b': need a < x <= c, check (x+c) % a != 0
            for j in range(m):
                c = S_list[j]
                if c >= x:
                    if (x + c) % a == 0:
                        return False
        
        return True

    def can_add_descending(S, x):
        """When iterating descending, all elements in S are > x.
        Only need to check: does x divide (b+c) for any pair b,c in S with b<=c?"""
        s_list = sorted(S)
        m = len(s_list)
        for i in range(m):
            b = s_list[i]
            for j in range(i, m):
                c = s_list[j]
                if (b + c) % x == 0:
                    return False
        return True

    def local_search_improve(S, time_limit):
        """Try to improve S by swapping elements to increase size."""
        S = set(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            S_list = sorted(S)
            for idx in range(len(S_list)):
                if time.time() > time_limit:
                    return S
                a = S_list[idx]
                S.remove(a)
                # Try to add a different element
                for x in range(1, n + 1):
                    if x not in S:
                        if can_add_fast(S, x):
                            S.add(x)
                            if len(S) > len(S_list) or x != a:
                                improved = True
                            break
                if not improved:
                    S.add(a)
        return S

    best_size = 0
    best_S = set()

    # Strategy 1: Descending greedy (fast, from inspiration)
    if time.time() < deadline - 10.0:
        S = set()
        for x in range(n, 0, -1):
            if time.time() > deadline - 5.0:
                break
            if can_add_descending(S, x):
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        # Try local search
        if time.time() < deadline - 10.0:
            S_improved = local_search_improve(S, deadline - 10.0)
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_S = set(S_improved)

    # Strategy 2: Start with upper portion, greedily add
    for restart in range(30):
        if time.time() > deadline - 10.0:
            break
        
        rng = random.Random(restart * 1000 + 42)
        base_start = n - rng.randint(40, 80)
        S = set(range(base_start, n + 1))
        
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_fast(valid_S, x):
                    valid_S.add(x)
            S = valid_S
        
        candidates = [x for x in range(1, base_start) if x not in S]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            if can_add_fast(S, x):
                S.add(x)
        
        candidates2 = [x for x in range(1, n + 1) if x not in S]
        rng.shuffle(candidates2)
        for x in candidates2:
            if time.time() > deadline - 5.0:
                break
            if can_add_fast(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        
        # Local search on best found
        if len(S) >= best_size - 1 and time.time() < deadline - 15.0:
            S_improved = local_search_improve(S, deadline - 15.0)
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_S = set(S_improved)

    # Strategy 3: Start with odd numbers
    if time.time() < deadline - 15.0:
        S = set(range(1, n + 1, 2))
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_fast(valid_S, x):
                    valid_S.add(x)
            S = valid_S
        
        candidates = [x for x in range(2, n + 1, 2) if x not in S]
        rng = random.Random(12345)
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            if can_add_fast(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        
        if time.time() < deadline - 15.0:
            S_improved = local_search_improve(S, deadline - 15.0)
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_S = set(S_improved)

    # Strategy 4: Start with numbers > n/3
    if time.time() < deadline - 15.0:
        S = set(range(86, n + 1))
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_fast(valid_S, x):
                    valid_S.add(x)
            S = valid_S
        
        candidates = [x for x in range(1, 86) if x not in S]
        rng = random.Random(54321)
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            if can_add_fast(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        
        if time.time() < deadline - 15.0:
            S_improved = local_search_improve(S, deadline - 15.0)
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_S = set(S_improved)

    # Strategy 5: Random subset start
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        rng = random.Random(restart * 2000 + 99)
        S = set(rng.sample(range(1, n + 1), rng.randint(20, 40)))
        
        if not is_valid_set(S):
            S_list = sorted(S)
            valid_S = set()
            for x in S_list:
                if can_add_fast(valid_S, x):
                    valid_S.add(x)
            S = valid_S
        
        candidates = [x for x in range(1, n + 1) if x not in S]
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 5.0:
                break
            if can_add_fast(S, x):
                S.add(x)
        
        candidates2 = [x for x in range(1, n + 1) if x not in S]
        rng.shuffle(candidates2)
        for x in candidates2:
            if time.time() > deadline - 5.0:
                break
            if can_add_fast(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
        
        if time.time() < deadline - 15.0:
            S_improved = local_search_improve(S, deadline - 15.0)
            if len(S_improved) > best_size:
                best_size = len(S_improved)
                best_S = set(S_improved)

    if best_size == 0:
        return list(range(129, 257))
    
    if not is_valid_set(best_S):
        S = set(range(129, 257))
        return sorted(S)
    
    return sorted(best_S)
