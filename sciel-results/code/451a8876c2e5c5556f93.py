import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    def is_valid_set(S):
        """Check if set S satisfies the constraint: for all a<b<=c in S, a does not divide b+c"""
        elems = sorted(S)
        n = len(elems)
        for i in range(n):
            a = elems[i]
            for j in range(i + 1, n):
                b = elems[j]
                for k in range(j, n):
                    c = elems[k]
                    if (b + c) % a == 0:
                        return False
        return True
    
    def can_add_to_set(x, S):
        """Check if x can be added to set S without violating constraints"""
        elems = sorted(S)
        n = len(elems)
        
        # x is the smallest element: check if x divides any pair sum
        for i in range(n):
            for j in range(i, n):
                if (elems[i] + elems[j]) % x == 0:
                    return False
        
        # x is the largest or middle element
        # For pairs where x is b or c, we need: for all a in S with a < x,
        # a does not divide (x + y) where y >= x and y in S, or y = x (if x in S, but x not in S yet)
        # Actually since x is new, we check:
        # Case 1: a < x <= x (b=c=x): a must not divide 2x
        # Case 2: a < x <= y where y in S and y >= x: a must not divide x + y
        
        for a in elems:
            if a < x:
                # Check a divides 2*x (when b=c=x)
                if (2 * x) % a == 0:
                    return False
                # Check a divides x + y for y in S with y >= x
                for y in elems:
                    if y >= x:
                        if (x + y) % a == 0:
                            return False
        
        return True
    
    def greedy_construct(order):
        """Greedy construction using given order of elements"""
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add_to_set(x, S):
                S.add(x)
        return S
    
    def local_search_add(S, candidates):
        """Try to add elements from candidates to S"""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for c in candidates:
                if time.time() > deadline - 5:
                    break
                if c not in S and can_add_to_set(c, S):
                    S.add(c)
                    improved = True
        return S
    
    def local_search_replace(S, candidates):
        """Try to replace elements in S with multiple from candidates"""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for s in list(S):
                if time.time() > deadline - 5:
                    break
                S_temp = S - {s}
                added = 0
                new_elems = []
                for c in sorted(candidates, reverse=True):
                    if time.time() > deadline - 5:
                        break
                    if c in S_temp or c == s:
                        continue
                    if can_add_to_set(c, S_temp):
                        S_temp.add(c)
                        added += 1
                        new_elems.append(c)
                if added >= 2:
                    S = S_temp
                    improved = True
                    break
        return S
    
    def local_search_remove(S):
        """Try removing elements to allow more additions"""
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for s in list(S):
                if time.time() > deadline - 5:
                    break
                S_temp = S - {s}
                # Try to add as many as possible
                added = 0
                for c in range(1, N + 1):
                    if time.time() > deadline - 5:
                        break
                    if c in S_temp or c == s:
                        continue
                    if can_add_to_set(c, S_temp):
                        S_temp.add(c)
                        added += 1
                if len(S_temp) > len(S):
                    S = S_temp
                    improved = True
                    break
        return S
    
    # Multiple restarts with different orders
    best_S = set()
    
    # Restart 1: Descending order (original)
    if time.time() < deadline:
        S1 = greedy_construct(range(N, 0, -1))
        S1 = local_search_add(S1, set(range(1, N + 1)) - S1)
        S1 = local_search_replace(S1, set(range(1, N + 1)) - S1)
        S1 = local_search_remove(S1)
        if len(S1) > len(best_S):
            best_S = S1
    
    # Restart 2: Ascending order
    if time.time() < deadline - 30:
        S2 = greedy_construct(range(1, N + 1))
        S2 = local_search_add(S2, set(range(1, N + 1)) - S2)
        S2 = local_search_replace(S2, set(range(1, N + 1)) - S2)
        S2 = local_search_remove(S2)
        if len(S2) > len(best_S):
            best_S = S2
    
    # Restart 3-8: Random orders
    random.seed(42)
    for restart in range(6):
        if time.time() > deadline - 30:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S_rand = greedy_construct(order)
        S_rand = local_search_add(S_rand, set(range(1, N + 1)) - S_rand)
        S_rand = local_search_replace(S_rand, set(range(1, N + 1)) - S_rand)
        S_rand = local_search_remove(S_rand)
        if len(S_rand) > len(best_S):
            best_S = S_rand
    
    # Final polish: try to add any remaining elements
    if time.time() < deadline - 5:
        best_S = local_search_add(best_S, set(range(1, N + 1)) - best_S)
        best_S = local_search_replace(best_S, set(range(1, N + 1)) - best_S)
    
    # Verify the result is valid
    if not is_valid_set(best_S):
        # Fallback: use a simple valid set
        best_S = set(range(N // 2 + 1, N + 1))
    
    return list(best_S)
