import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    # Precompute forbidden (b,c) pairs for each a (where a is the smallest)
    pairs_for_a = {}
    for a in range(1, 181):
        a2 = a * a
        pairs = set()
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                e = a2 // d
                b = a + d
                c = a + e
                if 1 <= b <= 180 and 1 <= c <= 180 and b != c and b != a and c != a:
                    pairs.add((b, c))
        pairs_for_a[a] = pairs
    
    def can_add(s_set, x):
        """Check if x can be added to s_set without creating a forbidden triple."""
        # x as the smallest (a=x): check if any forbidden (b,c) for x are both in s_set
        for (b, c) in pairs_for_a[x]:
            if b in s_set and c in s_set:
                return False
        # x as one of the larger elements (b or c): for each a in s_set with a < x,
        # check if there exists z in s_set such that 1/a = 1/x + 1/z
        for a in s_set:
            if a >= x:
                continue
            # 1/a = 1/x + 1/z => z = a*x/(x-a)
            den = x - a
            num = a * x
            if den > 0 and num % den == 0:
                z = num // den
                if 1 <= z <= 180 and z in s_set and z != x and z != a:
                    return False
        return True
    
    # Start with seed
    best_set = set(range(91, 181))
    best_size = len(best_set)
    
    # Greedy construction with different orders
    def greedy(order, initial):
        s_set = set(initial)
        for x in order:
            if x in s_set:
                continue
            if can_add(s_set, x):
                s_set.add(x)
        return s_set
    
    random.seed(42)
    all_elements = list(range(1, 181))
    
    orders = [
        list(reversed(all_elements)),
        all_elements,
        sorted(all_elements, key=lambda x: len(pairs_for_a[x])),
    ]
    
    for order in orders:
        s = greedy(order, set())
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Local search: try to add more elements
    if time.time() - start_time < DEADLINE:
        s_set = set(best_set)
        improved = True
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            for x in range(1, 181):
                if x in s_set:
                    continue
                if can_add(s_set, x):
                    s_set.add(x)
                    improved = True
                    if len(s_set) > best_size:
                        best_size = len(s_set)
                        best_set = s_set
    
    return list(best_set)
