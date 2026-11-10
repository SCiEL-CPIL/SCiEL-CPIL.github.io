import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 55  # Safety margin before 60s
    
    # Precompute all forbidden pairs for each 'a'
    # 1/a = 1/b + 1/c => (b-a)(c-a) = a^2
    # b > a, c > a
    pairs_for_a = {a: [] for a in range(1, 181)}
    # Also precompute: for each pair (b,c) with b<c, which a values make 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c = (b+c)/(bc) => a = bc/(b+c)
    # So a must divide bc and a = bc/(b+c) must be integer
    # Precompute forbidden_pairs: for each (b,c) pair, the set of a that conflict
    forbidden_a_for_bc = {}
    for a in range(1, 181):
        a2 = a * a
        for d in range(1, int(a2**0.5) + 1):
            if a2 % d == 0:
                e = a2 // d
                b = a + d
                c = a + e
                if 1 <= b <= 180 and 1 <= c <= 180 and b != c and b != a and c != a:
                    pairs_for_a[a].append((b, c))
                    # Store in reverse direction too
                    bc_key = (min(b,c), max(b,c))
                    if bc_key not in forbidden_a_for_bc:
                        forbidden_a_for_bc[bc_key] = set()
                    forbidden_a_for_bc[bc_key].add(a)
    
    # Precompute for each x, the list of (a, z) pairs where x is b or c
    # 1/a = 1/x + 1/z => z = ax/(x-a)
    # For each x, precompute all valid (a, z) pairs
    conflicts_for_x = {x: [] for x in range(1, 181)}
    for x in range(1, 181):
        for a in range(1, x):  # a < x
            den = x - a
            num = a * x
            if num % den == 0:
                z = num // den
                if 1 <= z <= 180 and z != x and z != a:
                    conflicts_for_x[x].append((a, z))
    
    # Helper: Check if x can be added to set S without creating a forbidden triple
    def can_add(x, S):
        # Case 1: x is 'a' - check if any (b,c) in pairs_for_a[x] are both in S
        for b, c in pairs_for_a[x]:
            if b in S and c in S:
                return False
        
        # Case 2: x is 'b' or 'c' - check precomputed conflicts
        for a, z in conflicts_for_x[x]:
            if a in S and z in S:
                return False
        
        return True
    
    # Greedy construction
    def greedy(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S
    
    # Local Search (Addition only)
    def local_search(S, max_steps=300):
        S = set(S)
        for _ in range(max_steps):
            if time.time() - start_time > DEADLINE:
                break
            
            # Try to add elements
            added = False
            # Try all candidates from [1, 180] not in S
            candidates = [i for i in range(1, 181) if i not in S]
            if candidates:
                random.shuffle(candidates)
                for y in candidates:
                    if can_add(y, S):
                        S.add(y)
                        added = True
                        break
            
            if not added:
                break
                
        return S
    
    best_set = set()
    best_size = 0
    
    random.seed(42)
    
    # Try different orders
    orders = []
    orders.append(list(range(1, 181)))
    orders.append(list(range(180, 0, -1)))
    
    # Sort by number of pairs (as 'a')
    degree_order = sorted(range(1, 181), key=lambda x: len(pairs_for_a[x]))
    orders.append(degree_order)
    orders.append(list(reversed(degree_order)))
    
    # Add more random orders
    for _ in range(20):
        r_order = list(range(1, 181))
        random.shuffle(r_order)
        orders.append(r_order)
        
    for order in orders:
        if time.time() - start_time > DEADLINE:
            break
            
        S = greedy(order)
        
        # Local search to try to expand
        S_expanded = local_search(S, max_steps=300)
        
        if len(S_expanded) > best_size:
            best_size = len(S_expanded)
            best_set = S_expanded
            
    # Final check and return
    return list(best_set)
