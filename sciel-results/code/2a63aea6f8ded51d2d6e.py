import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = 85.0
    n = 180
    
    # Precompute triples (a,b,c) such that 1/a = 1/b + 1/c
    # a < b < c
    # (b-a)(c-a) = a^2
    triples = []
    for a in range(1, n + 1):
        a2 = a * a
        d = 1
        while d * d <= a2:
            if a2 % d == 0:
                e = a2 // d
                if d < e:
                    b = a + d
                    c = a + e
                    if c <= n:
                        triples.append((a, b, c))
            d += 1
            
    # Build conflict lists
    conflicts = [[] for _ in range(n)]
    for (a, b, c) in triples:
        conflicts[a - 1].append((b - 1, c - 1))
        conflicts[b - 1].append((a - 1, c - 1))
        conflicts[c - 1].append((a - 1, b - 1))
        
    best_set = None
    best_size = 0
    
    def try_add(S, x):
        """Try to add index x to set S. Returns True if successful."""
        for (u, v) in conflicts[x]:
            if S[u] and S[v]:
                return False
        S[x] = True
        return True
    
    def greedy_from_suffix(K, order=None):
        """Build a valid set starting from suffix {K,...,n}, adding smaller elements in given order."""
        S = [False] * n
        for i in range(K - 1, n):
            S[i] = True
            
        # Check suffix validity
        for (a, b, c) in triples:
            if a >= K and b >= K and c >= K:
                return None
        
        cur_size = n - K + 1
        
        # Get candidate indices to add
        if order is None:
            candidates = list(range(K - 1, 0, -1))  # descending
        else:
            candidates = order[:]
        
        for x in candidates:
            if S[x]:
                continue
            if try_add(S, x):
                cur_size += 1
                
        return S, cur_size
    
    # Precompute various orderings for each K
    def get_orders(K):
        """Generate multiple orderings for candidates 1..K-1."""
        orders = []
        # Descending
        orders.append(list(range(K - 1, 0, -1)))
        # Ascending
        orders.append(list(range(1, K)))
        # Priority: fewer conflicts first
        orders.append(sorted(range(1, K), key=lambda x: len(conflicts[x])))
        # Reverse priority: more conflicts first
        orders.append(sorted(range(1, K), key=lambda x: -len(conflicts[x])))
        # A few random orders
        for _ in range(2):
            o = list(range(1, K))
            random.shuffle(o)
            orders.append(o)
        return orders
    
    # Strategy 1: Multi-order greedy from various suffixes
    for K in range(1, n + 1):
        if time.time() - t0 > DEADLINE - 15:
            break
        
        for order in get_orders(K):
            if time.time() - t0 > DEADLINE - 15:
                break
            result = greedy_from_suffix(K, order=order)
            if result is not None:
                S, cur_size = result
                if cur_size > best_size:
                    best_size = cur_size
                    best_set = S[:]
    
    # Strategy 2: Local search on best set - remove one, try to add two
    if best_set is not None and time.time() - t0 < DEADLINE - 5:
        S = best_set[:]
        for _ in range(20):  # Limited passes
            if time.time() - t0 > DEADLINE - 5:
                break
            improved = False
            for i in range(n):
                if not S[i]:
                    continue
                S[i] = False
                added = []
                for j in range(n):
                    if S[j] or j == i:
                        continue
                    if try_add(S, j):
                        added.append(j)
                        if len(added) >= 2:
                            break
                
                if len(added) >= 2:
                    improved = True
                else:
                    for j in added:
                        S[j] = False
                    S[i] = True
            
            if not improved:
                break
        
        cur_size = sum(S)
        if cur_size > best_size:
            best_size = cur_size
            best_set = S[:]
    
    # Strategy 3: Random restarts
    while time.time() - t0 < DEADLINE - 3:
        S = [False] * n
        for i in range(n // 2, n):
            if random.random() < 0.7:
                S[i] = True
        
        candidates = list(range(n))
        random.shuffle(candidates)
        for x in candidates:
            if not S[x]:
                try_add(S, x)
        
        cur_size = sum(S)
        if cur_size > best_size:
            best_size = cur_size
            best_set = S[:]
    
    if best_set is None:
        best_set = [False] * n
        best_set[n-1] = True
        best_size = 1

    out = [i + 1 for i in range(n) if best_set[i]]
    return out
