import time
import random
import sys

def construct():
    start = time.time()
    deadline = start + 85.0
    N = 180
    
    # Precompute all forbidden triples (a, b, c) with a < b < c
    # 1/a = 1/b + 1/c  => a = bc/(b+c)
    # Let b = u*k*(u+v), c = v*k*(u+v), a = u*v*k
    # where gcd(u,v)=1, u < v, k >= 1
    triples = set()
    for u in range(1, N + 1):
        for v in range(u + 1, N + 1):
            if (u % gcd(u, v)) != 0 or (v % gcd(u, v)) != 0: # Just check gcd
                continue
            # Actually just check gcd(u,v)==1
            if gcd(u, v) != 1:
                continue
            
            # Max k such that c <= N
            # c = v * k * (u + v)
            max_k = N // (v * (u + v))
            for k in range(1, max_k + 1):
                a = u * v * k
                b = u * k * (u + v)
                c = v * k * (u + v)
                if a < b < c and c <= N:
                    # Ensure distinctness (guaranteed by u!=v)
                    triples.add((a, b, c))
    
    # Convert to list for easier access
    triple_list = list(triples)
    
    # Precompute conflicts for each x
    # conflicts[x] is a list of pairs (u, v) such that (x, u, v) is a triple
    # We need to check if both u and v are in S
    # Note: a triple (a,b,c) contributes:
    # to conflicts[a]: (b,c)
    # to conflicts[b]: (a,c)
    # to conflicts[c]: (a,b)
    
    conflicts = [[] for _ in range(N + 1)]
    for a, b, c in triple_list:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
    
    # Degree of each node (number of conflicts)
    degrees = [len(conflicts[i]) for i in range(N + 1)]
    
    def can_add(S, x):
        """Check if x can be added to set S (represented as boolean list or set)."""
        for u, v in conflicts[x]:
            if S[u] and S[v]:
                return False
        return True

    def greedy_build(order):
        S = [False] * (N + 1)
        for x in order:
            if can_add(S, x):
                S[x] = True
        return S

    def local_search_add(S, max_iters=5000):
        """Try to add any missing element repeatedly."""
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() >= deadline - 5.0:
                break
            improved = False
            iters += 1
            for x in range(1, N + 1):
                if not S[x]:
                    if can_add(S, x):
                        S[x] = True
                        improved = True
                        break
        return S

    def local_search_swap(S, rng, max_iters=10000):
        """Try swapping elements to maintain or increase size, but specifically look for increases by removal then addition."""
        # This is a 1-removal, 1-addition search.
        # If we remove y and add x, size is same.
        # We need to find a sequence that increases size.
        # Better: Remove y, then try to add ANY x. If we can add more than 1, it's a gain.
        # But we only add 1 at a time in local_search_add.
        
        # Let's try: Remove y. Run local_search_add. If size increased, keep.
        best_S = S
        best_size = sum(S)
        
        for _ in range(max_iters):
            if time.time() >= deadline - 5.0:
                break
            y = rng.randint(1, N)
            if not S[y]:
                continue
            
            S[y] = False
            # Try to fill up
            local_search_add(S, max_iters=100)
            
            new_size = sum(S)
            if new_size > best_size:
                best_S = S[:]
                best_size = new_size
            else:
                # Revert
                S[y] = True
                # Note: reverting might break validity if local_search_add added elements that depended on y being removed?
                # No, local_search_add only adds valid elements.
                # But removing y might have allowed adding x. If we put y back, x might become invalid.
                # So we must be careful.
                # Actually, if new_size <= best_size, we should revert to the state BEFORE removing y.
                # But we modified S in place.
                # We need to store the state before removal.
                
        return best_S

    # Better Local Search:
    def hill_climb(S, rng, max_iters=20000):
        """
        Move: Pick a random y in S. Remove it.
        Then greedily add any element that can be added.
        If the resulting size is larger than before, accept.
        If not, revert.
        """
        best_S = S[:]
        best_size = sum(S)
        
        for _ in range(max_iters):
            if time.time() >= deadline - 5.0:
                break
            
            # Pick random y in S
            candidates = [i for i in range(1, N + 1) if S[i]]
            if not candidates:
                break
            y = rng.choice(candidates)
            
            S[y] = False
            
            # Try to add elements
            improved = True
            while improved:
                improved = False
                for x in range(1, N + 1):
                    if not S[x]:
                        if can_add(S, x):
                            S[x] = True
                            improved = True
                            break
            
            new_size = sum(S)
            if new_size > best_size:
                best_S = S[:]
                best_size = new_size
            else:
                # Revert
                # We need to restore S to the state before removing y.
                # This is tricky because we added other elements.
                # Simplest: Keep a copy of S before the move.
                pass # This implementation is flawed for reverting. Let's use a different approach.
        
        return best_S

    # Correct Hill Climb with Copy
    def hill_climb_safe(S, rng, max_iters=10000):
        best_S = S[:]
        best_size = sum(S)
        
        for _ in range(max_iters):
            if time.time() >= deadline - 5.0:
                break
            
            candidates = [i for i in range(1, N + 1) if S[i]]
            if not candidates:
                break
            y = rng.choice(candidates)
            
            S_temp = S[:]
            S_temp[y] = False
            
            # Try to add elements
            improved = True
            while improved:
                improved = False
                for x in range(1, N + 1):
                    if not S_temp[x]:
                        if can_add(S_temp, x):
                            S_temp[x] = True
                            improved = True
                            break
            
            new_size = sum(S_temp)
            if new_size > best_size:
                S = S_temp
                best_S = S[:]
                best_size = new_size
            # If not better, S remains unchanged (we used S_temp)
        
        return best_S

    # Initialize with greedy
    # Order by degree (least constrained first)
    order = sorted(range(1, N + 1), key=lambda x: degrees[x])
    S = greedy_build(order)
    S = local_search_add(S)
    
    best_S = S[:]
    best_size = sum(S)
    
    rng = random.Random(42)
    
    # Run multiple random restarts
    for i in range(1000):
        if time.time() >= deadline - 10.0:
            break
        
        # Random order
        order_rand = list(range(1, N + 1))
        rng.shuffle(order_rand)
        
        S_try = greedy_build(order_rand)
        S_try = local_search_add(S_try)
        
        sz = sum(S_try)
        if sz > best_size:
            best_S = S_try
            best_size = sz
        
        # Hill climb
        S_try = hill_climb_safe(S_try, rng, max_iters=500)
        sz = sum(S_try)
        if sz > best_size:
            best_S = S_try
            best_size = sz

    # Final polish
    best_S = local_search_add(best_S)
    best_S = hill_climb_safe(best_S, rng, max_iters=5000)
    best_S = local_search_add(best_S)
    
    # Convert to list
    result = [i for i in range(1, N + 1) if best_S[i]]
    
    # Verify
    # Check if any triple is fully contained
    for a, b, c in triple_list:
        if best_S[a] and best_S[b] and best_S[c]:
            # Invalid! Fallback
            # Just return a known valid set, e.g., all odd numbers
            return list(range(1, N + 1, 2))
            
    return result

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
