import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 180
    
    # Precompute forbidden triples and conflicts
    # Triple (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # (b-a)(c-a) = a^2
    conflict = defaultdict(list)
    
    for a in range(1, N + 1):
        a2 = a * a
        for u in range(1, a):
            if a2 % u == 0:
                v = a2 // u
                b = a + u
                c = a + v
                if b < c and c <= N:
                    # (a, b, c) is a valid triple
                    conflict[a].append((b, c))
                    conflict[b].append((a, c))
                    conflict[c].append((a, b))

    def check_conflict(x, S):
        """Check if adding x to S creates any forbidden triple."""
        for y, z in conflict[x]:
            if y in S and z in S:
                return True
        return False

    def greedy_build(order):
        S = set()
        for x in order:
            if not check_conflict(x, S):
                S.add(x)
        return S

    best_S = set()
    best_size = 0
    
    # Strategy 1: Deterministic orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
    ]
    
    # Strategy 2: Random orders
    for _ in range(100):
        nums = list(range(1, N + 1))
        random.shuffle(nums)
        orders.append(nums)
        
    for order in orders:
        if time.time() > deadline - 40:
            break
        S = greedy_build(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)
            
    # Local Search
    S = set(best_S)
    E = set(range(1, N + 1)) - S
    
    # Phase 1: Add any possible element
    changed = True
    while changed and time.time() < deadline - 20:
        changed = False
        for x in list(E):
            if not check_conflict(x, S):
                S.add(x)
                E.remove(x)
                changed = True
                best_size = len(S)
                best_S = set(S)
                
    # Phase 2: Local Search (Swap)
    # Try to remove x from S and add y from E
    # We do this by iterating x in S, removing it, then greedily adding from E.
    # If the resulting set is larger, keep it.
    
    for _ in range(50): # Iterations of local search
        if time.time() > deadline - 5:
            break
        
        improved = False
        S_list = list(S)
        random.shuffle(S_list)
        
        for x in S_list:
            if time.time() > deadline - 5:
                break
                
            # Remove x
            S.remove(x)
            E.add(x)
            
            # Try to add elements from E
            added = []
            for y in list(E):
                if y == x:
                    continue
                if not check_conflict(y, S):
                    S.add(y)
                    added.append(y)
                    E.remove(y)
            
            # Check if we improved
            # We removed 1, added len(added)
            # Net change: len(added) - 1
            if len(added) > 1:
                best_size = len(S)
                best_S = set(S)
                improved = True
            elif len(added) == 1:
                # Net change 0. We might accept this to escape local minima, 
                # but strictly we want larger. 
                # Let's accept if it allows future improvements? 
                # For now, revert if not strictly better to maintain size.
                # However, sometimes a same-size move is needed.
                # Let's try to accept same-size moves with low probability or if it changes structure.
                # Given time constraints, let's stick to strict improvement first.
                # If we find a strict improvement, we update best_S.
                pass
            else:
                # Revert
                for y in added:
                    S.remove(y)
                    E.add(y)
                S.add(x)
                E.remove(x)
                
        if not improved:
            # Try random restarts if stuck
            if time.time() < deadline - 10:
                for _ in range(10):
                    nums = list(range(1, N + 1))
                    random.shuffle(nums)
                    S_new = greedy_build(nums)
                    # Quick add
                    E_new = set(range(1, N + 1)) - S_new
                    for y in list(E_new):
                        if not check_conflict(y, S_new):
                            S_new.add(y)
                            E_new.remove(y)
                    if len(S_new) > best_size:
                        best_size = len(S_new)
                        best_S = set(S_new)
                        improved = True
                        break

    return list(best_S)
