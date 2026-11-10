import time
import random
from itertools import combinations
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    N = 128
    # Precompute GCDs
    gcds = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
    
    def can_add(S, x):
        # Check if adding x to S creates a bad triple
        # Bad triple: a, b in S, gcd(a,b) == gcd(a,x) == gcd(b,x)
        # Optimization: 
        # Group elements in S by their gcd with x.
        # If we have two elements a, b in S such that gcd(a,x) == gcd(b,x) == g,
        # then we need gcd(a,b) != g.
        # So for each g, let S_g = {a in S | gcd(a,x) == g}.
        # We need to check if there exist a, b in S_g such that gcd(a,b) == g.
        
        # This is faster if we precompute or if S is small.
        # With |S| < 60, simple O(|S|^2) is fine.
        for a, b in combinations(S, 2):
            if gcds[a][b] == gcds[a][x] == gcds[b][x]:
                return False
        return True

    def is_valid_subset(S):
        for a, b, c in combinations(S, 3):
            if gcds[a][b] == gcds[a][c] == gcds[b][c]:
                return False
        return True

    def greedy_construct(order):
        S = []
        for x in order:
            if can_add(S, x):
                S.append(x)
        return S

    def local_search(S):
        S = set(S)
        improved = True
        # To avoid infinite loops, we can limit iterations or use a counter
        iterations = 0
        while improved and time.time() < deadline:
            improved = False
            iterations += 1
            unselected = [i for i in range(1, N+1) if i not in S]
            random.shuffle(unselected)
            
            for x in unselected:
                if time.time() > deadline:
                    return list(S)
                
                if can_add(S, x):
                    S.add(x)
                    improved = True
                    unselected.remove(x)
                    continue
                
                # Find blockers
                blockers = []
                for a, b in combinations(S, 2):
                    if gcds[a][b] == gcds[a][x] == gcds[b][x]:
                        blockers.append((a, b))
                
                if not blockers:
                    continue

                # Try 1-for-1 swaps
                swap_done = False
                for a, b in blockers:
                    # Try remove a
                    if can_add(S - {a}, x):
                        S.remove(a)
                        S.add(x)
                        improved = True
                        swap_done = True
                        break
                    # Try remove b
                    if can_add(S - {b}, x):
                        S.remove(b)
                        S.add(x)
                        improved = True
                        swap_done = True
                        break
                
                if swap_done:
                    if x in unselected:
                        unselected.remove(x)
                    continue

                # Try 1-for-2 swaps
                # Remove a, b and add x, y
                # We pick the first blocker (a,b) for efficiency, or iterate?
                # Let's iterate over blockers to find a valid y
                for a, b in blockers:
                    S_minus = S - {a, b}
                    # Try to find y in unselected (excluding x)
                    # such that S_minus U {x, y} is valid
                    
                    # Pre-check: x must be addable to S_minus
                    if not can_add(S_minus, x):
                        continue
                    
                    # Now try y
                    for y in unselected:
                        if y == x:
                            continue
                        # Check if y is addable to S_minus
                        if not can_add(S_minus, y):
                            continue
                        # Check if x, y are compatible with S_minus
                        # i.e. no triple (x, y, u) with u in S_minus is bad
                        # and no triple (x, u, v) or (y, u, v) is bad (already checked by can_add)
                        # Wait, can_add(S_minus, x) checks triples (x, u, v) with u,v in S_minus.
                        # can_add(S_minus, y) checks triples (y, u, v) with u,v in S_minus.
                        # We still need to check triples (x, y, u) with u in S_minus.
                        
                        ok_xy = True
                        for u in S_minus:
                            if gcds[x][y] == gcds[x][u] == gcds[y][u]:
                                ok_xy = False
                                break
                        if ok_xy:
                            # Valid swap
                            S.remove(a)
                            S.remove(b)
                            S.add(x)
                            S.add(y)
                            improved = True
                            if x in unselected:
                                unselected.remove(x)
                            if y in unselected:
                                unselected.remove(y)
                            swap_done = True
                            break
                    if swap_done:
                        break

        return list(S)

    best_S = []
    
    # Try several random orders for greedy init
    for _ in range(30):
        if time.time() > deadline:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        S = greedy_construct(order)
        if len(S) > len(best_S):
            best_S = S
            
    # Local search on the best found
    if best_S:
        best_S = local_search(best_S)
        
    # Try local search on other good candidates if time permits
    for _ in range(10):
        if time.time() > deadline:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        S = greedy_construct(order)
        if len(S) > len(best_S) - 3:
            S = local_search(S)
            if len(S) > len(best_S):
                best_S = S
                
    # Final validation
    if best_S and not is_valid_subset(best_S):
        # Fallback to a simple valid set
        best_S = [i for i in range(1, N+1) if i % 2 == 0] # Multiples of 2?
        # Check if multiples of 2 are valid
        # gcd(2,4)=2, gcd(2,6)=2, gcd(4,6)=2 -> Bad.
        # Just return [1] if invalid
        best_S = [1]

    return best_S
