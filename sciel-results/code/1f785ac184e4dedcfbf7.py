import time
from math import gcd
from itertools import combinations
import random

def construct():
    deadline = time.time() + 85
    N = 128
    # Precompute GCD table
    g = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g[i][j] = g[j][i] = gcd(i, j)
    
    def is_bad_triple(a, b, c):
        return g[a][b] == g[a][c] == g[b][c]
    
    def can_add(S, x, S_set):
        """Check if x can be added to set S without creating a bad triple."""
        for i in range(len(S)):
            a = S[i]
            if a == x:
                continue
            for j in range(i+1, len(S)):
                b = S[j]
                if b == x:
                    continue
                if is_bad_triple(a, b, x):
                    return False
        return True
    
    def greedy_build(order):
        S = []
        S_set = set()
        for x in order:
            if x not in S_set and can_add(S, x, S_set):
                S.append(x)
                S_set.add(x)
        return S
    
    def local_search(S, S_set):
        """Try to improve S by adding elements or doing swaps."""
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try adding any element not in S
            for x in range(1, N+1):
                if x not in S_set:
                    if can_add(S, x, S_set):
                        S.append(x)
                        S_set.add(x)
                        improved = True
                        break
            if improved:
                continue
            # Try swaps: remove one, add two
            for i in range(len(S)):
                if time.time() > deadline - 2:
                    break
                a = S[i]
                S.remove(a)
                S_set.discard(a)
                added = 0
                for x in range(1, N+1):
                    if x not in S_set:
                        if can_add(S, x, S_set):
                            S.append(x)
                            S_set.add(x)
                            added += 1
                            if added >= 2:
                                break
                if added >= 2:
                    improved = True
                else:
                    # Undo
                    for _ in range(added):
                        y = S.pop()
                        S_set.discard(y)
                    S.append(a)
                    S_set.add(a)
        return S
    
    best = []
    
    # Multiple greedy strategies
    base_order = list(range(1, N+1))
    
    # Strategy 1: ascending
    S = greedy_build(base_order)
    S = local_search(S, set(S))
    if len(S) > len(best):
        best = S[:]
    
    # Strategy 2: descending
    S = greedy_build(base_order[::-1])
    S = local_search(S, set(S))
    if len(S) > len(best):
        best = S[:]
    
    # Strategy 3: random restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        order = base_order[:]
        random.shuffle(order)
        S = greedy_build(order)
        S = local_search(S, set(S))
        if len(S) > len(best):
            best = S[:]
    
    # Strategy 4: conflict-aware ordering
    # Count for each element how many bad triples it's involved in
    conflict_count = [0]*(N+1)
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            for c in range(b+1, N+1):
                if is_bad_triple(a, b, c):
                    conflict_count[a] += 1
                    conflict_count[b] += 1
                    conflict_count[c] += 1
    order = sorted(range(1, N+1), key=lambda x: conflict_count[x])
    S = greedy_build(order)
    S = local_search(S, set(S))
    if len(S) > len(best):
        best = S[:]
    
    # More random restarts with conflict-aware perturbation
    for restart in range(100):
        if time.time() > deadline - 5:
            break
        order = base_order[:]
        random.shuffle(order)
        S = greedy_build(order)
        S = local_search(S, set(S))
        if len(S) > len(best):
            best = S[:]
    
    return best
