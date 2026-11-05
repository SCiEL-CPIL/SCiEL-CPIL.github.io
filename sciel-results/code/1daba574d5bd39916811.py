import time
from math import gcd

def construct():
    deadline = time.time() + 85
    N = 128
    
    # Precompute GCD table
    g = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g[i][j] = g[j][i] = gcd(i, j)
    
    # Precompute all bad triples
    bad_triples = set()
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            for c in range(b+1, N+1):
                if g[a][b] == g[a][c] == g[b][c]:
                    bad_triples.add((a, b, c))
    
    # For each vertex, compute the set of other vertices it conflicts with
    # Two vertices conflict if they're part of a bad triple together
    # Actually, we need: adding x to S is valid iff no bad triple is fully contained in S ∪ {x}
    # So x conflicts with pair (a,b) if (a,b,x) is a bad triple
    
    # For each x, precompute the list of pairs (a,b) such that (a,b,x) is a bad triple
    # This allows fast checking: x can be added iff no pair in S forms a bad triple with x
    
    conflicts_with = [[] for _ in range(N+1)]
    for (a, b, c) in bad_triples:
        conflicts_with[a].append((b, c))
        conflicts_with[b].append((a, c))
        conflicts_with[c].append((a, b))
    
    def can_add(S_set, x):
        """Check if x can be added to S_set without creating a bad triple."""
        for (a, b) in conflicts_with[x]:
            if a in S_set and b in S_set:
                return False
        return True
    
    def greedy_build(order, S_init=None):
        S = list(S_init) if S_init else []
        S_set = set(S)
        for x in order:
            if x not in S_set and can_add(S_set, x):
                S.append(x)
                S_set.add(x)
        return S, S_set
    
    def local_search(S, S_set):
        """Improve S by adding elements or doing swaps."""
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try adding any element not in S
            for x in range(1, N+1):
                if x not in S_set:
                    if can_add(S_set, x):
                        S.append(x)
                        S_set.add(x)
                        improved = True
                        break
            if improved:
                continue
            # Try swaps: remove one, add one or two
            for i in range(len(S)):
                if time.time() > deadline - 2:
                    break
                a = S[i]
                S.remove(a)
                S_set.discard(a)
                added = []
                for x in range(1, N+1):
                    if x not in S_set:
                        if can_add(S_set, x):
                            S.append(x)
                            S_set.add(x)
                            added.append(x)
                            if len(added) >= 2:
                                break
                if len(added) >= 2:
                    improved = True
                else:
                    # Undo
                    for y in added:
                        S.remove(y)
                        S_set.discard(y)
                    S.append(a)
                    S_set.add(a)
        return S, S_set
    
    best = []
    best_set = set()
    
    # Strategy 1: Greedy with lowest conflict degree first
    conflict_degree = [len(conflicts_with[x]) for x in range(N+1)]
    order = sorted(range(1, N+1), key=lambda x: conflict_degree[x])
    S, S_set = greedy_build(order)
    S, S_set = local_search(S, S_set)
    if len(S) > len(best):
        best = S[:]
        best_set = S_set.copy()
    
    # Strategy 2: Greedy with highest conflict degree first (opposite)
    order = sorted(range(1, N+1), key=lambda x: -conflict_degree[x])
    S, S_set = greedy_build(order)
    S, S_set = local_search(S, S_set)
    if len(S) > len(best):
        best = S[:]
        best_set = S_set.copy()
    
    # Strategy 3: Multiple random restarts
    import random
    random.seed(42)
    for restart in range(200):
        if time.time() > deadline - 5:
            break
        order = list(range(1, N+1))
        random.shuffle(order)
        S, S_set = greedy_build(order)
        S, S_set = local_search(S, S_set)
        if len(S) > len(best):
            best = S[:]
            best_set = S_set.copy()
    
    # Strategy 4: Start from a large initial set and refine
    # Try starting with all numbers and removing conflicts
    # Or start with numbers that have low conflict degree
    
    # Strategy 5: Weighted greedy - prioritize by (conflict_degree, value)
    for weight_mode in range(3):
        if time.time() > deadline - 5:
            break
        if weight_mode == 0:
            order = sorted(range(1, N+1), key=lambda x: (conflict_degree[x], -x))
        elif weight_mode == 1:
            order = sorted(range(1, N+1), key=lambda x: (conflict_degree[x], x))
        else:
            order = sorted(range(1, N+1), key=lambda x: (-x, conflict_degree[x]))
        S, S_set = greedy_build(order)
        S, S_set = local_search(S, S_set)
        if len(S) > len(best):
            best = S[:]
            best_set = S_set.copy()
    
    # Strategy 6: More aggressive local search with multiple removals
    if time.time() < deadline - 5:
        S, S_set = best[:], best_set.copy()
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            # Try removing up to 2 elements and adding up to 3
            for i in range(len(S)):
                if time.time() > deadline - 3:
                    break
                a = S[i]
                S.remove(a)
                S_set.discard(a)
                added = []
                for x in range(1, N+1):
                    if x not in S_set:
                        if can_add(S_set, x):
                            S.append(x)
                            S_set.add(x)
                            added.append(x)
                            if len(added) >= 2:
                                break
                if len(added) >= 2:
                    pass  # improved
                else:
                    for y in added:
                        S.remove(y)
                        S_set.discard(y)
                    S.append(a)
                    S_set.add(a)
        if len(S) > len(best):
            best = S[:]
            best_set = S_set.copy()
    
    return best
