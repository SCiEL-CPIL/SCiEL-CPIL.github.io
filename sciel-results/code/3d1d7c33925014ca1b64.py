import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = 85.0
    n = 256
    
    def can_add(included, x):
        """Check if x can be added to included without creating a 3AP."""
        for a in included:
            # x as midpoint: need c = 2x - a in included
            c = 2 * x - a
            if 1 <= c <= n and c in included and c != x and c != a:
                return False
            # x as endpoint: need midpoint m = (a+x)/2 in included
            if (a + x) % 2 == 0:
                m = (a + x) // 2
                if 1 <= m <= n and m in included and m != x and m != a:
                    return False
        return True
    
    # Greedy initial solution: add elements in order of increasing x (decreasing weight)
    S_greedy = set()
    for x in range(1, n + 1):
        if can_add(S_greedy, x):
            S_greedy.add(x)
    
    best_set = set(S_greedy)
    best_weight = sum(1.0 / x for x in best_set)
    
    random.seed(42)
    
    def local_search(S, w):
        """Improve S via additions and single swaps."""
        improved = True
        while improved:
            improved = False
            # Add all possible elements
            for x in range(1, n + 1):
                if x not in S and can_add(S, x):
                    S.add(x)
                    w += 1.0 / x
                    improved = True
            # Try all single swaps: remove one, add the best possible
            for xr in list(S):
                if time.time() > deadline - 5:
                    return S, w
                S2 = set(S)
                S2.discard(xr)
                w2 = w - 1.0 / xr
                best_xa = None
                best_wc = w
                for xa in range(1, n + 1):
                    if xa not in S2 and can_add(S2, xa):
                        wc = w2 + 1.0 / xa
                        if wc > best_wc + 1e-12:
                            best_wc = wc
                            best_xa = xa
                if best_xa is not None:
                    S = S2
                    S.add(best_xa)
                    w = best_wc
                    improved = True
        return S, w
    
    # Improve greedy solution
    S, w = local_search(set(S_greedy), best_weight)
    if w > best_weight:
        best_set, best_weight = S, w
    
    # Multiple restarts with perturbation
    for r in range(300):
        if time.time() > deadline - 10:
            break
        S_r = set(S_greedy)
        k = random.randint(1, 15)
        for _ in range(k):
            if S_r:
                S_r.discard(random.choice(list(S_r)))
        # Refill greedily
        for x in range(1, n + 1):
            if x not in S_r and can_add(S_r, x):
                S_r.add(x)
        S_r, w_r = local_search(S_r, sum(1.0 / x for x in S_r))
        if w_r > best_weight:
            best_set, best_weight = S_r, w_r
    
    return sorted(best_set)
