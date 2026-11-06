import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    limit = 55.0  # Safety margin for 90s kill limit
    
    n = 256
    weights = np.array([1.0 / i for i in range(1, n + 1)])
    
    # Build conflict graph: two elements conflict if they can be part of a 3-AP
    # For a set S to have no 3-AP, for any a < b < c in S, we need (c-a) != 2*(b-a)
    # Equivalently: no three elements form an AP
    # Two elements x, y "conflict" if there exists a third element z such that {x,y,z} is a 3-AP
    # But actually, the constraint is on triples, not pairs. Let me think differently.
    
    # Better approach: Precompute for each element, which other elements it "blocks"
    # If we include element x, then for any other element y, if there exists z such that
    # {x, y, z} is a 3-AP, then we cannot include all three.
    
    # Actually, the cleanest formulation: we want a subset S with no 3-AP.
    # A 3-AP is a set {a, a+d, a+2d} for d > 0.
    # So we need: for all a, d >= 1, we cannot have all of {a, a+d, a+2d} in S.
    
    # For branch and bound, let's use the following approach:
    # 1. Order elements by weight (descending) for the branching
    # 2. At each node, we decide to include or exclude the next element
    # 3. Upper bound: current weight + sum of weights of all remaining elements that don't conflict
    
    # Precompute: for each element i, compute which elements it "forbids" given a partial solution
    # This is complex. Instead, let's use a simpler but effective approach:
    
    # Greedy initialization + local search with multiple restarts
    
    def can_add(S_set, x):
        """Check if adding x to S_set creates a 3-AP"""
        for y in S_set:
            # Check if x is middle: y, x, z => z = 2*x - y
            z = 2 * x - y
            if z in S_set:
                return False
            # Check if y is middle: x, y, z => z = 2*y - x
            z = 2 * y - x
            if z in S_set:
                return False
            # Check if z is middle: x, z, y where z = (x+y)/2
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if z in S_set:
                    return False
        return True
    
    def greedy_init(order):
        """Greedy: add elements in given order if they don't conflict"""
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S
    
    def local_search(S_set, max_iters=5000, deadline=None):
        """1-swap local search: try replacing one element with another"""
        S_list = sorted(S_set)
        current_weight = sum(weights[x - 1] for x in S_list)
        
        for _ in range(max_iters):
            if deadline and time.time() > deadline:
                break
            improved = False
            for i, x in enumerate(S_list):
                # Try removing x
                S_temp = set(S_list)
                S_temp.remove(x)
                # Try adding each element not in S
                for y in range(1, n + 1):
                    if y in S_temp:
                        continue
                    if can_add(S_temp, y):
                        new_weight = current_weight - weights[x - 1] + weights[y - 1]
                        if new_weight > current_weight + 1e-12:
                            S_temp.add(y)
                            S_list = sorted(S_temp)
                            current_weight = new_weight
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        return S_list, current_weight
    
    # Strategy: Try multiple greedy orderings and local search from each
    best_set = set()
    best_weight = 0.0
    
    # Ordering 1: by weight descending (natural greedy)
    order1 = sorted(range(1, n + 1), key=lambda x: -weights[x - 1])
    S1 = greedy_init(order1)
    S1_list, w1 = local_search(S1, max_iters=3000, deadline=start_time + limit - 5)
    if w1 > best_weight:
        best_weight = w1
        best_set = set(S1_list)
    
    # Ordering 2: random shuffles
    np.random.seed(42)
    for trial in range(5):
        if time.time() > start_time + limit - 10:
            break
        order = list(range(1, n + 1))
        np.random.shuffle(order)
        S_t = greedy_init(order)
        S_t_list, w_t = local_search(S_t, max_iters=2000, deadline=start_time + limit - 5)
        if w_t > best_weight:
            best_weight = w_t
            best_set = set(S_t_list)
    
    # Additional refinement: try to add more elements or swap for better ones
    # Multiple passes of local search
    for _ in range(3):
        if time.time() > start_time + limit - 3:
            break
        best_list, best_weight = local_search(list(best_set), max_iters=5000, deadline=start_time + limit - 2)
    
    if best_set:
        return sorted(best_set)
    else:
        # Fallback: return {1}
        return [1]
