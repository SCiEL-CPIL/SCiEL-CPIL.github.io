"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85.0
    n = 512
    
    def can_add(x, in_set):
        # Check all 4-APs that could include x
        # x as 4th term: x-3d, x-2d, x-d, x
        for d in range(1, (x // 3) + 1):
            if in_set[x-d] and in_set[x-2*d] and in_set[x-3*d]:
                return False
        # x as 1st term: x, x+d, x+2d, x+3d
        for d in range(1, (n - x) // 3 + 1):
            if in_set[x+d] and in_set[x+2*d] and in_set[x+3*d]:
                return False
        # x as 2nd term: x-d, x, x+d, x+2d
        max_d = min(x, (n - x) // 2)
        for d in range(1, max_d + 1):
            if in_set[x-d] and in_set[x+d] and in_set[x+2*d]:
                return False
        # x as 3rd term: x-2d, x-d, x, x+d
        max_d = min(x // 2, n - x)
        for d in range(1, max_d + 1):
            if in_set[x-2*d] and in_set[x-d] and in_set[x+d]:
                return False
        return True
    
    def greedy(order, in_set, selected):
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, in_set):
                in_set[x] = True
                selected.append(x)
    
    def base3_sum(x):
        s = 0
        while x > 0:
            s += x % 3
            x //= 3
        return s
    
    best_set = []
    
    # Strategy 1: Natural order
    in_set1 = [False] * (n + 1)
    selected1 = []
    greedy(list(range(1, n + 1)), in_set1, selected1)
    if len(selected1) > len(best_set):
        best_set = list(selected1)
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Strategy 2: Base-3 digit sum order
    order2 = sorted(range(1, n + 1), key=lambda x: (base3_sum(x), x))
    in_set2 = [False] * (n + 1)
    selected2 = []
    greedy(order2, in_set2, selected2)
    if len(selected2) > len(best_set):
        best_set = list(selected2)
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Strategy 3: Reverse natural order
    order3 = list(range(n, 0, -1))
    in_set3 = [False] * (n + 1)
    selected3 = []
    greedy(order3, in_set3, selected3)
    if len(selected3) > len(best_set):
        best_set = list(selected3)
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Strategy 4: Random permutations (multiple tries)
    random.seed(42)
    for trial in range(10):
        if time.time() > deadline - 5:
            break
        order4 = list(range(1, n + 1))
        random.shuffle(order4)
        in_set4 = [False] * (n + 1)
        selected4 = []
        greedy(order4, in_set4, selected4)
        if len(selected4) > len(best_set):
            best_set = list(selected4)
    
    if time.time() > deadline:
        return sorted(best_set)
    
    # Local improvement: try to add elements that were rejected
    # by checking if removing one existing element allows adding two new ones
    in_set_best = [False] * (n + 1)
    for x in best_set:
        in_set_best[x] = True
    
    # Try simple local search: for each element not in set, see if we can add it
    # by removing at most one element from the set
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        current_set = list(best_set)
        in_set_cur = [False] * (n + 1)
        for x in current_set:
            in_set_cur[x] = True
        
        # Try adding each missing element
        for x in range(1, n + 1):
            if time.time() > deadline - 2:
                break
            if in_set_cur[x]:
                continue
            if can_add(x, in_set_cur):
                in_set_cur[x] = True
                current_set.append(x)
                improved = True
                break
        
        if improved:
            best_set = current_set
            in_set_best = in_set_cur
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
