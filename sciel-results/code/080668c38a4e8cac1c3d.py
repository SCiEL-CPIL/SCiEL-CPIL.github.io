"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    deadline = time.time() + 80
    
    n = 128
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
    
    # Find all forbidden triples and build pair->conflicting_z map
    pair_conflicts = {}  # (x,y) with x<y -> set of z such that {x,y,z} is forbidden
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    key = (a, b)
                    if key not in pair_conflicts:
                        pair_conflicts[key] = set()
                    pair_conflicts[key].add(c)
                    key2 = (a, c)
                    if key2 not in pair_conflicts:
                        pair_conflicts[key2] = set()
                    pair_conflicts[key2].add(b)
                    key3 = (b, c)
                    if key3 not in pair_conflicts:
                        pair_conflicts[key3] = set()
                    pair_conflicts[key3].add(a)
    
    def can_add(selected_set, x):
        """Check if x can be added to selected_set without creating a forbidden triple."""
        for (a, b) in combinations(selected_set, 2):
            if a > b:
                a, b = b, a
            key = (a, b)
            if key in pair_conflicts and x in pair_conflicts[key]:
                return False
        return True
    
    def greedy(order):
        selected = set()
        for x in order:
            if can_add(selected, x):
                selected.add(x)
        return selected
    
    base_order = list(range(1, n + 1))
    
    # Multiple greedy orders
    best = greedy(base_order)
    best = max(best, greedy(base_order[::-1]), key=len)
    
    # Order by degree (number of pairs that conflict)
    deg = [0] * (n + 1)
    for (a, b) in pair_conflicts:
        deg[a] += 1
        deg[b] += 1
    order_low_deg = sorted(range(1, n + 1), key=lambda x: deg[x])
    best = max(best, greedy(order_low_deg), key=len)
    
    # Local search: try to improve by swaps
    def local_search(selected):
        selected = set(selected)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any element
            for y in range(1, n + 1):
                if y not in selected and can_add(selected, y):
                    selected.add(y)
                    improved = True
                    break
            if not improved:
                # Try single swaps
                for x in list(selected):
                    for y in range(1, n + 1):
                        if y in selected:
                            continue
                        new_set = (selected - {x}) | {y}
                        if can_add(new_set - {y}, y):
                            selected = new_set
                            improved = True
                            break
                    if improved:
                        break
        return selected
    
    best_set = local_search(best)
    
    # Randomized restarts
    random.seed(12345)
    for _ in range(50):
        if time.time() > deadline:
            break
        rnd_order = base_order[:]
        random.shuffle(rnd_order)
        candidate = greedy(rnd_order)
        candidate = local_search(candidate)
        if len(candidate) > len(best_set):
            best_set = candidate
    
    return list(best_set)
# EVOLVE-BLOCK-END
