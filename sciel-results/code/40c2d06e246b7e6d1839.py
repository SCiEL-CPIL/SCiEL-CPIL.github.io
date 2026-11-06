"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

N = 16

# EVOLVE-BLOCK-START

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _can_add(point, chosen):
    for i in range(len(chosen)):
        for j in range(i+1, len(chosen)):
            if _is_isosceles(point, chosen[i], chosen[j]):
                return False
    return True

def _greedy_construction(order, start_set=None):
    if start_set is None:
        chosen = []
    else:
        chosen = list(start_set)
    chosen_set = set(chosen)
    for p in order:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    return chosen

def _make_maximal(chosen, all_points):
    """Greedily add points to make the set maximal."""
    chosen = list(chosen)
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            chosen.append(p)
            chosen_set.add(p)
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 300.0  # 300s wall-clock, leave margin
    
    all_points = list(itertools.product(range(N), repeat=2))
    random.seed(42)
    
    best_set = None
    best_size = N * N + 1
    
    # Strategy 1: Multiple random-order greedy constructions
    while time.time() < deadline - 90:
        order = all_points[:]
        random.shuffle(order)
        chosen = _greedy_construction(order)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = list(chosen)
    
    # Strategy 2: Structured orderings
    structured_orders = [
        all_points,
        [(y, x) for x in range(N) for y in range(N)],
        sorted(all_points, key=lambda p: (p[0] + p[1], p[0])),
        sorted(all_points, key=lambda p: (p[0] - p[1], p[0])),
        sorted(all_points, key=lambda p: (p[0] * p[1], p[0])),
        sorted(all_points, key=lambda p: (p[0] ^ p[1], p[0])),
        sorted(all_points, key=lambda p: (p[0] + 2*p[1], p[0])),
        sorted(all_points, key=lambda p: (2*p[0] + p[1], p[0])),
    ]
    
    for order in structured_orders:
        if time.time() > deadline - 90:
            break
        chosen = _greedy_construction(order)
        if len(chosen) < best_size:
            best_size = len(chosen)
            best_set = list(chosen)
    
    # Strategy 3: Local search - remove points and re-maximize
    if best_set is not None and time.time() < deadline - 30:
        current = list(best_set)
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            for idx in range(len(current)):
                if time.time() > deadline - 30:
                    break
                candidate = [p for i, p in enumerate(current) if i != idx]
                new_set = _make_maximal(candidate, all_points)
                if len(new_set) < len(current):
                    current = new_set
                    best_size = len(current)
                    best_set = list(current)
                    improved = True
                    break
    
    # Strategy 4: Perturbation search
    if best_set is not None and time.time() < deadline:
        current = list(best_set)
        for _ in range(50000):
            if time.time() > deadline:
                break
            if len(current) < best_size:
                best_size = len(current)
                best_set = list(current)
            
            k = random.randint(1, 3)
            if len(current) <= k:
                continue
            to_remove = random.sample(range(len(current)), k)
            to_remove_set = set(to_remove)
            candidate = [p for i, p in enumerate(current) if i not in to_remove_set]
            new_set = _make_maximal(candidate, all_points)
            
            if len(new_set) <= len(current):
                current = new_set
    
    if best_set is None:
        best_set = _greedy_construction(all_points)
    
    return best_set
# EVOLVE-BLOCK-END
