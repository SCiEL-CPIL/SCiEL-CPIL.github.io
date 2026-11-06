import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    N = 24
    # Calculate LCM of 1..24
    L = 1
    for i in range(1, N + 1):
        L = L * i // gcd(L, i)
        
    v = [0] * (N + 1)
    for i in range(1, N + 1):
        v[i] = L // i

    def is_subset_sum(target, values):
        if target == 0:
            return True
        # Sort descending to prune faster
        values = sorted(values, reverse=True)
        reachable = {0}
        for val in values:
            new_reachable = set()
            for s in reachable:
                ns = s + val
                if ns == target:
                    return True
                if ns < target:
                    new_reachable.add(ns)
            reachable.update(new_reachable)
        return target in reachable

    def is_valid(S):
        S_list = list(S)
        for x in S_list:
            others = [v[y] for y in S_list if y != x]
            if is_subset_sum(v[x], others):
                return False
        return True

    def greedy_init(order):
        S = set()
        for x in order:
            if is_valid(S | {x}):
                S.add(x)
        return S

    best_S = set()
    best_size = 0
    
    # Initial Greedy
    # Use a fixed seed for initialization to ensure a strong start
    random.seed(42)
    for _ in range(50):
        order = list(range(1, N + 1))
        random.shuffle(order)
        S = greedy_init(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    for order in [list(range(1, N + 1)), list(range(N, 0, -1))]:
        S = greedy_init(order)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    S = best_S
    current_size = len(S)
    
    no_improve_count = 0
    
    while time.time() < deadline:
        # Try to add an element
        added = False
        candidates = [x for x in range(1, N + 1) if x not in S]
        random.shuffle(candidates)
        for z in candidates:
            if is_valid(S | {z}):
                S.add(z)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_S = S
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                added = True
                break
        
        if not added:
            # Try swaps
            if S:
                x = random.choice(list(S))
                z_candidates = [x for x in range(1, N + 1) if x not in S]
                if z_candidates:
                    random.shuffle(z_candidates)
                    for z in z_candidates:
                        new_S = (S - {x}) | {z}
                        if is_valid(new_S):
                            S = new_S
                            no_improve_count += 1
                            break
            else:
                break

        # Restart if stuck
        if no_improve_count > 200:
            S = best_S
            current_size = len(S)
            no_improve_count = 0
            # Perturb: remove a random element to allow re-addition of different ones
            if S:
                x = random.choice(list(S))
                S.remove(x)
                current_size -= 1

    return list(best_S)
