import time
import random
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Precompute bad_pairs[x]: set of (a,b) with a<b, a!=x, b!=x
    # such that lcm(a,b) == lcm(a,x) == lcm(b,x)
    N = 128
    bad_pairs = [set() for _ in range(N + 1)]
    
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                l1 = lcm(a, b)
                l2 = lcm(a, x)
                if l1 == l2:
                    l3 = lcm(b, x)
                    if l1 == l3:
                        # (a, b) is a bad pair for x
                        bad_pairs[x].add((a, b))
    
    def can_add(x, selected_set, selected_list):
        """Check if x can be added to the set without creating a bad triple."""
        # For each pair (a,b) in selected, check if (a,b) is in bad_pairs[x]
        n = len(selected_list)
        for i in range(n):
            a = selected_list[i]
            for j in range(i + 1, n):
                b = selected_list[j]
                if a < b:
                    if (a, b) in bad_pairs[x]:
                        return False
                else:
                    if (b, a) in bad_pairs[x]:
                        return False
        return True
    
    def greedy_add(order):
        """Greedy add elements in given order."""
        selected = []
        selected_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if can_add(x, selected_set, selected):
                selected.append(x)
                selected_set.add(x)
        return selected
    
    # Multi-restart random greedy
    best_set = set()
    best_size = 0
    
    # First try a few deterministic orders
    deterministic_orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        # Even numbers first, then odds
        [x for x in range(1, N + 1) if x % 2 == 0] + [x for x in range(1, N + 1) if x % 2 == 1],
        # Odds first, then evens
        [x for x in range(1, N + 1) if x % 2 == 1] + [x for x in range(1, N + 1) if x % 2 == 0],
    ]
    
    for order in deterministic_orders:
        if time.time() > deadline:
            break
        result = greedy_add(order)
        if len(result) > best_size:
            best_size = len(result)
            best_set = set(result)
    
    # Random restarts
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 5:  # Leave 5s for local search
        order = list(range(1, N + 1))
        rng.shuffle(order)
        result = greedy_add(order)
        attempts += 1
        if len(result) > best_size:
            best_size = len(result)
            best_set = set(result)
    
    # Local search: try to add excluded elements, or swap
    if time.time() < deadline - 2:
        current = set(best_set)
        current_list = list(current)
        excluded = [x for x in range(1, N + 1) if x not in current]
        
        # Try adding each excluded element
        for x in excluded:
            if time.time() > deadline:
                break
            if can_add(x, current, current_list):
                current.add(x)
                current_list.append(x)
        
        # Try swaps: remove one element, add one or two others
        for x in list(excluded):
            if time.time() > deadline:
                break
            if x in current:
                continue
            # Try removing one element y from current and adding x
            for y in current_list:
                if time.time() > deadline:
                    break
                if y == x:
                    continue
                # Check if x can be added when y is removed
                temp_list = [z for z in current_list if z != y]
                if can_add(x, set(temp_list), temp_list):
                    # Also check that removing y doesn't break anything (it shouldn't, since we're removing)
                    current.remove(y)
                    current.add(x)
                    current_list = list(current)
                    break
    
    result = list(best_set)
    if not result:
        result = [1]
    
    return result
