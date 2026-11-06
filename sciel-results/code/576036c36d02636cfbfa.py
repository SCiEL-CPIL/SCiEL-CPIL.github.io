import time
import sys
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin for 60s limit
    
    N = 256
    
    def can_add(y, in_set, active_list):
        """Check if y can be added to the set defined by in_set/active_list without creating a 3-AP."""
        for x in active_list:
            # Case 1: x, y are endpoints, middle is (x+y)/2
            if (x + y) % 2 == 0:
                mid = (x + y) // 2
                if 1 <= mid <= N and in_set[mid]:
                    return False
            # Case 2: x is first, y is second, third is 2y-x
            third = 2 * y - x
            if 1 <= third <= N and in_set[third]:
                return False
            # Case 3: y is first, x is second, third is 2x-y
            third = 2 * x - y
            if 1 <= third <= N and in_set[third]:
                return False
        return True

    def get_score(in_set):
        return sum(1.0 / x for x in range(1, N + 1) if in_set[x])

    def greedy_fill(in_set, active_list, order):
        for x in order:
            if not in_set[x] and can_add(x, in_set, active_list):
                in_set[x] = True
                active_list.append(x)
        return in_set, active_list

    best_set = [False] * (N + 1)
    best_active = []
    best_score = 0.0

    # Strategy 1: Greedy from 1 to N
    in_set = [False] * (N + 1)
    active_list = []
    order = list(range(1, N + 1))
    in_set, active_list = greedy_fill(in_set, active_list, order)
    s = get_score(in_set)
    if s > best_score:
        best_score = s
        best_set = in_set[:]
        best_active = active_list[:]

    # Strategy 2: Randomized Greedy
    for trial in range(100):
        if time.time() > deadline - 10:
            break
        in_set = [False] * (N + 1)
        active_list = []
        order = list(range(1, N + 1))
        random.shuffle(order)
        # Bias towards small numbers?
        # Just shuffle is fine for diversity
        in_set, active_list = greedy_fill(in_set, active_list, order)
        s = get_score(in_set)
        if s > best_score:
            best_score = s
            best_set = in_set[:]
            best_active = active_list[:]

    # Strategy 3: Simulated Annealing / Local Search from best
    # We will try to improve best_set
    in_set = best_set[:]
    active_list = best_active[:]
    cur_score = best_score
    
    # Precompute reciprocals for speed
    recips = [0.0] * (N + 1)
    for i in range(1, N + 1):
        recips[i] = 1.0 / i

    T = 1.0
    # Cooling schedule
    # We have ~40s left. 
    # Each SA step involves a few can_add calls.
    # can_add is O(|S|) ~ 50-100 ops.
    # 10000 steps * 100 ops = 1e6 ops. Very fast.
    # We can do many steps.
    
    steps = 0
    last_improvement = time.time()
    
    while time.time() < deadline - 2:
        if time.time() - last_improvement > 5.0:
            # No improvement for 5s, reduce T or restart?
            # Let's just reduce T aggressively
            T *= 0.9
            if T < 0.01:
                break
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.3:
            # Add a random element
            candidates = [i for i in range(1, N + 1) if not in_set[i]]
            if not candidates:
                continue
            y = random.choice(candidates)
            if can_add(y, in_set, active_list):
                # Accept add
                in_set[y] = True
                active_list.append(y)
                delta = recips[y]
                if delta > 0:
                    cur_score += delta
                    if cur_score > best_score:
                        best_score = cur_score
                        best_set = in_set[:]
                        best_active = active_list[:]
                        last_improvement = time.time()
            else:
                # Rejected by constraint, treat as energy penalty? 
                # Or just skip. Let's skip invalid moves in SA for simplicity, 
                # but that makes it a random walk on valid states.
                pass
                
        elif move_type < 0.6:
            # Remove a random element
            if not active_list:
                continue
            idx = random.randint(0, len(active_list) - 1)
            x = active_list[idx]
            in_set[x] = False
            active_list.pop(idx)
            delta = -recips[x]
            # Always accept removal? No, we want to maximize. 
            # In SA, we accept bad moves with prob exp(delta/T).
            # But removing reduces score. We should only remove if it enables better adds later.
            # So we must accept removals with some probability, or only if we plan to add.
            # Let's do Swap instead of Remove/Add separately to keep size stable-ish.
            
            # Let's revert this branch and just do Swap.
            in_set[x] = True
            active_list.insert(idx, x)
            
        else:
            # Swap: Remove x, Add y
            if not active_list:
                continue
            idx = random.randint(0, len(active_list) - 1)
            x = active_list[idx]
            
            # Remove x temporarily
            in_set[x] = False
            active_list.pop(idx)
            
            # Try to add y
            candidates = [i for i in range(1, N + 1) if not in_set[i]]
            if candidates:
                y = random.choice(candidates)
                if can_add(y, in_set, active_list):
                    # Calculate delta
                    delta = recips[y] - recips[x]
                    # Accept if delta > 0 or with prob exp(delta/T)
                    if delta > 0 or random.random() < (2.718281828 ** (delta / T)):
                        # Accept swap
                        in_set[y] = True
                        active_list.append(y)
                        cur_score += delta
                        if cur_score > best_score:
                            best_score = cur_score
                            best_set = in_set[:]
                            best_active = active_list[:]
                            last_improvement = time.time()
                    else:
                        # Reject swap, restore x
                        in_set[x] = True
                        active_list.insert(idx, x)
                else:
                    # y not valid, restore x
                    in_set[x] = True
                    active_list.insert(idx, x)
            
            # Cool down
            T *= 0.999
            steps += 1

    return [x for x in range(1, N + 1) if best_set[x]]
