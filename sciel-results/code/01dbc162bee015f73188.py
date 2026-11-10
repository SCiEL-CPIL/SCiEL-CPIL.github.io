"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    n = len(candidates)
    
    # Precompute incompatibility: incompatible[a][b] = True if lcm(a,b) <= 256
    # Use a set of frozensets for O(1) lookup
    incompatible_pairs = set()
    for a in range(2, N + 1):
        for b in range(a + 1, N + 1):
            if lcm(a, b) <= 256:
                incompatible_pairs.add((a, b))
    
    def is_compat(a, b):
        if a == b:
            return True
        return (a, b) not in incompatible_pairs and (b, a) not in incompatible_pairs
    
    def reciprocal_sum(subset):
        total = Fraction(0, 1)
        for x in subset:
            total += Fraction(1, x)
        return total
    
    def is_valid(subset):
        s = set(subset)
        for i, a in enumerate(subset):
            for b in subset[i+1:]:
                if not is_compat(a, b):
                    return False
        return True
    
    # Greedy construction with different orderings
    def greedy(order):
        selected = []
        selected_set = set()
        for x in order:
            if all(is_compat(x, y) for y in selected):
                selected.append(x)
                selected_set.add(x)
        return selected
    
    # Generate multiple initial solutions
    best_solution = None
    best_sum = Fraction(0, 1)
    
    # Strategy 1: Ascending order (small first - high reciprocal)
    sol1 = greedy(candidates)
    s1 = reciprocal_sum(sol1)
    if s1 > best_sum:
        best_sum = s1
        best_solution = sol1[:]
    
    # Strategy 2: Descending order
    sol2 = greedy(sorted(candidates, reverse=True))
    s2 = reciprocal_sum(sol2)
    if s2 > best_sum:
        best_sum = s2
        best_solution = sol2[:]
    
    # Strategy 3: By number of incompatibilities (fewest first)
    inc_count = {}
    for x in candidates:
        count = sum(1 for y in candidates if x != y and (min(x,y), max(x,y)) in incompatible_pairs)
        inc_count[x] = count
    order3 = sorted(candidates, key=lambda x: (inc_count[x], x))
    sol3 = greedy(order3)
    s3 = reciprocal_sum(sol3)
    if s3 > best_sum:
        best_sum = s3
        best_solution = sol3[:]
    
    # Strategy 4: Randomized greedy - multiple restarts
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 15:
            break
        random.shuffle(candidates)
        sol_r = greedy(candidates)
        s_r = reciprocal_sum(sol_r)
        if s_r > best_sum:
            best_sum = s_r
            best_solution = sol_r[:]
    
    # Local search: swap and add/remove
    current = best_solution[:]
    current_set = set(current)
    current_sum = best_sum
    
    def try_improve(current, current_set, current_sum):
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            
            # Try adding any compatible element not in current
            for x in candidates:
                if x in current_set:
                    continue
                if all(is_compat(x, y) for y in current):
                    new_set = current + [x]
                    new_set.sort()
                    new_set_set = set(new_set)
                    new_sum = current_sum + Fraction(1, x)
                    if new_sum > current_sum:
                        current = new_set
                        current_set = new_set_set
                        current_sum = new_sum
                        improved = True
                        break
            
            # Try swapping: replace an element with a smaller compatible one
            if not improved:
                current.sort()
                for i in range(len(current)):
                    if time.time() > deadline - 3:
                        break
                    old_elem = current[i]
                    others = [current[j] for j in range(len(current)) if j != i]
                    # Try all smaller elements, not just up to old_elem
                    for x in range(2, old_elem):
                        if x in current_set:
                            continue
                        if all(is_compat(x, y) for y in others):
                            new_set = current[:]
                            new_set[i] = x
                            new_set.sort()
                            new_set_set = set(new_set)
                            new_sum = current_sum - Fraction(1, old_elem) + Fraction(1, x)
                            if new_sum > current_sum:
                                current = new_set
                                current_set = new_set_set
                                current_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
            
            # Try pair-swap: replace two elements with two different ones
            if not improved and len(current) >= 2:
                for i in range(len(current)):
                    if time.time() > deadline - 3:
                        break
                    for j in range(i+1, len(current)):
                        if time.time() > deadline - 3:
                            break
                        old1, old2 = current[i], current[j]
                        others = [current[k] for k in range(len(current)) if k != i and k != j]
                        # Try replacing with two smaller elements
                        for x in range(2, old1):
                            if x in current_set or x == old2:
                                continue
                            if not all(is_compat(x, y) for y in others):
                                continue
                            for y in range(2, old2):
                                if y in current_set or y == old1 or y == x:
                                    continue
                                if not all(is_compat(y, z) for z in others) or not is_compat(x, y):
                                    continue
                                new_sum = current_sum - Fraction(1, old1) - Fraction(1, old2) + Fraction(1, x) + Fraction(1, y)
                                if new_sum > current_sum:
                                    new_set = current[:]
                                    new_set[i] = x
                                    new_set[j] = y
                                    new_set.sort()
                                    new_set_set = set(new_set)
                                    current = new_set
                                    current_set = new_set_set
                                    current_sum = new_sum
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return current, current_set, current_sum
    
    current, current_set, current_sum = try_improve(current, current_set, current_sum)
    if current_sum > best_sum:
        best_sum = current_sum
        best_solution = current[:]
    
    # Simulated annealing step with improved neighbor generation
    if time.time() < deadline - 10:
        sa_start = time.time()
        sa_time_budget = deadline - 10 - sa_start
        current = best_solution[:]
        current_set = set(current)
        current_sum = best_sum
        
        def sa_neighbor(current, current_set):
            """Generate a neighbor by adding, removing, or swapping one element."""
            r = random.random()
            if r < 0.4 and len(current) < 255:
                # Add a random compatible element
                candidates_to_add = [x for x in candidates if x not in current_set and all(is_compat(x, y) for y in current)]
                if candidates_to_add:
                    x = random.choice(candidates_to_add)
                    new_set = current + [x]
                    new_set.sort()
                    return new_set, set(new_set), current_sum + Fraction(1, x)
            elif r < 0.7:
                # Remove a random element
                if len(current) > 1:
                    idx = random.randint(0, len(current) - 1)
                    removed = current[idx]
                    new_set = current[:idx] + current[idx+1:]
                    return new_set, set(new_set), current_sum - Fraction(1, removed)
            else:
                # Swap: remove one, add one compatible
                if len(current) > 0:
                    idx = random.randint(0, len(current) - 1)
                    old_elem = current[idx]
                    others = [current[j] for j in range(len(current)) if j != idx]
                    candidates_to_add = [x for x in candidates if x not in current_set and x != old_elem and all(is_compat(x, y) for y in others)]
                    if candidates_to_add:
                        x = random.choice(candidates_to_add)
                        new_set = current[:]
                        new_set[idx] = x
                        new_set.sort()
                        return new_set, set(new_set), current_sum - Fraction(1, old_elem) + Fraction(1, x)
            return current, current_set, current_sum
        
        # Better SA parameters
        temp = 0.05
        cooling = 0.998
        min_temp = 0.0005
        while time.time() < sa_start + sa_time_budget:
            new_set, new_set_set, new_sum = sa_neighbor(current, current_set)
            delta = new_sum - current_sum
            if delta > 0 or (temp > min_temp and random.random() < pow(2.718281828, delta / temp)):
                current = new_set
                current_set = new_set_set
                current_sum = new_sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_solution = current[:]
            temp *= cooling
            if temp < min_temp:
                break
    
    # Final validation
    if not is_valid(best_solution):
        # Fallback to simple greedy
        selected = []
        for x in range(2, N + 1):
            if all(is_compat(x, y) for y in selected):
                selected.append(x)
        return selected
    
    return best_solution
# EVOLVE-BLOCK-END
