"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, lcm
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # Precompute incompatibility: (a,b) in incompatible if lcm(a,b) <= 256
    # Use a set for O(1) lookup
    incompatible = set()
    for a in range(2, N + 1):
        for b in range(a + 1, N + 1):
            if lcm(a, b) <= 256:
                incompatible.add((a, b))
    
    def is_compat(a, b):
        if a == b:
            return True
        return (min(a, b), max(a, b)) not in incompatible
    
    def reciprocal_sum(subset):
        total = Fraction(0, 1)
        for x in subset:
            total += Fraction(1, x)
        return total
    
    def check_pairwise_compatible(subset):
        s = set(subset)
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                if not is_compat(subset[i], subset[j]):
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
    
    # Compute incompatibility count for each number
    inc_count = {}
    for x in candidates:
        count = sum(1 for y in candidates if x != y and (min(x, y), max(x, y)) in incompatible)
        inc_count[x] = count
    
    # Generate multiple initial solutions
    best_solution = None
    best_sum = Fraction(0, 1)
    
    # Strategy 1: Ascending order (small first - high reciprocal)
    sol1 = greedy(candidates)
    s1 = reciprocal_sum(sol1)
    if s1 > best_sum:
        best_sum = s1
        best_solution = sol1[:]
    
    # Strategy 2: By number of incompatibilities (fewest first)
    order2 = sorted(candidates, key=lambda x: (inc_count[x], x))
    sol2 = greedy(order2)
    s2 = reciprocal_sum(sol2)
    if s2 > best_sum:
        best_sum = s2
        best_solution = sol2[:]
    
    # Strategy 3: Descending order
    sol3 = greedy(sorted(candidates, reverse=True))
    s3 = reciprocal_sum(sol3)
    if s3 > best_sum:
        best_sum = s3
        best_solution = sol3[:]
    
    # Strategy 4: Randomized greedy - multiple restarts
    num_restarts = 40
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        random.shuffle(candidates)
        sol_r = greedy(candidates)
        s_r = reciprocal_sum(sol_r)
        if s_r > best_sum:
            best_sum = s_r
            best_solution = sol_r[:]
    
    # Local search with multiple move types
    def local_search(current, current_set, current_sum):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try adding any compatible element not in current
            for x in sorted(candidates, key=lambda v: v):  # small first for max sum
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
            
            # Try swapping: replace a larger element with a smaller compatible one
            if not improved:
                current.sort()
                for i in range(len(current)):
                    if time.time() > deadline - 5:
                        break
                    old_elem = current[i]
                    others = [current[j] for j in range(len(current)) if j != i]
                    # Try replacing with smaller compatible elements
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
            
            # Try 2-swap: remove two elements, add two smaller compatible ones
            if not improved and len(current) >= 2:
                current.sort()
                for i in range(len(current)):
                    if time.time() > deadline - 5:
                        break
                    for j in range(i + 1, len(current)):
                        if time.time() > deadline - 5:
                            break
                        old1 = current[i]
                        old2 = current[j]
                        others = [current[k] for k in range(len(current)) if k != i and k != j]
                        # Try adding two smaller elements
                        for x in range(2, min(old1, old2)):
                            if x in current_set:
                                continue
                            if not all(is_compat(x, y) for y in others):
                                continue
                            for y in range(x + 1, max(old1, old2)):
                                if y in current_set or y == x:
                                    continue
                                if not all(is_compat(y, z) for z in others) or not is_compat(x, y):
                                    continue
                                new_sum = current_sum - Fraction(1, old1) - Fraction(1, old2) + Fraction(1, x) + Fraction(1, y)
                                if new_sum > current_sum:
                                    new_set = others + [x, y]
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
            
            # Try removing an element if it allows adding multiple smaller ones
            if not improved and len(current) > 1:
                for i in range(len(current)):
                    if time.time() > deadline - 5:
                        break
                    old_elem = current[i]
                    others = [current[j] for j in range(len(current)) if j != i]
                    # Check if removing old_elem allows adding smaller compatible elements
                    new_sum_delta = Fraction(0, 1)
                    added_elems = []
                    for x in range(2, old_elem):
                        if x in current_set:
                            continue
                        if all(is_compat(x, y) for y in others) and all(is_compat(x, z) for z in added_elems):
                            added_elems.append(x)
                            new_sum_delta += Fraction(1, x)
                    if new_sum_delta - Fraction(1, old_elem) > 0:
                        new_set = others + added_elems
                        new_set.sort()
                        new_set_set = set(new_set)
                        new_sum = current_sum - Fraction(1, old_elem) + new_sum_delta
                        if new_sum > current_sum:
                            current = new_set
                            current_set = new_set_set
                            current_sum = new_sum
                            improved = True
                            break
        
        return current, current_set, current_sum
    
    current, current_set, current_sum = local_search(best_solution[:], set(best_solution), best_sum)
    if current_sum > best_sum:
        best_sum = current_sum
        best_solution = current[:]
    
    # Simulated annealing for additional improvement
    if time.time() < deadline - 8:
        sa_start = time.time()
        sa_time_budget = deadline - 8 - sa_start
        
        current = best_solution[:]
        current_set = set(current)
        current_sum = best_sum
        
        def sa_neighbor(current, current_set):
            """Generate a neighbor by adding, removing, or swapping one element."""
            r = random.random()
            if r < 0.3:
                # Add a random compatible element
                candidates_to_add = [x for x in candidates if x not in current_set and all(is_compat(x, y) for y in current)]
                if candidates_to_add:
                    x = random.choice(candidates_to_add)
                    new_set = current + [x]
                    new_set.sort()
                    new_set_set = set(new_set)
                    return new_set, new_set_set
            elif r < 0.6:
                # Remove a random element
                if len(current) > 1:
                    idx = random.randint(0, len(current) - 1)
                    new_set = current[:idx] + current[idx+1:]
                    new_set_set = set(new_set)
                    return new_set, new_set_set
            else:
                # Swap a random element with a compatible one
                if len(current) > 0:
                    idx = random.randint(0, len(current) - 1)
                    old_elem = current[idx]
                    others = [current[j] for j in range(len(current)) if j != idx]
                    # Find compatible replacements
                    replacements = [x for x in candidates if x not in current_set and x != old_elem and all(is_compat(x, y) for y in others)]
                    if replacements:
                        x = random.choice(replacements)
                        new_set = current[:]
                        new_set[idx] = x
                        new_set.sort()
                        new_set_set = set(new_set)
                        return new_set, new_set_set
            
            return current[:], current_set
        
        # Simple simulated annealing
        temperature = 0.1
        cooling_rate = 0.999
        best_sa_sum = best_sum
        best_sa_solution = best_solution[:]
        
        while time.time() - sa_start < sa_time_budget:
            neighbor, neighbor_set = sa_neighbor(current, current_set)
            neighbor_sum = reciprocal_sum(neighbor)
            
            delta = neighbor_sum - current_sum
            if delta > 0 or random.random() < pow(2.718281828, delta / temperature):
                current = neighbor
                current_set = neighbor_set
                current_sum = neighbor_sum
                
                if current_sum > best_sa_sum:
                    best_sa_sum = current_sum
                    best_sa_solution = current[:]
            
            temperature *= cooling_rate
        
        if best_sa_sum > best_sum:
            best_sum = best_sa_sum
            best_solution = best_sa_solution[:]
    
    # Final validation
    if not check_pairwise_compatible(best_solution):
        # Fallback to simple greedy
        selected = []
        for x in range(2, N + 1):
            if all(is_compat(x, y) for y in selected):
                selected.append(x)
        return selected
    
    return best_solution
# EVOLVE-BLOCK-END
