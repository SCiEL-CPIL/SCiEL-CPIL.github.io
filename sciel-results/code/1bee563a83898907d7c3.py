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
    n_cand = len(candidates)
    # Index mapping: candidate x -> index x-2
    idx_of = {x: x - 2 for x in candidates}
    
    # Precompute incompatibility: incompatible[a][b] = True if lcm(a,b) <= 256
    # Use a set of tuples for O(1) lookup
    incompatible_pairs = set()
    for a in range(2, N + 1):
        for b in range(a + 1, N + 1):
            if lcm(a, b) <= 256:
                incompatible_pairs.add((a, b))
    
    # Build compatibility bitmask for each candidate
    # compat_mask[i] has bit j set if candidates[i] is compatible with candidates[j]
    compat_mask = [0] * n_cand
    for i, a in enumerate(candidates):
        mask = 0
        for j, b in enumerate(candidates):
            if i == j:
                mask |= (1 << j)
            elif (a, b) not in incompatible_pairs and (b, a) not in incompatible_pairs:
                mask |= (1 << j)
        compat_mask[i] = mask
    
    # Float reciprocal values for fast comparison
    recip_float = [1.0 / x for x in candidates]
    
    def reciprocal_sum_exact(subset):
        total = Fraction(0, 1)
        for x in subset:
            total += Fraction(1, x)
        return total
    
    def reciprocal_sum_float(subset):
        return sum(1.0 / x for x in subset)
    
    def is_valid(subset):
        s = set(subset)
        for i, a in enumerate(subset):
            for b in subset[i+1:]:
                if (a, b) in incompatible_pairs or (b, a) in incompatible_pairs:
                    return False
        return True
    
    # Greedy construction with different orderings
    def greedy(order):
        selected = []
        selected_mask = 0
        for x in order:
            i = idx_of[x]
            if (compat_mask[i] & ~selected_mask) == (1 << i):
                # x is compatible with all selected
                selected.append(x)
                selected_mask |= (1 << i)
        return selected
    
    # Generate multiple initial solutions
    # Initialize with a valid single-element set to avoid None
    best_solution = [2]
    best_sum_float = 1.0 / 2
    
    # Strategy 1: Ascending order (small first - high reciprocal)
    sol1 = greedy(candidates)
    s1 = reciprocal_sum_float(sol1)
    if s1 > best_sum_float:
        best_sum_float = s1
        best_solution = sol1[:]
    
    # Strategy 2: Descending order
    sol2 = greedy(sorted(candidates, reverse=True))
    s2 = reciprocal_sum_float(sol2)
    if s2 > best_sum_float:
        best_sum_float = s2
        best_solution = sol2[:]
    
    # Strategy 3: By number of incompatibilities (fewest first)
    inc_count = {}
    for x in candidates:
        i = idx_of[x]
        # Count incompatible: total candidates - 1 - compatible count
        comp_count = bin(compat_mask[i] & ~((1 << i))).count('1')
        inc_count[x] = n_cand - 1 - comp_count
    order3 = sorted(candidates, key=lambda x: (inc_count[x], x))
    sol3 = greedy(order3)
    s3 = reciprocal_sum_float(sol3)
    if s3 > best_sum_float:
        best_sum_float = s3
        best_solution = sol3[:]
    
    # Strategy 4: Randomized greedy - multiple restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        shuffled = candidates[:]
        random.shuffle(shuffled)
        sol_r = greedy(shuffled)
        s_r = reciprocal_sum_float(sol_r)
        if s_r > best_sum_float:
            best_sum_float = s_r
            best_solution = sol_r[:]
    
    # Local search: swap and add/remove using bitmask for speed
    current = best_solution[:]
    current_set = set(current)
    current_mask = 0
    for x in current:
        current_mask |= (1 << idx_of[x])
    current_sum_float = best_sum_float
    
    def try_improve(current, current_set, current_mask, current_sum_float):
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try adding any compatible element not in current
            # An element x can be added if it's compatible with all in current
            # i.e., (compat_mask[i] & current_mask) has all bits of current_mask set
            # equivalently: (current_mask & ~compat_mask[i]) == 0
            best_add = None
            best_add_val = 0.0
            for x in candidates:
                if x in current_set:
                    continue
                i = idx_of[x]
                if (current_mask & ~compat_mask[i]) == 0:
                    val = recip_float[i]
                    if val > best_add_val:
                        best_add_val = val
                        best_add = x
            
            if best_add is not None:
                current.append(best_add)
                current.sort()
                current_set.add(best_add)
                current_mask |= (1 << idx_of[best_add])
                current_sum_float += recip_float[idx_of[best_add]]
                improved = True
                continue
            
            # Try swapping: replace an element with a smaller compatible one
            current.sort()
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                old_elem = current[i]
                old_idx = idx_of[old_elem]
                others_mask = current_mask & ~(1 << old_idx)
                
                # Try all smaller elements not in current
                best_swap = None
                best_swap_gain = 0.0
                for x in range(2, old_elem):
                    if x in current_set:
                        continue
                    xi = idx_of[x]
                    # Check x compatible with all others
                    if (others_mask & ~compat_mask[xi]) == 0:
                        gain = recip_float[xi] - recip_float[old_idx]
                        if gain > best_swap_gain:
                            best_swap_gain = gain
                            best_swap = x
                
                if best_swap is not None:
                    current[i] = best_swap
                    current.sort()
                    current_set.remove(old_elem)
                    current_set.add(best_swap)
                    current_mask &= ~(1 << old_idx)
                    current_mask |= (1 << idx_of[best_swap])
                    current_sum_float += recip_float[idx_of[best_swap]] - recip_float[old_idx]
                    improved = True
                    break
            
            # Try pair-swap: replace two elements with two different ones
            if not improved and len(current) >= 2:
                current.sort()
                found = False
                for i in range(len(current)):
                    if time.time() > deadline - 5:
                        break
                    for j in range(i + 1, len(current)):
                        if time.time() > deadline - 5:
                            break
                        old1, old2 = current[i], current[j]
                        old1_idx, old2_idx = idx_of[old1], idx_of[old2]
                        others_mask = current_mask & ~(1 << old1_idx) & ~(1 << old2_idx)
                        
                        # Try replacing with two smaller elements
                        for x in range(2, old1):
                            if x in current_set or x == old2:
                                continue
                            xi = idx_of[x]
                            if (others_mask & ~compat_mask[xi]) != 0:
                                continue
                            for y in range(2, old2):
                                if y in current_set or y == old1 or y == x:
                                    continue
                                yi = idx_of[y]
                                if (others_mask & ~compat_mask[yi]) != 0:
                                    continue
                                if (1 << yi) & ~compat_mask[xi]:
                                    continue
                                gain = recip_float[xi] + recip_float[yi] - recip_float[old1_idx] - recip_float[old2_idx]
                                if gain > 0:
                                    current[i] = x
                                    current[j] = y
                                    current.sort()
                                    current_set.remove(old1)
                                    current_set.remove(old2)
                                    current_set.add(x)
                                    current_set.add(y)
                                    current_mask &= ~(1 << old1_idx)
                                    current_mask &= ~(1 << old2_idx)
                                    current_mask |= (1 << xi)
                                    current_mask |= (1 << yi)
                                    current_sum_float += gain
                                    improved = True
                                    found = True
                                    break
                            if found:
                                break
                    if found:
                        break
        
        return current, current_set, current_mask, current_sum_float
    
    current, current_set, current_mask, current_sum_float = try_improve(
        current, current_set, current_mask, current_sum_float)
    if current_sum_float > best_sum_float:
        best_sum_float = current_sum_float
        best_solution = current[:]
    
    # Simulated annealing step
    if time.time() < deadline - 12:
        sa_start = time.time()
        sa_time_budget = deadline - 12 - sa_start
        current = best_solution[:]
        current_set = set(current)
        current_mask = 0
        for x in current:
            current_mask |= (1 << idx_of[x])
        current_sum_float = best_sum_float
        
        def sa_neighbor(current, current_set, current_mask, current_sum_float):
            """Generate a neighbor by adding, removing, or swapping one element."""
            r = random.random()
            if r < 0.4 and len(current) < 255:
                # Add a random compatible element
                candidates_to_add = []
                for x in candidates:
                    if x in current_set:
                        continue
                    i = idx_of[x]
                    if (current_mask & ~compat_mask[i]) == 0:
                        candidates_to_add.append(x)
                if candidates_to_add:
                    x = random.choice(candidates_to_add)
                    new_set = current + [x]
                    new_set.sort()
                    new_mask = current_mask | (1 << idx_of[x])
                    return new_set, set(new_set), new_mask, current_sum_float + recip_float[idx_of[x]]
            elif r < 0.7:
                # Remove a random element
                if len(current) > 1:
                    idx = random.randint(0, len(current) - 1)
                    removed = current[idx]
                    new_set = current[:idx] + current[idx+1:]
                    new_mask = current_mask & ~(1 << idx_of[removed])
                    return new_set, set(new_set), new_mask, current_sum_float - recip_float[idx_of[removed]]
            else:
                # Swap: remove one, add one compatible
                if len(current) > 0:
                    idx = random.randint(0, len(current) - 1)
                    old_elem = current[idx]
                    old_idx = idx_of[old_elem]
                    others_mask = current_mask & ~(1 << old_idx)
                    candidates_to_add = []
                    for x in candidates:
                        if x in current_set or x == old_elem:
                            continue
                        xi = idx_of[x]
                        if (others_mask & ~compat_mask[xi]) == 0:
                            candidates_to_add.append(x)
                    if candidates_to_add:
                        x = random.choice(candidates_to_add)
                        new_set = current[:]
                        new_set[idx] = x
                        new_set.sort()
                        new_mask = current_mask & ~(1 << old_idx)
                        new_mask |= (1 << idx_of[x])
                        return new_set, set(new_set), new_mask, current_sum_float - recip_float[old_idx] + recip_float[idx_of[x]]
            return current, current_set, current_mask, current_sum_float
        
        # SA parameters
        temp = 0.05
        cooling = 0.998
        min_temp = 0.0005
        while time.time() < sa_start + sa_time_budget:
            new_set, new_set_set, new_mask, new_sum = sa_neighbor(
                current, current_set, current_mask, current_sum_float)
            delta = new_sum - current_sum_float
            if delta > 0 or (temp > min_temp and random.random() < pow(2.718281828, delta / temp)):
                current = new_set
                current_set = new_set_set
                current_mask = new_mask
                current_sum_float = new_sum
                if current_sum_float > best_sum_float:
                    best_sum_float = current_sum_float
                    best_solution = current[:]
            temp *= cooling
            if temp < min_temp:
                break
    
    # Final exact polish pass: do targeted 1-swap and 2-swap with exact Fraction
    if time.time() < deadline - 2:
        polish_set = best_solution[:]
        polish_set.sort()
        polish_sum = reciprocal_sum_exact(polish_set)
        
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try 1-swap with exact comparison
            for i in range(len(polish_set)):
                old_elem = polish_set[i]
                others = [polish_set[j] for j in range(len(polish_set)) if j != i]
                for x in range(2, old_elem):
                    if x in polish_set:
                        continue
                    if all((x, y) not in incompatible_pairs and (y, x) not in incompatible_pairs for y in others):
                        new_sum = polish_sum - Fraction(1, old_elem) + Fraction(1, x)
                        if new_sum > polish_sum:
                            polish_set[i] = x
                            polish_set.sort()
                            polish_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
            
            # Try 2-swap with exact comparison
            if not improved and len(polish_set) >= 2:
                for i in range(len(polish_set)):
                    if time.time() > deadline - 2:
                        break
                    for j in range(i + 1, len(polish_set)):
                        if time.time() > deadline - 2:
                            break
                        old1, old2 = polish_set[i], polish_set[j]
                        others = [polish_set[k] for k in range(len(polish_set)) if k != i and k != j]
                        for x in range(2, old1):
                            if x in polish_set or x == old2:
                                continue
                            if not all((x, y) not in incompatible_pairs and (y, x) not in incompatible_pairs for y in others):
                                continue
                            for y in range(2, old2):
                                if y in polish_set or y == old1 or y == x:
                                    continue
                                if not all((y, z) not in incompatible_pairs and (z, y) not in incompatible_pairs for z in others):
                                    continue
                                if (x, y) in incompatible_pairs or (y, x) in incompatible_pairs:
                                    continue
                                new_sum = polish_sum - Fraction(1, old1) - Fraction(1, old2) + Fraction(1, x) + Fraction(1, y)
                                if new_sum > polish_sum:
                                    polish_set[i] = x
                                    polish_set[j] = y
                                    polish_set.sort()
                                    polish_sum = new_sum
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        if polish_sum > reciprocal_sum_exact(best_solution):
            best_solution = polish_set
    
    # Final validation
    if not is_valid(best_solution):
        # Fallback to simple greedy
        selected = []
        for x in range(2, N + 1):
            if all((x, y) not in incompatible_pairs and (y, x) not in incompatible_pairs for y in selected):
                selected.append(x)
        return selected
    
    return best_solution
# EVOLVE-BLOCK-END
