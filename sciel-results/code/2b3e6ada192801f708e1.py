import time
import random
import sys
import numpy as np
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Leave 5s margin for 60s limit

    max_val = 4096
    max_sum = 3 * max_val

    best_set = []
    best_size = 0

    def try_add_fast(x, arr, used_sums):
        """Try to add x to arr (sorted list), checking all triples involving x."""
        n = len(arr)
        if n == 0:
            used_sums.add(3 * x)
            return True

        # Check (x,x,x)
        s = 3 * x
        if s in used_sums:
            return False

        # Check (x,x,y) and (x,y,y) for y in arr — vectorized
        arr_np = np.array(arr, dtype=np.int32)
        sums_xx_y = 2 * x + arr_np
        sums_x_yy = x + 2 * arr_np
        # Check if any of these are in used_sums
        for s_val in np.unique(sums_xx_y):
            if int(s_val) in used_sums:
                return False
        for s_val in np.unique(sums_x_yy):
            if int(s_val) in used_sums:
                return False

        # Check (x,y,z) for y <= z in arr — vectorized
        # Generate all pairs (i,j) with i <= j
        if n > 1:
            ii, jj = np.triu_indices(n)
            pair_sums = arr_np[ii] + arr_np[jj]
            triple_sums = x + pair_sums
            for s_val in np.unique(triple_sums):
                if int(s_val) in used_sums:
                    return False

        # All checks passed, update used_sums
        used_sums.add(3 * x)
        for s_val in sums_xx_y:
            used_sums.add(int(s_val))
        for s_val in sums_x_yy:
            used_sums.add(int(s_val))
        if n > 1:
            for s_val in triple_sums:
                used_sums.add(int(s_val))

        return True

    def greedy_build(order, deadline):
        """Build a set using greedy approach with given order."""
        current_set = []
        used_sums = set()

        for x in order:
            if time.time() > deadline - 1.0:
                break
            if try_add_fast(x, current_set, used_sums):
                current_set.append(x)

        return current_set

    def build_used_sums(arr):
        """Build the set of all triple sums from a sorted array."""
        used_sums = set()
        n = len(arr)
        arr_np = np.array(arr, dtype=np.int32)
        for i in range(n):
            a = arr_np[i]
            for j in range(i, n):
                b = arr_np[j]
                for k in range(j, n):
                    c = arr_np[k]
                    used_sums.add(int(a + b + c))
        return used_sums

    def count_collisions(candidate, arr, used_sums):
        """Count how many triple sums involving candidate would collide with existing sums."""
        n = len(arr)
        if n == 0:
            return 0
        
        arr_np = np.array(arr, dtype=np.int32)
        collision_count = 0
        
        # Check (x,x,x)
        if 3 * candidate in used_sums:
            collision_count += 1
        
        # Check (x,x,y) and (x,y,y)
        sums_xx_y = 2 * candidate + arr_np
        sums_x_yy = candidate + 2 * arr_np
        for s_val in np.unique(sums_xx_y):
            if int(s_val) in used_sums:
                collision_count += 1
        for s_val in np.unique(sums_x_yy):
            if int(s_val) in used_sums:
                collision_count += 1
        
        # Check (x,y,z) for y <= z in arr
        if n > 1:
            ii, jj = np.triu_indices(n)
            pair_sums = arr_np[ii] + arr_np[jj]
            triple_sums = candidate + pair_sums
            for s_val in np.unique(triple_sums):
                if int(s_val) in used_sums:
                    collision_count += 1
        
        return collision_count

    def simulated_annealing(initial_set, deadline, max_iterations=50):
        """Simulated annealing to escape local optima using collision density estimation."""
        current_set = sorted(initial_set)
        used_sums = build_used_sums(current_set)
        
        best_set = current_set[:]
        best_size = len(current_set)
        
        # Temperature schedule
        T_start = 50.0
        T_end = 0.1
        cooling_rate = (T_end / T_start) ** (1.0 / max(1, max_iterations))
        
        in_set = set(current_set)
        candidates = [v for v in range(1, max_val + 1) if v not in in_set]
        
        for iteration in range(max_iterations):
            if time.time() > deadline - 2.0:
                break
            
            T = T_start * (cooling_rate ** iteration)
            
            # Pick a random element to potentially remove
            n = len(current_set)
            if n == 0:
                break
            
            i = random.randint(0, n - 1)
            removed = current_set[i]
            temp_set = current_set[:i] + current_set[i+1:]
            temp_sums = build_used_sums(temp_set)
            
            # Estimate collision density for top candidates
            if len(candidates) > 50:
                # Sample a subset for speed
                sample_size = min(50, len(candidates))
                sample_candidates = random.sample(candidates, sample_size)
            else:
                sample_candidates = candidates
            
            # Score candidates by collision count (lower is better)
            scored_candidates = []
            for cand in sample_candidates:
                coll = count_collisions(cand, temp_set, temp_sums)
                scored_candidates.append((coll, cand))
            
            scored_candidates.sort()
            
            # Select a candidate using softmax-like probability based on collision count
            # Lower collision count = higher probability
            if scored_candidates:
                min_coll = scored_candidates[0][0]
                # Boltzmann weights
                weights = []
                for coll, cand in scored_candidates:
                    w = np.exp(-(coll - min_coll) / max(T, 0.01))
                    weights.append(w)
                
                total_w = sum(weights)
                if total_w > 0:
                    r = random.random() * total_w
                    cumulative = 0
                    chosen_idx = 0
                    for idx, w in enumerate(weights):
                        cumulative += w
                        if r <= cumulative:
                            chosen_idx = idx
                            break
                    
                    chosen_candidate = scored_candidates[chosen_idx][1]
                    
                    # Try to add the chosen candidate
                    if try_add_fast(chosen_candidate, temp_set, temp_sums):
                        # Accept the move
                        new_set = temp_set + [chosen_candidate]
                        new_set.sort()
                        current_set = new_set
                        used_sums = temp_sums
                        
                        # Update candidates list
                        in_set = set(current_set)
                        candidates = [v for v in range(1, max_val + 1) if v not in in_set]
                        
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = current_set[:]
                    else:
                        # Move not valid, try next best candidate
                        for coll, cand in scored_candidates[1:]:
                            if try_add_fast(cand, temp_set, temp_sums):
                                new_set = temp_set + [cand]
                                new_set.sort()
                                current_set = new_set
                                used_sums = temp_sums
                                
                                in_set = set(current_set)
                                candidates = [v for v in range(1, max_val + 1) if v not in in_set]
                                
                                if len(current_set) > best_size:
                                    best_size = len(current_set)
                                    best_set = current_set[:]
                                break
        return best_set

    # Generate diverse ordering strategies
    candidates = list(range(1, max_val + 1))

    orders = []

    # 1. Pure ascending
    orders.append(candidates[:])

    # 2. Descending (large numbers first)
    orders.append(candidates[::-1])

    # 3-6. Modular orderings
    for mod in [3, 5, 7, 11]:
        c_mod = sorted(candidates, key=lambda x: (x % mod, x))
        orders.append(c_mod)

    # 7-10. Random shuffles with fixed seeds for reproducibility
    rng = random.Random(42)
    for _ in range(4):
        c = candidates[:]
        rng.shuffle(c)
        orders.append(c)

    # 11-14. Biased random: favor certain ranges
    for bias_start, bias_end in [(1, 100), (100, 500), (500, 2000), (2000, 4096)]:
        c = candidates[:]
        rng.shuffle(c)
        # Move elements in bias range to front
        biased = [x for x in c if bias_start <= x <= bias_end]
        rest = [x for x in c if not (bias_start <= x <= bias_end)]
        orders.append(biased + rest)

    # Run greedy for each order
    for idx, order in enumerate(orders):
        if time.time() > deadline - 3.0:
            break

        result = greedy_build(order, deadline)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result[:]

            # If we have a good set, try local search
            if best_size >= 30 and time.time() < deadline - 15.0:
                improved = local_search(best_set, deadline)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_set = improved[:]

    # Final simulated annealing pass on the best set found
    if best_size >= 20 and time.time() < deadline - 5.0:
        sa_improved = simulated_annealing(best_set, deadline)
        if len(sa_improved) > best_size:
            best_size = len(sa_improved)
            best_set = sa_improved[:]

    return best_set
