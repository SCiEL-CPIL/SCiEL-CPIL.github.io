import time
import random
import math
import numpy as np

def set_sizes(A):
    A = list(A)
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return len(sums), len(prods)

def objective(A):
    s, p = set_sizes(A)
    return max(s, p)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin under 90s
    
    # We search for the optimal "shape" in a small integer range [1, SMALL_MAX]
    # and then scale it up.
    # Scale invariance: |kA+kA| = |A+A| and |kA*kA| = |A*A|
    
    SMALL_MAX = 60 
    n = 20
    
    best_obj_global = float('inf')
    best_set_global = None

    def local_search(current_set, max_iters=200, deadline=None):
        """Steepest descent local search to polish a candidate."""
        current_set = list(current_set)
        current_obj = objective(current_set)
        
        for _ in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            improved = False
            best_neighbor = None
            best_neighbor_obj = current_obj
            
            # Try replacing each element with nearby values and some random
            current_set_sorted = sorted(current_set)
            for i in range(n):
                old_val = current_set[i]
                
                # Try nearby values (within ±5) and a few random ones
                candidates = set()
                for delta in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
                    v = old_val + delta
                    if 1 <= v <= SMALL_MAX and v != old_val:
                        candidates.add(v)
                # Add 4 random candidates
                for _ in range(4):
                    candidates.add(random.randint(1, SMALL_MAX))
                
                for new_val in candidates:
                    if new_val == old_val or new_val in current_set:
                        continue
                    neighbor = current_set[:]
                    neighbor[i] = new_val
                    if len(set(neighbor)) != n:
                        continue
                    n_obj = objective(neighbor)
                    if n_obj < best_neighbor_obj:
                        best_neighbor_obj = n_obj
                        best_neighbor = neighbor[:]
            
            if best_neighbor is not None and best_neighbor_obj < current_obj:
                current_set = best_neighbor
                current_obj = best_neighbor_obj
                improved = True
            else:
                break
                
        return current_set, current_obj

    def simulated_annealing(start_set, max_iters=5000):
        current_set = list(start_set)
        current_obj = objective(current_set)
        
        # Initial temperature based on typical objective variation
        T = 15.0
        T_min = 0.1
        cooling_rate = 0.995
        
        best_local_set = current_set[:]
        best_local_obj = current_obj
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            neighbor = current_set[:]
            move_type = random.random()
            
            if move_type < 0.5:
                # Replace one element
                idx = random.randint(0, n-1)
                new_val = random.randint(1, SMALL_MAX)
                if new_val not in current_set or new_val == current_set[idx]:
                    neighbor[idx] = new_val
            elif move_type < 0.8:
                # Swap two elements with new random values (2-move)
                i1, i2 = random.sample(range(n), 2)
                v1 = random.randint(1, SMALL_MAX)
                v2 = random.randint(1, SMALL_MAX)
                if (v1 not in current_set or v1 == current_set[i1]) and \
                   (v2 not in current_set or v2 == current_set[i2]) and \
                   v1 != v2:
                    neighbor[i1] = v1
                    neighbor[i2] = v2
                else:
                    # Fallback to single replace if 2-move fails validity easily
                    idx = random.randint(0, n-1)
                    new_val = random.randint(1, SMALL_MAX)
                    if new_val not in current_set or new_val == current_set[idx]:
                        neighbor[idx] = new_val
            else:
                # Small perturbation
                idx = random.randint(0, n-1)
                delta = random.choice([-2, -1, 1, 2, -5, 5])
                new_val = current_set[idx] + delta
                if 1 <= new_val <= SMALL_MAX and new_val not in current_set:
                    neighbor[idx] = new_val
                else:
                    new_val = random.randint(1, SMALL_MAX)
                    if new_val not in current_set:
                        neighbor[idx] = new_val
                    else:
                        continue

            if len(set(neighbor)) != n:
                continue
                
            neighbor_obj = objective(neighbor)
            delta_obj = neighbor_obj - current_obj
            
            if delta_obj < 0 or random.random() < math.exp(-delta_obj / T):
                current_set = neighbor
                current_obj = neighbor_obj
                
                if current_obj < best_local_obj:
                    best_local_obj = current_obj
                    best_local_set = current_set[:]
            
            T = max(T * cooling_rate, T_min)
            
        # Polish with local search
        final_set, final_obj = local_search(best_local_set, max_iters=50)
        if final_obj < best_local_obj:
            best_local_set = final_set
            best_local_obj = final_obj
            
        return best_local_set, best_local_obj

    # Generate diverse seeds
    random.seed(42)
    seeds = []
    
    # 1. Arithmetic Progression
    seeds.append(list(range(1, n+1)))
    seeds.append(list(range(2, n+2)))
    
    # 2. "Clustered" sets (dense near 1, sparse elsewhere)
    for base in [1, 2, 3]:
        s = [base]
        while len(s) < n:
            # Add small increments
            cand = s[-1] + random.randint(1, 3)
            if cand <= SMALL_MAX and cand not in s:
                s.append(cand)
            else:
                cand = random.randint(1, SMALL_MAX)
                if cand not in s:
                    s.append(cand)
        seeds.append(s)
    
    # 3. "Mixed" - AP for first 10, random for next 10
    s = list(range(1, 11))
    while len(s) < n:
        cand = random.randint(10, SMALL_MAX)
        if cand not in s:
            s.append(cand)
    seeds.append(s)
    
    # 4. Random sets
    for _ in range(15):
        seeds.append(random.sample(range(1, SMALL_MAX+1), n))
    
    # 5. Structured patterns
    # Powers of 2 compressed
    powers = [1, 2, 4, 8, 16, 32]
    s = list(powers)
    while len(s) < n:
        cand = random.randint(30, SMALL_MAX)
        if cand not in s:
            s.append(cand)
    seeds.append(s)
    
    # Fibonacci-like
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    s = list(set(fib))
    while len(s) < n:
        cand = random.randint(50, SMALL_MAX)
        if cand not in s:
            s.append(cand)
    seeds.append(s)
    
    best_found_obj = float('inf')
    best_found_set = None

    # Run SA on each seed
    for seed in seeds:
        if time.time() > deadline - 10.0:  # Leave 10s for final polish
            break
        
        # Ensure seed is valid
        if len(set(seed)) != n or any(x < 1 or x > SMALL_MAX for x in seed):
            continue
        
        sa_set, sa_obj = simulated_annealing(seed, max_iters=4000)
        
        if sa_obj < best_found_obj:
            best_found_obj = sa_obj
            best_found_set = sa_set[:]
        
        # If we've found a very good solution, we can reduce iterations for remaining seeds
        # or break early if time is running out.
        
    # Final intensive polish on the best solution found
    if best_found_set is not None and time.time() < deadline - 3.0:
        # Multiple rounds of local search with increasing tolerance
        polished_set = best_found_set[:]
        polished_obj = best_found_obj
        
        for round_num in range(3):
            if time.time() > deadline - 1.0:
                break
            polished_set, polished_obj = local_search(polished_set, max_iters=150, deadline=deadline - 1.0)
        
        if polished_obj < best_found_obj:
            best_found_obj = polished_obj
            best_found_set = polished_set
    
    if best_found_set is None:
        best_found_set = list(range(1, n+1))
        best_found_obj = objective(best_found_set)
        
    # Scale up to fit 10000
    max_val = max(best_found_set)
    scale = 10000 // max_val
    if scale < 1:
        scale = 1
        
    final_set = [x * scale for x in best_found_set]
    
    # Verify final set
    if len(set(final_set)) != n or any(x > 10000 or x < 1 for x in final_set):
        # Fallback to simple AP scaled
        ap = list(range(1, n+1))
        max_ap = max(ap)
        sc = 10000 // max_ap
        final_set = [x * sc for x in ap]
        
    return final_set
