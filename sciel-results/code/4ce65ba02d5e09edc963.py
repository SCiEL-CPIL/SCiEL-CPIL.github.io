import time
import random
import math

def set_sizes(A):
    """
    Computes the sizes of the sumset A+A and product set A*A.
    A is a list of distinct positive integers.
    """
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
    """
    Returns the maximum of the sumset and product set sizes.
    """
    s, p = set_sizes(A)
    return max(s, p)

def local_search(current_set, max_iters=100, small_max=100):
    """
    Steepest descent local search to polish a candidate.
    Tries to replace each element with a value in [1, small_max] that improves the objective.
    """
    n = len(current_set)
    current_set = list(current_set)
    current_obj = objective(current_set)
    
    improved = True
    iter_count = 0
    while improved and iter_count < max_iters:
        improved = False
        iter_count += 1
        best_neighbor = None
        best_neighbor_obj = current_obj
        
        # For each position, try a few random replacements
        # To be more exhaustive but still fast, we can try all values in a subset
        # or just random samples. Given small_max=100, trying all 100 for 20 elements
        # is 2000 evaluations per iteration. 100 iterations = 200,000 evals.
        # Each eval is ~210 ops. 42M ops. Might be slow in Python if done naively.
        # Let's limit to random samples or a smaller range around current values.
        
        for i in range(n):
            old_val = current_set[i]
            # Try random replacements
            for _ in range(10):
                new_val = random.randint(1, small_max)
                if new_val == old_val or new_val in current_set:
                    continue
                neighbor = current_set[:]
                neighbor[i] = new_val
                # Check distinctness (already handled by 'new_val in current_set' check? 
                # No, if new_val is in current_set but at a different index, it's a duplicate.
                # 'new_val in current_set' checks if it exists anywhere.
                # So this ensures distinctness.)
                
                n_obj = objective(neighbor)
                if n_obj < best_neighbor_obj:
                    best_neighbor_obj = n_obj
                    best_neighbor = neighbor[:]
            
            # Also try nearby values
            for delta in [-2, -1, 1, 2]:
                new_val = old_val + delta
                if 1 <= new_val <= small_max and new_val != old_val and new_val not in current_set:
                    neighbor = current_set[:]
                    neighbor[i] = new_val
                    n_obj = objective(neighbor)
                    if n_obj < best_neighbor_obj:
                        best_neighbor_obj = n_obj
                        best_neighbor = neighbor[:]

        if best_neighbor is not None and best_neighbor_obj < current_obj:
            current_set = best_neighbor
            current_obj = best_neighbor_obj
            improved = True
            
    return current_set, current_obj

def simulated_annealing(start_set, max_iters=2000, small_max=100):
    """
    Simulated Annealing to find a good candidate in the small domain.
    """
    n = len(start_set)
    current_set = list(start_set)
    current_obj = objective(current_set)
    
    T = 20.0
    T_min = 0.1
    cooling_rate = 0.995
    
    best_local_set = current_set[:]
    best_local_obj = current_obj
    
    for i in range(max_iters):
        neighbor = current_set[:]
        move_type = random.random()
        
        if move_type < 0.6:
            # Replace one element
            idx = random.randint(0, n-1)
            new_val = random.randint(1, small_max)
            if new_val not in current_set or new_val == current_set[idx]:
                neighbor[idx] = new_val
        elif move_type < 0.8:
            # Swap two elements with new random values
            i1, i2 = random.sample(range(n), 2)
            v1 = random.randint(1, small_max)
            v2 = random.randint(1, small_max)
            if (v1 not in current_set or v1 == current_set[i1]) and \
               (v2 not in current_set or v2 == current_set[i2]) and \
               v1 != v2:
                neighbor[i1] = v1
                neighbor[i2] = v2
            else:
                # Fallback to single replace
                idx = random.randint(0, n-1)
                new_val = random.randint(1, small_max)
                if new_val not in current_set or new_val == current_set[idx]:
                    neighbor[idx] = new_val
        else:
            # Small perturbation
            idx = random.randint(0, n-1)
            delta = random.choice([-5, -2, -1, 1, 2, 5])
            new_val = current_set[idx] + delta
            if 1 <= new_val <= small_max and new_val not in current_set:
                neighbor[idx] = new_val
            else:
                new_val = random.randint(1, small_max)
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
    final_set, final_obj = local_search(best_local_set, max_iters=50, small_max=small_max)
    if final_obj < best_local_obj:
        best_local_set = final_set
        best_local_obj = final_obj
        
    return best_local_set, best_local_obj

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin under 90s
    
    n = 20
    SMALL_MAX = 100
    
    best_obj_global = float('inf')
    best_set_global = None

    # Generate diverse seeds
    random.seed(42)
    seeds = []
    
    # 1. Arithmetic Progression
    seeds.append(list(range(1, n+1)))
    seeds.append(list(range(2, n+2)))
    
    # 2. Random sets
    for _ in range(10):
        seeds.append(random.sample(range(1, SMALL_MAX+1), n))
        
    # 3. "Clustered" sets (dense near 1)
    base = 1
    s = [base]
    while len(s) < n:
        cand = s[-1] + random.randint(1, 3)
        if cand <= SMALL_MAX and cand not in s:
            s.append(cand)
        else:
            cand = random.randint(1, SMALL_MAX)
            if cand not in s:
                s.append(cand)
    seeds.append(s)
    
    # 4. "Mixed" - AP for first 10, random for next 10
    s = list(range(1, 11))
    while len(s) < n:
        cand = random.randint(10, SMALL_MAX)
        if cand not in s:
            s.append(cand)
    seeds.append(s)
    
    # 5. "Sparse" - Geometric-like (small ratio)
    s = [1]
    while len(s) < n:
        # Try to add next geometric term, but if it's too big, add random
        next_geometric = s[-1] * 2
        if next_geometric <= SMALL_MAX and next_geometric not in s:
            s.append(next_geometric)
        else:
            cand = random.randint(1, SMALL_MAX)
            if cand not in s:
                s.append(cand)
    seeds.append(s)

    best_found_obj = float('inf')
    best_found_set = None

    # Run SA on each seed
    for seed in seeds:
        if time.time() > deadline:
            break
            
        # Ensure seed is valid
        if len(set(seed)) != n or any(x < 1 or x > SMALL_MAX for x in seed):
            continue
            
        sa_set, sa_obj = simulated_annealing(seed, max_iters=1500, small_max=SMALL_MAX)
        
        if sa_obj < best_found_obj:
            best_found_obj = sa_obj
            best_found_set = sa_set[:]

    # If no valid set found (should not happen), fallback
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
    if len(set(final_set)) != n or any(x > 10000 for x in final_set):
        # Fallback to simple AP
        return list(range(1, n+1))
        
    return final_set
