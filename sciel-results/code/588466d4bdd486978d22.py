# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    
    Strategy:
    1. Compute forbidden differences F.
    2. Use bitmask-based representation.
    3. Multiple greedy initializations.
    4. Local search with:
       - Greedy fill
       - Randomized swaps and adds
       - 2-opt style moves
    5. Simulated annealing for escape from local optima.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 560  # Safety margin
    
    # 1. Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        val = (x * x) % m
        if val != 0:
            forbidden.add(val)
    
    # 2. Precompute conflict masks
    conflict_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in range(m):
            if i == j:
                continue
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Conflict degree for each element
    conflict_degree = [bin(conflict_mask[i]).count('1') for i in range(m)]
    
    def can_add_bitmask(c, mask):
        return (mask & conflict_mask[c]) == 0
    
    def popcount(x):
        return bin(x).count('1')
    
    def mask_to_list(mask):
        return [i for i in range(m) if (mask >> i) & 1]
    
    # Greedy initialization
    def greedy_init(order):
        mask = 0
        for cand in order:
            if can_add_bitmask(cand, mask):
                mask |= (1 << cand)
        return mask
    
    # Generate initial masks with diverse orderings
    orders = [
        list(range(m)),
        list(range(m-1, -1, -1)),
        sorted(range(m), key=lambda i: conflict_degree[i]),
        sorted(range(m), key=lambda i: -conflict_degree[i]),
        list(range(0, m, 2)),
        list(range(1, m, 2)),
    ]
    
    for seed in range(100):
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        orders.append(order)
    
    # Golden ratio shuffle
    for seed in range(30):
        order = list(range(m))
        rng = random.Random(seed)
        golden_ratio = (1 + 5**0.5) / 2
        for i in range(m - 1, 0, -1):
            j = int((i * golden_ratio) % (i + 1))
            order[i], order[j] = order[j], order[i]
        orders.append(order)
    
    best_init_mask = 0
    best_init_size = -1
    for order in orders:
        init_mask = greedy_init(order)
        size = popcount(init_mask)
        if size > best_init_size:
            best_init_size = size
            best_init_mask = init_mask
    
    if best_init_mask == 0:
        best_init_mask = 1
        best_init_size = 1
    
    global_best_mask = best_init_mask
    global_best_size = best_init_size
    
    # Local search with simulated annealing
    def local_search_with_sa(initial_mask, rng, time_limit):
        current_mask = initial_mask
        current_size = popcount(current_mask)
        
        # Greedy fill
        improved = True
        while improved:
            improved = False
            for c in range(m):
                if not (current_mask & (1 << c)):
                    if can_add_bitmask(c, current_mask):
                        current_mask |= (1 << c)
                        current_size += 1
                        improved = True
                        if current_size > best_local_size:
                            best_local_size = current_size
                            best_local_mask = current_mask
        
        best_local_mask = current_mask
        best_local_size = current_size
        
        # Simulated annealing parameters
        initial_temp = 1.0
        final_temp = 0.001
        cooling_rate = 0.9995
        temp = initial_temp
        
        last_time = time.time()
        while time.time() < time_limit:
            move_type = rng.random()
            
            if move_type < 0.4:
                # Add random element
                c = rng.randint(0, m - 1)
                if not (current_mask & (1 << c)):
                    if can_add_bitmask(c, current_mask):
                        new_mask = current_mask | (1 << c)
                        new_size = current_size + 1
                        delta = 1
                        if delta > 0 or rng.random() < np.exp(delta / temp):
                            current_mask = new_mask
                            current_size = new_size
                            if current_size > best_local_size:
                                best_local_size = current_size
                                best_local_mask = current_mask
            elif move_type < 0.75:
                # Swap: remove one, add one
                a = rng.randint(0, m - 1)
                b = rng.randint(0, m - 1)
                if (current_mask & (1 << a)) and not (current_mask & (1 << b)):
                    temp_mask = current_mask ^ (1 << a)
                    if (temp_mask & conflict_mask[b]) == 0:
                        new_mask = temp_mask | (1 << b)
                        delta = 0
                        if rng.random() < np.exp(delta / temp):
                            current_mask = new_mask
            else:
                # Remove one (to escape local optimum)
                a = rng.randint(0, m - 1)
                if (current_mask & (1 << a)):
                    new_mask = current_mask ^ (1 << a)
                    new_size = current_size - 1
                    delta = -1
                    if delta > 0 or rng.random() < np.exp(delta / temp):
                        current_mask = new_mask
                        current_size = new_size
            
            # Greedy fill after move
            if rng.random() < 0.15:
                improved = True
                while improved:
                    improved = False
                    for k in range(m):
                        if not (current_mask & (1 << k)):
                            if can_add_bitmask(k, current_mask):
                                current_mask |= (1 << k)
                                current_size += 1
                                improved = True
                if current_size > best_local_size:
                    best_local_size = current_size
                    best_local_mask = current_mask
            
            # Cool down
            temp *= cooling_rate
            if temp < final_temp:
                temp = final_temp
        
        return best_local_mask, best_local_size
    
    # Main search with restarts
    rng_main = random.Random(42)
    
    # Allocate time: 70% to first deep search, 30% to restarts
    total_time = deadline - start_time
    first_search_limit = start_time + total_time * 0.7
    
    # First deep search
    best_mask, best_size = local_search_with_sa(best_init_mask, rng_main, first_search_limit)
    if best_size > global_best_size:
        global_best_size = best_size
        global_best_mask = best_mask
    
    # Restart with perturbed versions
    remaining_limit = deadline - 30
    restart_count = 0
    while time.time() < remaining_limit and restart_count < 10:
        # Perturb the best solution
        perturbed = global_best_mask
        # Remove 3-5 random elements
        num_remove = rng_main.randint(3, 5)
        elements = mask_to_list(perturbed)
        rng_main.shuffle(elements)
        for i in range(min(num_remove, len(elements))):
            perturbed ^= (1 << elements[i])
        
        rng_sub = random.Random(1000 + restart_count)
        sub_limit = time.time() + min(30, remaining_limit - (time.time() - start_time))
        best_mask, best_size = local_search_with_sa(perturbed, rng_sub, sub_limit)
        if best_size > global_best_size:
            global_best_size = best_size
            global_best_mask = best_mask
        restart_count += 1
    
    # Convert best mask to array
    result_list = mask_to_list(global_best_mask)
    
    # Verify
    for i in range(len(result_list)):
        for j in range(i + 1, len(result_list)):
            a, b = result_list[i], result_list[j]
            d1 = (a - b) % m
            d2 = (b - a) % m
            if d1 in forbidden or d2 in forbidden:
                result_list = [0]
                break
    
    return np.array(result_list, dtype=np.int64)

# EVOLVE-BLOCK-END
