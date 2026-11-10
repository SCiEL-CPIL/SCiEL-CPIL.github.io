import time
import random
import sys

N = 256

# EVOLVE-BLOCK-START

def _is_valid_solution(arr):
    """
    Verifies:
    1. All elements are distinct integers in [1, N].
    2. It is a Sidon set (all unordered pair sums a+b with a<=b are distinct).
    3. It is maximal (adding any missing element from [1, N] destroys the Sidon property).
    """
    if not arr:
        return False
    
    arr_set = set(arr)
    if len(arr_set) != len(arr):
        return False
    
    for x in arr:
        if x < 1 or x > N:
            return False
            
    # Compute all sums
    sums = set()
    for i in range(len(arr)):
        a = arr[i]
        for j in range(i, len(arr)):
            b = arr[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
            
    # Check maximality
    for x in range(1, N + 1):
        if x in arr_set:
            continue
        
        # Try to add x
        conflict = False
        for a in arr:
            if (x + a) in sums:
                conflict = True
                break
        if conflict:
            continue
            
        if (2 * x) in sums:
            continue
            
        # If no conflict, x can be added, so not maximal
        return False
        
    return True

def _calculate_fitness(mask):
    """
    mask: integer bitmask representing the set.
    Returns a tuple (is_valid, size, penalty) where lower penalty is better.
    We want to minimize size, but validity is paramount.
    """
    # Extract elements
    arr = []
    m = mask
    idx = 0
    while m:
        if m & 1:
            arr.append(idx + 1)
        m >>= 1
        idx += 1
    
    if not arr:
        return False, 0, 1000000

    arr_set = set(arr)
    if len(arr_set) != len(arr):
        return False, len(arr), 1000000

    # Check Sidon
    sums = set()
    conflicts = 0
    for i in range(len(arr)):
        a = arr[i]
        for j in range(i, len(arr)):
            b = arr[j]
            s = a + b
            if s in sums:
                conflicts += 1
            else:
                sums.add(s)
    
    if conflicts > 0:
        # Penalty for invalid Sidon set
        return False, len(arr), 1000000 + conflicts * 100 + len(arr)

    # Check Maximality
    unblocked = 0
    for x in range(1, N + 1):
        if x in arr_set:
            continue
        
        # Check if x is blocked
        blocked = False
        for a in arr:
            if (x + a) in sums:
                blocked = True
                break
        if not blocked and (2 * x) in sums:
            blocked = True
            
        if not blocked:
            unblocked += 1
            
    if unblocked > 0:
        # Penalty for non-maximal Sidon set
        return True, len(arr), 10000 + unblocked * 10 + len(arr)
        
    return True, len(arr), len(arr)

def _greedy_construct(order):
    """
    Greedily constructs a Sidon set by iterating through `order`.
    Returns the constructed set as a list.
    """
    selected = []
    sums = set()
    
    for x in order:
        conflict = False
        for a in selected:
            if (x + a) in sums:
                conflict = True
                break
        if conflict:
            continue
            
        if (2 * x) in sums:
            continue
            
        selected.append(x)
        for a in selected:
            sums.add(x + a)
        sums.add(2 * x)
        
    return selected

def _list_to_mask(arr):
    mask = 0
    for x in arr:
        mask |= (1 << (x - 1))
    return mask

def _mask_to_list(mask):
    arr = []
    m = mask
    idx = 0
    while m:
        if m & 1:
            arr.append(idx + 1)
        m >>= 1
        idx += 1
    return arr

def construct():
    start_time = time.time()
    # Safety margin: 55s limit (5s buffer for overhead)
    deadline = start_time + 55.0
    
    best_mask = 0
    best_fitness = (False, 0, 10000000) # (is_valid, size, penalty)
    
    rng = random.Random(42)
    
    # 1. Initialize with a good greedy solution
    # Try a few deterministic orders
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        s = _greedy_construct(order)
        mask = _list_to_mask(s)
        fitness = _calculate_fitness(mask)
        if fitness[2] < best_fitness[2]:
            best_fitness = fitness
            best_mask = mask
            
    # Also try a random greedy to get a diverse start
    for _ in range(5):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        s = _greedy_construct(order)
        mask = _list_to_mask(s)
        fitness = _calculate_fitness(mask)
        if fitness[2] < best_fitness[2]:
            best_fitness = fitness
            best_mask = mask
            
    # 2. Genetic Algorithm / Simulated Annealing Hybrid
    # Population of masks
    pop_size = 50
    population = [best_mask] * pop_size
    
    # Add some random variations
    for i in range(pop_size):
        # Randomly flip some bits in the best mask
        m = best_mask
        for _ in range(rng.randint(1, 5)):
            bit = rng.randint(0, N - 1)
            m ^= (1 << bit)
        population[i] = m
        
    # Sort population by fitness
    def sort_key(mask):
        return _calculate_fitness(mask)
        
    population = sorted(population, key=sort_key)
    
    current_temp = 100.0
    cooling_rate = 0.95
    
    while time.time() < deadline:
        # Update best
        pop_fitnesses = [sort_key(m) for m in population]
        if pop_fitnesses[0][2] < best_fitness[2]:
            best_fitness = pop_fitnesses[0]
            best_mask = population[0]
            
        # Check if we found a valid maximal set with small size
        # If best is valid and maximal, we might stop if size is low enough
        if best_fitness[0] and best_fitness[2] <= 13:
             break

        # Generate new population
        new_population = []
        
        # Elitism: keep top 10%
        elite_count = max(1, pop_size // 10)
        for i in range(elite_count):
            new_population.append(population[i])
            
        # Crossover and Mutation
        while len(new_population) < pop_size:
            # Select parents
            # Tournament selection
            p1 = rng.choice(population[:pop_size//2])
            p2 = rng.choice(population[:pop_size//2])
            
            # Uniform Crossover
            child = 0
            for i in range(N):
                if rng.random() < 0.5:
                    if p1 & (1 << i):
                        child |= (1 << i)
                else:
                    if p2 & (1 << i):
                        child |= (1 << i)
                        
            # Mutation: Flip 1-3 bits
            for _ in range(rng.randint(1, 3)):
                bit = rng.randint(0, N - 1)
                child ^= (1 << bit)
                
            new_population.append(child)
            
        # Evaluate new population
        population = new_population
        population = sorted(population, key=sort_key)
        
        # Cooling
        current_temp *= cooling_rate

    # Final verification and return
    best_arr = _mask_to_list(best_mask)
    
    if not _is_valid_solution(best_arr):
        # Fallback to a known valid construction if something went wrong
        # Just run one more greedy if fallback fails
        s = _greedy_construct(list(range(1, N+1)))
        return s
        
    return best_arr

# EVOLVE-BLOCK-END
