import time
import random
import math
import numpy as np

def compute_metrics(A):
    """
    Computes the maximum of the size of the sumset A+A and product set A*A.
    A is a list of distinct positive integers.
    Uses NumPy for vectorization.
    """
    arr = np.array(A, dtype=np.int64)
    
    # Sumset A+A
    sums = arr[:, None] + arr[None, :]
    sum_set_size = len(np.unique(sums))
    
    # Product set A*A
    prods = arr[:, None] * arr[None, :]
    prod_set_size = len(np.unique(prods))
    
    return max(sum_set_size, prod_set_size)

def generate_smooth_pool(max_val=10000, max_prime=31):
    """Generate a rich pool of smooth numbers."""
    candidates = set()
    # Small integers
    for i in range(1, 100):
        candidates.add(i)
    
    # Powers of small primes
    for base in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]:
        val = 1
        while val <= max_val:
            candidates.add(val)
            val *= base
    
    # Products of two small primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for p1 in primes:
        for p2 in primes:
            prod = p1 * p2
            if prod <= max_val:
                candidates.add(prod)
            # Products of three small primes
            for p3 in primes[:6]:
                prod3 = prod * p3
                if prod3 <= max_val:
                    candidates.add(prod3)
    
    # Filter
    pool = [x for x in candidates if 1 <= x <= max_val]
    random.shuffle(pool)
    return pool

def pareto_front(population_metrics):
    """
    population_metrics: list of (sum_size, prod_size)
    Returns list of (sum_size, prod_size) that are Pareto optimal.
    """
    n = len(population_metrics)
    is_dominated = [False] * n
    for i in range(n):
        si, pi = population_metrics[i]
        for j in range(n):
            if i == j:
                continue
            sj, pj = population_metrics[j]
            if (sj <= si and pj <= pi) and (sj < si or pj < pi):
                is_dominated[i] = True
                break
    return [population_metrics[i] for i in range(n) if not is_dominated[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Pre-generate pool
    pool = generate_smooth_pool(10000, 31)
    
    # Initial diverse seeds
    seeds = [
        list(range(1, 21)),
        [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36],
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 3, 5, 7, 9, 11, 13],
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67], # Primes (bad for product, good for sum diversity?)
        [1, 2, 4, 6, 8, 12, 16, 24, 36, 48, 72, 96, 144, 192, 288, 432, 648, 864, 1296, 1728] # Geometric-ish
    ]
    
    # Filter valid seeds
    valid_seeds = []
    for s in seeds:
        if len(set(s)) == 20 and all(1 <= x <= 10000 for x in s):
            valid_seeds.append(s)
    
    # Initialize population with seeds + random samples from pool
    population = []
    for s in valid_seeds:
        population.append(s[:])
    
    # Add random individuals
    while len(population) < 30:
        # Sample 20 distinct from pool
        if len(pool) >= 20:
            ind = random.sample(pool, 20)
        else:
            ind = list(range(1, 21))
        # Ensure distinct (sample guarantees this)
        population.append(ind)
    
    # Evaluate initial population
    pop_metrics = []
    for ind in population:
        m = compute_metrics(ind)
        # We need sum and prod separately for Pareto, but compute_metrics returns max.
        # Let's compute them separately for the Pareto front logic.
        arr = np.array(ind, dtype=np.int64)
        sums = arr[:, None] + arr[None, :]
        prods = arr[:, None] * arr[None, :]
        ss = len(np.unique(sums))
        ps = len(np.unique(prods))
        pop_metrics.append((ss, ps))
    
    best_score = min(max(ss, ps) for ss, ps in pop_metrics)
    best_idx = pop_metrics.index(next((m for m in pop_metrics if max(m) == best_score), pop_metrics[0]))
    best_A = population[best_idx][:]
    
    # Evolutionary Loop
    pop_size = len(population)
    elite_size = 5
    mutation_rate = 0.3
    crossover_prob = 0.5
    
    while time.time() < deadline:
        # 1. Selection: Keep elites based on scalar fitness (max(ss, ps))
        # Sort by max(ss, ps)
        sorted_indices = sorted(range(pop_size), key=lambda i: max(pop_metrics[i][0], pop_metrics[i][1]))
        elites = [population[i][:] for i in sorted_indices[:elite_size]]
        
        # 2. Generate new generation
        new_population = []
        new_metrics = []
        
        # Add elites
        for e in elites:
            new_population.append(e)
            # Re-evaluate or reuse? Reuse is faster if we track it, but let's just evaluate for simplicity in the loop below.
            # Actually, we have the metrics for the old population. Let's map them.
            # This is getting complex. Let's just re-evaluate the new population at the end of generation.
        
        # Fill rest of population
        target_pop_size = pop_size
        while len(new_population) < target_pop_size:
            # Choose parents
            # Tournament selection on Pareto front or scalar?
            # Let's use a mix: 50% from top scalar, 50% from Pareto front
            if random.random() < 0.5:
                # Scalar tournament
                i1 = sorted_indices[random.randint(0, 4)]
                i2 = sorted_indices[random.randint(0, 4)]
            else:
                # Pareto front
                pf = pareto_front(pop_metrics)
                if not pf:
                    pf = pop_metrics
                # Pick random from Pareto front indices
                pf_indices = [i for i, m in enumerate(pop_metrics) if m in pf]
                if not pf_indices:
                    pf_indices = [0]
                i1 = pf_indices[random.randint(0, len(pf_indices)-1)]
                i2 = pf_indices[random.randint(0, len(pf_indices)-1)]
            
            p1 = population[i1]
            p2 = population[i2]
            
            # Crossover
            if random.random() < crossover_prob and len(p1) == 20 and len(p2) == 20:
                # Uniform crossover
                child = []
                for k in range(20):
                    if random.random() < 0.5:
                        child.append(p1[k])
                    else:
                        child.append(p2[k])
                # Resolve duplicates
                seen = set()
                unique_child = []
                for v in child:
                    if v not in seen:
                        seen.add(v)
                        unique_child.append(v)
                # Fill missing
                if len(unique_child) < 20:
                    # Sample from pool to fill
                    missing_count = 20 - len(unique_child)
                    available = [x for x in pool if x not in seen]
                    if len(available) >= missing_count:
                        fillers = random.sample(available, missing_count)
                        unique_child.extend(fillers)
                    else:
                        # Fallback
                        for x in range(1, 100):
                            if x not in seen and len(unique_child) < 20:
                                unique_child.append(x)
                                seen.add(x)
                child = unique_child[:20]
            else:
                child = p1[:]
            
            # Mutation
            # Random replacement
            num_mutations = int(random.gauss(mutation_rate * 20, 2))
            num_mutations = max(0, min(5, num_mutations))
            
            for _ in range(num_mutations):
                idx = random.randint(0, 19)
                # Replace with random from pool or neighbor
                if random.random() < 0.7:
                    new_val = random.choice(pool)
                else:
                    old_val = child[idx]
                    delta = random.randint(-10, 10)
                    new_val = old_val + delta
                    if new_val < 1: new_val = 1
                    if new_val > 10000: new_val = 10000
                
                # Ensure distinctness
                child_set = set(child)
                child_set.discard(child[idx])
                if new_val in child_set:
                    # Try a few times to find a unique value
                    found = False
                    for _ in range(5):
                        alt = random.choice(pool)
                        if alt not in child_set:
                            new_val = alt
                            found = True
                            break
                    if not found:
                        continue # Skip mutation if can't find unique
                
                child[idx] = new_val
            
            new_population.append(child)
        
        # Evaluate new population
        # This is the bottleneck. 30 individuals * 20^2 ops is fast, but unique is O(N log N).
        # 30 * 400 = 12000 ops. Very fast.
        for ind in new_population:
            arr = np.array(ind, dtype=np.int64)
            sums = arr[:, None] + arr[None, :]
            prods = arr[:, None] * arr[None, :]
            ss = len(np.unique(sums))
            ps = len(np.unique(prods))
            new_metrics.append((ss, ps))
            score = max(ss, ps)
            if score < best_score:
                best_score = score
                best_A = ind[:]
        
        population = new_population
        pop_metrics = new_metrics
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Final Polish: Local search on best_A
    # Use the remaining time for a quick local search on the best individual
    A = best_A[:]
    current_score = best_score
    
    # Simple local search: try replacing one element with a neighbor from pool
    # Sort pool by "smoothness" or just random
    # We have 5 seconds left.
    
    # To speed up local search, we can pre-compute some things? No, just run it.
    # Limit iterations.
    max_iters = 500
    iter_count = 0
    while iter_count < max_iters and time.time() < deadline - 0.5:
        iter_count += 1
        idx = random.randint(0, 19)
        old_val = A[idx]
        
        # Try 5 random candidates
        best_cand = None
        best_cand_score = current_score
        
        candidates = random.sample(pool, 5)
        # Add neighbors
        for d in range(-5, 6):
            nb = old_val + d
            if 1 <= nb <= 10000:
                candidates.append(nb)
        
        A_set = set(A)
        A_set.discard(old_val)
        
        for c in candidates:
            if c in A_set or c == old_val:
                continue
            if c < 1 or c > 10000:
                continue
            
            A[idx] = c
            s = compute_metrics(A)
            
            if s < best_cand_score:
                best_cand_score = s
                best_cand = c
        
        if best_cand is not None:
            A[idx] = best_cand
            current_score = best_cand_score
            if current_score < best_score:
                best_score = current_score
                best_A = A[:]
        else:
            A[idx] = old_val
            # If stuck, break early? Or continue.
            # Let's continue to explore.
    
    return best_A

# Quick test to ensure it runs
if __name__ == "__main__":
    res = construct()
    print(res)
    print(compute_metrics(res))
