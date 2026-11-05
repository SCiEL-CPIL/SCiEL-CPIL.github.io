def construct():
    import random
    import time
    import math

    def compute_sizes(A):
        sums = set()
        prods = set()
        n = len(A)
        for i in range(n):
            ai = A[i]
            for j in range(i, n):
                aj = A[j]
                sums.add(ai + aj)
                prods.add(ai * aj)
                if i != j:
                    sums.add(aj + ai) # Redundant if using set, but logic check
                    prods.add(aj * ai) # Redundant
        # Wait, the loop `for i in range(n): for j in range(i, n)` covers all pairs (i,j) with i<=j.
        # Since addition and multiplication are commutative, A[i]+A[j] == A[j]+A[i].
        # So we just need to add ai+aj and ai*aj for i<=j.
        return len(sums), len(prods)

    def score(A):
        s, p = compute_sizes(A)
        return max(s, p)

    def simulated_annealing(initial_A, start_temp=100.0, end_temp=1.0, max_iter=20000, time_limit=10.0):
        current_A = initial_A[:]
        current_score = score(current_A)
        best_A = current_A[:]
        best_score = current_score
        
        start_time = time.time()
        rng = random.Random(42)
        
        # Pre-compute some stats or just run
        # We use a geometric cooling schedule
        # T_k = T_start * (T_end / T_start)^(k / max_iter)
        
        for i in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            # Calculate current temperature
            # Simpler: linear or exponential decay
            # Let's use exponential decay based on iteration count
            if max_iter > 0:
                alpha = i / max_iter
                temp = start_temp * (end_temp / start_temp) ** alpha
            else:
                temp = end_temp
            
            # Generate neighbor
            # 1. Swap two elements
            # 2. Replace one element
            move_type = rng.random()
            new_A = current_A[:]
            
            if move_type < 0.5:
                # Swap
                i1, i2 = rng.sample(range(20), 2)
                new_A[i1], new_A[i2] = new_A[i2], new_A[i1]
            else:
                # Replace
                i1 = rng.randint(0, 19)
                new_val = rng.randint(1, 10000)
                # Ensure distinctness
                if new_val in new_A:
                    # Find a random distinct value
                    while new_val in new_A:
                        new_val = rng.randint(1, 10000)
                new_A[i1] = new_val
            
            # Check validity (distinct, <= 10000)
            if len(set(new_A)) != 20 or any(x > 10000 or x < 1 for x in new_A):
                continue
                
            new_score = score(new_A)
            delta = new_score - current_score
            
            # Acceptance criterion
            if delta < 0:
                # Always accept improvement
                current_A = new_A
                current_score = new_score
                if current_score < best_score:
                    best_A = current_A[:]
                    best_score = current_score
            else:
                # Metropolis criterion
                # Probability of accepting worse solution
                if temp > 0:
                    prob = math.exp(-delta / temp)
                    if rng.random() < prob:
                        current_A = new_A
                        current_score = new_score
                # else: temp is 0, only accept improvements (handled above)
        
        return best_A, best_score

    # Generate diverse initial seeds
    seeds = []
    
    # 1. Arithmetic Progression
    seeds.append(list(range(1, 21)))
    
    # 2. Geometric Progression (scaled to fit)
    # We want 20 distinct ints <= 10000.
    # r=1.5 -> 1.5^19 ~ 2218. 1.6^19 ~ 6176. 1.7^19 ~ 16980 (too big).
    # Let's try r=1.6
    gp = []
    val = 1
    for _ in range(20):
        gp.append(val)
        val = int(val * 1.6) + 1
    # Ensure distinct and sorted
    gp = sorted(set(gp))
    if len(gp) < 20:
        # Fill with small numbers not in gp
        for i in range(1, 10001):
            if i not in gp:
                gp.append(i)
            if len(gp) == 20:
                break
    seeds.append(gp[:20])
    
    # 3. Mixed: First 10 small, next 10 larger
    seed3 = list(range(1, 11)) + [100 * i for i in range(1, 11)]
    seeds.append(seed3)
    
    # 4. Lattice based (2,3)
    lattice = set()
    for a in range(15):
        for b in range(15):
            v = (2 ** a) * (3 ** b)
            if v <= 10000:
                lattice.add(v)
    lattice_list = sorted(lattice)
    if len(lattice_list) >= 20:
        seeds.append(lattice_list[:20])
    
    # 5. Random
    random.seed(1)
    seeds.append(sorted(random.sample(range(1, 10001), 20)))
    
    best_overall_A = None
    best_overall_score = float('inf')
    
    # Run SA on each seed
    # We have ~60s. Let's allocate ~10s per seed.
    # 5 seeds * 10s = 50s. Safe.
    
    for seed in seeds:
        if len(seed) != 20 or len(set(seed)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in seed):
            continue
            
        # Run SA
        A, sc = simulated_annealing(seed, start_temp=50.0, end_temp=0.1, max_iter=10000, time_limit=10.0)
        
        if sc < best_overall_score:
            best_overall_score = sc
            best_overall_A = A[:]
            
    if best_overall_A is None:
        return list(range(1, 21))
        
    return sorted(best_overall_A)
