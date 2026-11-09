<<<<<<< SEARCH
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill

    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    MAX_VAL = 128

    def calc_reps(A_list):
        """Compute representation counts for sums in [N_LOW, N_HIGH]."""
        reps = [0] * RANGE_SIZE
        n = len(A_list)
        for i in range(n):
            ai = A_list[i]
            for j in range(i, n):
                s = ai + A_list[j]
                if N_LOW <= s <= N_HIGH:
                    reps[s - N_LOW] += 1
        return reps

    def is_covered(reps):
        return all(r > 0 for r in reps)

    def get_peak(reps):
        return max(reps)

    def swap_reps(A_list, old_idx, new_val):
        """
        Given A_list and index old_idx, compute new reps after replacing
        A_list[old_idx] with new_val. Uses incremental update: O(n) instead of O(n²).
        """
        n = len(A_list)
        old_val = A_list[old_idx]
        if old_val == new_val:
            return calc_reps(A_list)

        # Start from current reps
        reps = calc_reps(A_list)  # We'll do full calc but could optimize further

        # Actually, let's do incremental:
        # Remove contributions of old_val, add contributions of new_val
        reps = [0] * RANGE_SIZE
        for i in range(n):
            ai = A_list[i]
            for j in range(i, n):
                s = ai + A_list[j]
                if N_LOW <= s <= N_HIGH:
                    reps[s - N_LOW] += 1

        # That's still O(n²). Let me do proper incremental.
        # Actually for n~50-80, O(n²) = ~2500-6400 ops per eval, which is fast.
        # The bottleneck is the number of iterations, not each eval.
        return reps

    def fast_reps_from_set(A_set):
        """Compute reps from a set."""
        A_list = list(A_set)
        return calc_reps(A_list)

    def try_local_search(A_init, time_limit):
        """
        Run local search starting from A_init.
        Uses swap moves with incremental evaluation.
        Returns (best_A, best_peak).
        """
        A = list(A_init)
        reps = calc_reps(A)
        if not is_covered(reps):
            return None, float('inf')

        peak = get_peak(reps)
        best_A = list(A)
        best_peak = peak

        # Simulated annealing with swap moves
        T = 5.0
        T_min = 0.05
        cooling = 0.998
        n_steps = 0
        no_improve = 0
        max_no_improve = 5000

        while time.time() < time_limit and T > T_min:
            n_steps += 1
            if no_improve > max_no_improve:
                break

            n = len(A)
            if n < 2:
                break

            # Pick random index to replace
            i = random.randint(0, n - 1)
            old_val = A[i]

            # Pick random new value
            new_val = random.randint(0, MAX_VAL)
            if new_val == old_val:
                continue
            # Check if new_val already in A (avoid duplicates in set representation)
            if new_val in A:
                continue

            # Compute new reps
            A[i] = new_val
            new_reps = calc_reps(A)

            if not is_covered(new_reps):
                A[i] = old_val
                T *= cooling
                continue

            new_peak = get_peak(new_reps)
            delta = new_peak - peak

            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.3
            else:
                if T > 0.01:
                    accept = random.random() < math.exp(-delta / T)

            if accept:
                peak = new_peak
                reps = new_reps
                no_improve = 0
                if peak < best_peak:
                    best_peak = peak
                    best_A = list(A)
            else:
                A[i] = old_val
                no_improve += 1

            T *= cooling

        # Try to shrink: remove elements that don't break coverage
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(len(A) - 1, -1, -1):
                if time.time() > time_limit:
                    break
                saved = A.pop(i)
                new_reps = calc_reps(A)
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    if new_peak <= best_peak:
                        best_A = list(A)
                        best_peak = new_peak
                        peak = new_peak
                        reps = new_reps
                        improved = True
                    else:
                        A.append(saved)
                else:
                    A.append(saved)

        return best_A, best_peak

    # Generate diverse initial candidates
    def generate_initial_candidates():
        """Generate several diverse initial covering sets."""
        candidates = []

        # 1. Symmetric set: pairs (x, 128-x) for selected x
        # This naturally covers sums near 128
        for num_pairs in [16, 20, 24, 28, 32]:
            A = set()
            # Always include 0 and 128
            A.add(0)
            A.add(128)
            # Add symmetric pairs
            available = list(range(1, 64))
            random.shuffle(available)
            for x in available[:num_pairs]:
                A.add(x)
                A.add(128 - x)
            candidates.append(A)

        # 2. Arithmetic progression based
        for start in range(0, 20):
            for step in [2, 3, 4, 5, 6]:
                A = set()
                v = start
                while v <= 128:
                    A.add(v)
                    v += step
                if len(A) >= 10:
                    candidates.append(A)

        # 3. Random sets of various sizes
        for size in [30, 40, 50, 60, 70]:
            A = set(random.sample(range(0, 129), size))
            candidates.append(A)

        # 4. Structured: multiples + offsets
        for base in [8, 10, 12, 16]:
            A = set()
            for m in range(0, 129, base):
                A.add(m)
            # Add some offsets
            offsets = random.sample(range(1, base), min(3, base - 1))
            for m in range(0, 129, base):
                for o in offsets:
                    if m + o <= 128:
                        A.add(m + o)
            candidates.append(A)

        return candidates

    def ensure_coverage(A):
        """Add elements to A to ensure coverage of [128,256]."""
        A = set(A)
        reps = calc_reps(list(A))
        while not is_covered(reps) and len(A) < 129:
            missing = [N_LOW + i for i, r in enumerate(reps) if r == 0]
            if not missing:
                break
            S = random.choice(missing)
            # Find a y such that y + z = S for some z in A, or y not in A
            candidates = []
            for z in A:
                y = S - z
                if 0 <= y <= MAX_VAL and y not in A:
                    candidates.append(y)
            if candidates:
                A.add(random.choice(candidates))
            else:
                # Just add a random element
                non_A = [v for v in range(MAX_VAL + 1) if v not in A]
                if non_A:
                    A.add(random.choice(non_A))
                else:
                    break
            reps = calc_reps(list(A))
        return A

    # Main search: multiple restarts with diverse inits
    best_A_overall = None
    best_peak_overall = float('inf')

    # Time budget per restart
    num_restarts = 8
    time_per_restart = (deadline - time.time()) / num_restarts

    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break

        # Generate candidates and pick a few good ones
        cands = generate_initial_candidates()
        random.shuffle(cands)

        # Try up to 3 candidates per restart
        tried = 0
        for cand in cands:
            if time.time() > deadline - 2:
                break
            if tried >= 3:
                break

            # Ensure coverage
            A_covered = ensure_coverage(cand)
            reps = calc_reps(list(A_covered))
            if not is_covered(reps):
                continue

            peak = get_peak(reps)
            if peak < best_peak_overall:
                best_peak_overall = peak
                best_A_overall = list(A_covered)

            # Run local search
            remaining = min(deadline - time.time() - 1, time_per_restart / 3)
            if remaining < 0.5:
                break

            result_A, result_peak = try_local_search(list(A_covered), time.time() + remaining)
            if result_A is not None and result_peak < best_peak_overall:
                best_peak_overall = result_peak
                best_A_overall = result_A

            tried += 1

    # Final: try to improve best_A_overall with more time
    if best_A_overall is not None and time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        result_A, result_peak = try_local_search(best_A_overall, time.time() + remaining)
        if result_A is not None and result_peak < best_peak_overall:
            best_peak_overall = result_peak
            best_A_overall = result_A

    if best_A_overall is None:
        # Fallback: guaranteed covering set
        return list(range(64, 129))

    return best_A_overall

if __name__ == "__main__":
    t0 = time.time()
    A = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, |A|={len(A)}")
    reps = [0]*129
    A_list = list(A)
    for i in range(len(A_list)):
        for j in range(i, len(A_list)):
            s = A_list[i] + A_list[j]
            if 128 <= s <= 256:
                reps[s-128] += 1
    print(f"Covered: {all(r>0 for r in reps)}, Peak: {max(reps)}")
=======
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill

    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1  # 129
    MAX_VAL = 128

    def calc_reps(A_list):
        """Compute representation counts for sums in [N_LOW, N_HIGH]."""
        reps = [0] * RANGE_SIZE
        n = len(A_list)
        for i in range(n):
            ai = A_list[i]
            for j in range(i, n):
                s = ai + A_list[j]
                if N_LOW <= s <= N_HIGH:
                    reps[s - N_LOW] += 1
        return reps

    def is_covered(reps):
        return all(r > 0 for r in reps)

    def get_peak(reps):
        return max(reps)

    def get_fitness(A_list):
        """Fitness: negative peak (lower is better), penalty if not covered."""
        reps = calc_reps(A_list)
        if not is_covered(reps):
            return float('inf')
        return get_peak(reps)

    def generate_diverse_initial():
        """Generate a diverse initial population."""
        pop = []
        
        # Symmetric sets
        for num_pairs in [12, 16, 20, 24, 28, 32]:
            A = set()
            A.add(0)
            A.add(128)
            available = list(range(1, 64))
            random.shuffle(available)
            for x in available[:num_pairs]:
                A.add(x)
                A.add(128 - x)
            pop.append(list(A))
        
        # Arithmetic progressions
        for step in [2, 3, 4, 5, 6, 8]:
            for start in range(0, step):
                A = set()
                v = start
                while v <= 128:
                    A.add(v)
                    v += step
                if len(A) >= 10:
                    pop.append(list(A))
        
        # Random sets
        for size in [25, 35, 45, 55, 65, 75]:
            A = set(random.sample(range(0, 129), size))
            pop.append(list(A))
        
        # Structured: multiples + offsets
        for base in [6, 8, 10, 12, 16]:
            A = set()
            for m in range(0, 129, base):
                A.add(m)
            offsets = random.sample(range(1, base), min(3, base - 1))
            for m in range(0, 129, base):
                for o in offsets:
                    if m + o <= 128:
                        A.add(m + o)
            pop.append(list(A))
        
        return pop

    def ensure_coverage(A):
        """Add elements to A to ensure coverage of [128,256]."""
        A = set(A)
        reps = calc_reps(list(A))
        while not is_covered(reps) and len(A) < 129:
            missing = [N_LOW + i for i, r in enumerate(reps) if r == 0]
            if not missing:
                break
            S = random.choice(missing)
            candidates = []
            for z in A:
                y = S - z
                if 0 <= y <= MAX_VAL and y not in A:
                    candidates.append(y)
            if candidates:
                A.add(random.choice(candidates))
            else:
                non_A = [v for v in range(MAX_VAL + 1) if v not in A]
                if non_A:
                    A.add(random.choice(non_A))
                else:
                    break
            reps = calc_reps(list(A))
        return list(A)

    def local_search(A, time_limit):
        """Local search with swap moves."""
        A = list(A)
        reps = calc_reps(A)
        if not is_covered(reps):
            return A, float('inf')
        
        peak = get_peak(reps)
        best_A = list(A)
        best_peak = peak
        
        T = 3.0
        T_min = 0.05
        cooling = 0.999
        no_improve = 0
        max_no_improve = 3000
        
        while time.time() < time_limit and T > T_min:
            if no_improve > max_no_improve:
                break
            
            n = len(A)
            if n < 2:
                break
            
            i = random.randint(0, n - 1)
            old_val = A[i]
            new_val = random.randint(0, MAX_VAL)
            if new_val == old_val or new_val in A:
                continue
            
            A[i] = new_val
            new_reps = calc_reps(A)
            
            if not is_covered(new_reps):
                A[i] = old_val
                T *= cooling
                continue
            
            new_peak = get_peak(new_reps)
            delta = new_peak - peak
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                accept = random.random() < 0.3
            else:
                if T > 0.01:
                    accept = random.random() < math.exp(-delta / T)
            
            if accept:
                peak = new_peak
                reps = new_reps
                no_improve = 0
                if peak < best_peak:
                    best_peak = peak
                    best_A = list(A)
            else:
                A[i] = old_val
                no_improve += 1
            
            T *= cooling
        
        # Try to shrink
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(len(A) - 1, -1, -1):
                if time.time() > time_limit:
                    break
                saved = A.pop(i)
                new_reps = calc_reps(A)
                if is_covered(new_reps):
                    new_peak = get_peak(new_reps)
                    if new_peak <= best_peak:
                        best_A = list(A)
                        best_peak = new_peak
                        peak = new_peak
                        reps = new_reps
                        improved = True
                    else:
                        A.append(saved)
                else:
                    A.append(saved)
        
        return best_A, best_peak

    def genetic_algorithm(pop_size, time_limit):
        """Genetic algorithm for optimizing A."""
        # Initialize population
        population = []
        init_candidates = generate_diverse_initial()
        random.shuffle(init_candidates)
        
        for cand in init_candidates[:pop_size]:
            A_covered = ensure_coverage(cand)
            fitness = get_fitness(A_covered)
            population.append((A_covered, fitness))
        
        # Sort by fitness
        population.sort(key=lambda x: x[1])
        
        best_A = population[0][0]
        best_peak = population[0][1]
        
        # GA loop
        mutation_rate = 0.3
        crossover_rate = 0.7
        elite_size = 5
        
        while time.time() < time_limit:
            new_pop = []
            
            # Elitism: keep best solutions
            for i in range(min(elite_size, len(population))):
                new_pop.append((list(population[i][0]), population[i][1]))
            
            # Generate offspring
            while len(new_pop) < pop_size and time.time() < time_limit:
                # Selection: tournament
                def select():
                    idx1 = random.randint(0, len(population) - 1)
                    idx2 = random.randint(0, len(population) - 1)
                    if population[idx1][1] <= population[idx2][1]:
                        return population[idx1][0]
                    else:
                        return population[idx2][0]
                
                parent1 = select()
                parent2 = select()
                
                # Crossover or mutation
                if random.random() < crossover_rate and len(parent1) > 0 and len(parent2) > 0:
                    # Uniform crossover
                    child = set()
                    for v in parent1:
                        if random.random() < 0.5:
                            child.add(v)
                    for v in parent2:
                        if random.random() < 0.5:
                            child.add(v)
                    child = list(child)
                else:
                    child = list(parent1)
                
                # Mutation
                if random.random() < mutation_rate and len(child) > 0:
                    # Swap a random element
                    if random.random() < 0.5:
                        idx = random.randint(0, len(child) - 1)
                        child[idx] = random.randint(0, MAX_VAL)
                    else:
                        # Add or remove
                        if random.random() < 0.5 and len(child) > 5:
                            idx = random.randint(0, len(child) - 1)
                            child.pop(idx)
                        elif len(child) < 128:
                            child.append(random.randint(0, MAX_VAL))
                
                # Ensure coverage
                child = ensure_coverage(child)
                fitness = get_fitness(child)
                
                new_pop.append((child, fitness))
            
            # Update best
            for A, fit in new_pop:
                if fit < best_peak:
                    best_peak = fit
                    best_A = list(A)
            
            # Replace population
            population = new_pop
            population.sort(key=lambda x: x[1])
        
        return best_A, best_peak

    # Main: run GA with multiple restarts
    best_A_overall = None
    best_peak_overall = float('inf')
    
    num_restarts = 4
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 2:
            break
        
        pop_size = 30
        result_A, result_peak = genetic_algorithm(pop_size, time.time() + time_per_restart)
        
        if result_peak < best_peak_overall:
            best_peak_overall = result_peak
            best_A_overall = list(result_A)
        
        # Final local search on best
        if best_A_overall is not None and time.time() < deadline - 2:
            remaining = min(deadline - time.time() - 1, 5)
            ls_A, ls_peak = local_search(best_A_overall, time.time() + remaining)
            if ls_peak < best_peak_overall:
                best_peak_overall = ls_peak
                best_A_overall = ls_A
    
    if best_A_overall is None:
        return list(range(64, 129))
    
    return best_A_overall

if __name__ == "__main__":
    t0 = time.time()
    A = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s, |A|={len(A)}")
    reps = [0]*129
    A_list = list(A)
    for i in range(len(A_list)):
        for j in range(i, len(A_list)):
            s = A_list[i] + A_list[j]
            if 128 <= s <= 256:
                reps[s-128] += 1
    print(f"Covered: {all(r>0 for r in reps)}, Peak: {max(reps)}")
>>>>>>> REPLACE
