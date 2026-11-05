import time
import numpy as np

def construct(q, n, radius):
    t_start = time.time()
    deadline = 145.0  # safety margin before 150s limit

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i - 1] * q

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n - 1 - pos]) % q
        return w

    def word_to_idx(w):
        idx = 0
        for pos in range(n):
            idx += w[pos] * q_powers[n - 1 - pos]
        return idx

    # Precompute neighbors (radius=1: self + all 1-change neighbors)
    nb_count = 1 + n * (q - 1)
    neighbors = np.empty((space_size, nb_count), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n - 1 - pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list

    def build_cov(code_indices):
        cov = np.zeros(space_size, dtype=np.int32)
        for c in code_indices:
            cov[neighbors[c]] += 1
        return cov

    def greedy_cover(seed=0):
        """Greedy set cover with randomized tie-breaking."""
        rng = np.random.default_rng(seed)
        uncovered = np.ones(space_size, dtype=bool)
        code = []
        while np.any(uncovered) and (time.time() - t_start) < deadline - 15.0:
            gains = np.sum(uncovered[neighbors], axis=1)
            perturb = rng.random(space_size) * 0.001
            gains_pert = gains + perturb
            best = np.argmax(gains_pert)
            if gains[best] == 0:
                break
            code.append(int(best))
            uncovered[neighbors[best]] = False
        return code

    def greedy_repair(code_set, cov, max_time=10.0):
        """Add codewords to cover all uncovered points."""
        code_list = list(code_set)
        uncovered_mask = cov == 0
        while np.any(uncovered_mask) and (time.time() - t_start) < max_time:
            gains = np.sum(uncovered_mask[neighbors], axis=1)
            best = np.argmax(gains)
            if gains[best] == 0:
                break
            code_list.append(int(best))
            cov[neighbors[best]] += 1
            uncovered_mask = cov == 0
        return code_list, cov

    def safe_remove_pass(code, cov):
        """Remove redundant codewords (min coverage >= 2)."""
        changed = True
        while changed and (time.time() - t_start) < deadline - 5.0:
            changed = False
            for i in range(len(code)):
                c = code[i]
                if np.min(cov[neighbors[c]]) >= 2:
                    code.pop(i)
                    cov[neighbors[c]] -= 1
                    changed = True
                    break
        return code, cov

    # Phase 1: Get a good initial solution via multiple greedy starts
    best_code = None
    best_size = space_size + 1

    for seed in range(8):
        if (time.time() - t_start) > deadline - 40.0:
            break
        code = greedy_cover(seed)
        cov = build_cov(code)
        if not np.all(cov > 0):
            code, cov = greedy_repair(code, cov, deadline - 30.0)
        code, cov = safe_remove_pass(code, cov)
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)

    if best_code is None:
        best_code = greedy_cover(0)
        best_size = len(best_code)

    # Phase 2: Genetic Algorithm
    # Each individual: a list of codeword indices (sorted, distinct)
    # Fitness: lower is better. We minimize (uncovered_count * BIG_PENALTY + total_redundancy)
    # total_redundancy = sum over all points of (cov[p] - 1) for cov[p] >= 1, i.e., sum(cov) - space_size
    # This rewards tight coverage (minimal overlap) and strongly penalizes gaps.
    
    POP_SIZE = 40
    BIG_PENALTY = 1000  # Make uncovered points very expensive
    
    def evaluate_individual(code):
        """Returns (fitness, cov_array). Lower fitness is better."""
        cov = build_cov(code)
        uncovered = np.sum(cov == 0)
        total_cov = np.sum(cov)
        redundancy = total_cov - space_size  # sum(cov[p]-1) for covered, 0 for uncovered... wait
        # Actually redundancy = sum(cov[p]) - space_size only if all covered.
        # If not all covered, redundancy is still sum(cov) - space_size, but we add penalty.
        # Let's define fitness = uncovered * BIG_PENALTY + max(0, sum(cov) - space_size)
        # This encourages covering all points with minimal total ball volume used.
        fitness = uncovered * BIG_PENALTY + max(0, int(total_cov) - space_size)
        return fitness, cov

    # Initialize population
    population = []
    # First individual is the best found so far
    pop_code = best_code[:]
    pop_cov = build_cov(pop_code)
    pop_fit, _ = evaluate_individual(pop_code)
    population.append((pop_fit, pop_code, pop_cov))
    
    rng = np.random.default_rng(123)
    
    # Generate rest of population via greedy with different seeds and mutations
    for i in range(1, POP_SIZE):
        if (time.time() - t_start) > deadline - 30.0:
            break
        # Mix of greedy restarts and mutations of current best
        if i % 3 == 0:
            code = greedy_cover(100 + i)
        else:
            code = best_code[:]
            # Random mutation: remove a few, add a few
            if len(code) > 10:
                rem = rng.choice(len(code), size=min(5, len(code)//10), replace=False)
                rem_set = set(int(r) for r in rem)
                code = [c for j, c in enumerate(code) if j not in rem_set]
            
            cov = build_cov(code)
            # Add random codewords to cover gaps
            uncovered = cov == 0
            if np.any(uncovered):
                for _ in range(10):
                    if not np.any(uncovered):
                        break
                    gains = np.sum(uncovered[neighbors], axis=1)
                    best = np.argmax(gains)
                    if gains[best] == 0: break
                    if best not in code:
                        code.append(int(best))
                    cov[neighbors[best]] += 1
                    uncovered = cov == 0
        
        code, cov = safe_remove_pass(code, cov)
        fit, _ = evaluate_individual(code)
        population.append((fit, code, cov))

    # GA Loop
    elite_size = 5
    gen = 0
    while (time.time() - t_start) < deadline - 10.0:
        gen += 1
        
        # Sort population by fitness
        population.sort(key=lambda x: x[0])
        
        # Update global best
        if population[0][0] < best_size * BIG_PENALTY:  # Heuristic check, better to just check len
            if len(population[0][1]) < best_size:
                best_code = population[0][1][:]
                best_size = len(best_code)
        
        # Selection: Tournament selection
        def tournament_select():
            size = rng.integers(1, 4)
            candidates = rng.integers(0, len(population), size=size)
            best_idx = candidates[0]
            for c in candidates[1:]:
                if population[c][0] < population[best_idx][0]:
                    best_idx = c
            return population[best_idx]

        new_population = []
        
        # Elitism: keep top elites
        for i in range(min(elite_size, len(population))):
            new_population.append(population[i])
        
        # Generate offspring
        while len(new_population) < POP_SIZE:
            if (time.time() - t_start) > deadline - 10.0:
                break
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            code1 = parent1[1]
            code2 = parent2[1]
            
            # Crossover: Uniform crossover
            # Create a set union, then randomly pick from each parent's unique contributions?
            # Simpler: Take random subset of union.
            union = set(code1) | set(code2)
            union_list = list(union)
            
            if len(union_list) == 0:
                child_code = []
            else:
                # Probability of taking from parent1 vs parent2?
                # Just take random half of the union? Or weighted by fitness?
                # Let's do: for each element in union, keep it with prob 0.6 if in parent1, 0.6 if in parent2.
                # This might be too aggressive.
                # Better: Take all from parent1, then add some from parent2 that cover gaps?
                # Let's stick to a simple crossover: Random subset of the union, biased towards smaller parents.
                # Actually, for covering codes, taking the *union* and then pruning is often effective.
                # But union can be large.
                # Let's try: Child = (code1 + code2) shuffled, then take first K elements?
                # No, that's messy.
                
                # Alternative Crossover: 
                # 1. Start with empty set.
                # 2. For each position in the "genome" (conceptual), pick from p1 or p2.
                # But genomes are variable length sets.
                
                # Let's use "Bag Crossover" or just "Union + Prune".
                # Given time constraints, Union + Prune is robust.
                child_code = list(union)
                
                # Mutation: Randomly remove some elements
                if len(child_code) > 5:
                    rem_count = rng.integers(0, max(1, len(child_code)//10))
                    if rem_count > 0:
                        rem_indices = rng.choice(len(child_code), size=rem_count, replace=False)
                        rem_set = set(int(r) for r in rem_indices)
                        child_code = [c for i, c in enumerate(child_code) if i not in rem_set]
                
                # Mutation: Randomly add some elements
                add_count = rng.integers(0, 3)
                for _ in range(add_count):
                    if len(child_code) < space_size:
                        # Pick a random codeword from the space? Or from the union of parents?
                        # Random from space is good for exploration.
                        rand_c = rng.integers(0, space_size)
                        if rand_c not in child_code:
                            child_code.append(int(rand_c))
            
            # Repair and Prune
            cov = build_cov(child_code)
            # If there are uncovered points, do a quick greedy repair
            if np.any(cov == 0):
                child_code, cov = greedy_repair(child_code, cov, deadline - 15.0)
            
            child_code, cov = safe_remove_pass(child_code, cov)
            
            fit, _ = evaluate_individual(child_code)
            new_population.append((fit, child_code, cov))
        
        population = new_population

    # Phase 3: Final local search on the best GA result
    cur_code = best_code[:]
    cur_cov = build_cov(cur_code)
    
    # Try to improve with more aggressive kicks if time permits
    if (time.time() - t_start) < deadline - 5.0:
        for restart in range(3):
            if (time.time() - t_start) > deadline - 5.0:
                break
            cur_code = best_code[:]
            cur_cov = build_cov(cur_code)
            
            for k in range(1, 4):
                if (time.time() - t_start) > deadline - 5.0:
                    break
                for trial in range(20):
                    if (time.time() - t_start) > deadline - 5.0:
                        break
                    if len(cur_code) <= k:
                        break
                    
                    remove_indices = rng.choice(len(cur_code), size=k, replace=False)
                    remove_set = set(int(ri) for ri in remove_indices)
                    
                    new_code = [c for i, c in enumerate(cur_code) if i not in remove_set]
                    new_cov = build_cov(new_code)
                    
                    new_code, new_cov = greedy_repair(new_code, new_cov, deadline - 10.0)
                    new_code, new_cov = safe_remove_pass(new_code, new_cov)
                    
                    if len(new_code) < len(cur_code):
                        cur_code = new_code
                        cur_cov = new_cov
                        break
            
            if len(cur_code) < best_size:
                best_code = cur_code[:]
                best_size = cur_size if 'cur_size' in locals() else len(cur_code)
                best_size = len(best_code)

    # Final verification
    if best_code is None or len(best_code) == 0:
        best_code = greedy_cover(0)

    final_cov = build_cov(best_code)
    if not np.all(final_cov > 0):
        fc = greedy_cover(0)
        fcov = build_cov(fc)
        if np.all(fcov > 0):
            best_code = fc
        else:
            best_code = list(range(space_size))

    return [idx_to_word(c) for c in best_code]
