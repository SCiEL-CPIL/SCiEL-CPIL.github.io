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
        while np.any(uncovered) and (time.time() - t_start) < deadline - 20.0:
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

    # Phase 1: Get a good initial solution via multiple greedy starts
    best_code = None
    best_size = space_size + 1

    for seed in range(5):
        if (time.time() - t_start) > deadline - 40.0:
            break
        code = greedy_cover(seed)
        cov = build_cov(code)
        if not np.all(cov > 0):
            code, cov = greedy_repair(code, cov, deadline - 30.0)
        if len(code) < best_size:
            best_code = code[:]
            best_size = len(code)

    # Phase 2: Aggressive removal to minimize initial solution
    if best_code is not None:
        cur_code = best_code[:]
        cur_cov = build_cov(cur_code)
        changed = True
        while changed and (time.time() - t_start) < deadline - 30.0:
            changed = False
            for i in range(len(cur_code)):
                c = cur_code[i]
                if np.min(cur_cov[neighbors[c]]) >= 2:
                    cur_code.pop(i)
                    cur_cov[neighbors[c]] -= 1
                    changed = True
                    break
        if len(cur_code) < best_size:
            best_code = cur_code[:]
            best_size = len(cur_code)

    # Phase 3: Genetic Algorithm
    # Representation: fixed-size array of codeword indices (multiset allowed)
    # Fitness: -uncovered_count - overlap_penalty (lower is better)
    
    if best_code is None:
        best_code = greedy_cover(0)
        best_size = len(best_code)
    
    # GA Parameters
    pop_size = 60
    target_size = best_size  # We evolve sets of this size
    # Allow slightly larger sets in population to find better configurations
    ga_deadline = deadline - 5.0
    
    # Initialize population
    # Individual 0: best known
    # Others: mutated versions of best known + random greedy starts
    rng = np.random.default_rng(12345)
    
    population = []
    # Add best known (padded if necessary, though it should be <= target_size usually)
    # If best_size < target_size, we pad with duplicates or randoms? 
    # Better: evolve exactly best_size codewords. If we find a better one, we shrink.
    # Actually, let's evolve a fixed size K. 
    # Start with K = best_size. 
    # If best_size is too small to cover, fitness will be high (bad).
    # But we know best_size covers.
    
    # Let's allow the population to have size K = best_size.
    # Mutation: change a position to a random codeword.
    # Crossover: uniform mix of two parents.
    
    def fitness(ind):
        """Lower is better. ind is np.array of indices."""
        cov = np.zeros(space_size, dtype=np.int32)
        # Vectorized update is hard with variable ind, but ind size is small (~186)
        # space_size is 2187. neighbors is 2187 x 19.
        # cov[neighbors[ind]] += 1
        # This is a fancy indexing operation.
        idxs = ind.astype(np.int32)
        cov[neighbors[idxs]] += 1
        uncovered = np.sum(cov == 0)
        # Overlap penalty: sum of (cov-1) where cov>1
        overlap = np.sum(np.maximum(cov - 1, 0))
        # Weighting: Uncovered is critical. Overlap is secondary.
        # If uncovered > 0, we want to minimize it first.
        # Score = uncovered * 1000 + overlap
        return uncovered * 1000 + overlap

    # Initialize Population
    pop = []
    for i in range(pop_size):
        if i == 0:
            # Best known
            ind = np.array(best_code, dtype=np.int32)
            if len(ind) < target_size:
                # Pad with random
                pad = rng.integers(0, space_size, size=target_size - len(ind))
                ind = np.concatenate([ind, pad])
            elif len(ind) > target_size:
                ind = ind[:target_size]
            pop.append(ind)
        elif i < 5:
            # Mutated best known
            ind = np.array(best_code, dtype=np.int32)
            if len(ind) < target_size:
                pad = rng.integers(0, space_size, size=target_size - len(ind))
                ind = np.concatenate([ind, pad])
            elif len(ind) > target_size:
                ind = ind[:target_size]
            # Mutate a few positions
            num_mut = rng.integers(1, 5)
            mut_idx = rng.choice(target_size, size=num_mut, replace=False)
            new_vals = rng.integers(0, space_size, size=num_mut)
            ind[mut_idx] = new_vals
            pop.append(ind)
        else:
            # Random greedy start (truncated/padded to target_size)
            code = greedy_cover(i + 500)
            cov = build_cov(code)
            if not np.all(cov > 0):
                code, cov = greedy_repair(code, cov, deadline - 20.0)
            ind = np.array(code, dtype=np.int32)
            if len(ind) < target_size:
                pad = rng.integers(0, space_size, size=target_size - len(ind))
                ind = np.concatenate([ind, pad])
            elif len(ind) > target_size:
                # Keep best target_size? Just truncate for now, or random sample
                ind = ind[:target_size]
            pop.append(ind)
    
    # Evaluate initial population
    scores = np.array([fitness(ind) for ind in pop])
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_ind = pop[best_idx].copy()
    
    # GA Loop
    gen = 0
    while (time.time() - t_start) < ga_deadline:
        gen += 1
        new_pop = []
        
        # Elitism: keep best individual
        new_pop.append(best_ind.copy())
        
        # Generate offspring
        while len(new_pop) < pop_size:
            # Tournament Selection
            t_size = 3
            t_idx = rng.integers(0, pop_size, size=t_size)
            t_scores = scores[t_idx]
            winner = t_idx[np.argmin(t_scores)]
            parent1 = pop[winner]
            
            # Second parent: random or tournament
            if rng.random() < 0.5:
                parent2 = pop[rng.integers(0, pop_size)]
            else:
                t_idx2 = rng.integers(0, pop_size, size=t_size)
                t_scores2 = scores[t_idx2]
                winner2 = t_idx2[np.argmin(t_scores2)]
                parent2 = pop[winner2]
            
            # Crossover: Uniform
            mask = rng.random(target_size) < 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation
            # Probability of mutation per position
            mut_prob = 0.05 + 0.05 * (gen / 100.0) # Increasing mutation? Or constant
            # Let's use a fixed reasonable rate
            mut_prob = 0.1
            mut_mask = rng.random(target_size) < mut_prob
            if np.any(mut_mask):
                # Choose new values
                # To stay somewhat relevant, we could pick neighbors of current value?
                # But random in space is fine for global search
                new_vals = rng.integers(0, space_size, size=np.sum(mut_mask))
                child[mut_mask] = new_vals
            
            new_pop.append(child)
        
        pop = new_pop
        scores = np.array([fitness(ind) for ind in pop])
        
        # Update best
        curr_best_idx = np.argmin(scores)
        curr_best_score = scores[curr_best_idx]
        
        if curr_best_score < best_score:
            best_score = curr_best_score
            best_ind = pop[curr_best_idx].copy()
            # Check if we can shrink the solution
            # If uncovered == 0, we have a valid cover of size target_size.
            # We can try to remove redundant codewords to get a smaller valid code.
            if best_score < 1000: # uncovered == 0
                # Extract unique codewords
                unique_code = list(set(int(x) for x in best_ind))
                # Verify coverage of unique set
                # It might be that removing duplicates didn't break coverage
                # But we need to check if we can remove *other* codewords too
                # Use the removal logic on this unique set
                cur_code = unique_code
                cur_cov = build_cov(cur_code)
                if np.all(cur_cov > 0):
                    changed = True
                    while changed and (time.time() - t_start) < ga_deadline - 2.0:
                        changed = False
                        for i in range(len(cur_code)):
                            c = cur_code[i]
                            if np.min(cur_cov[neighbors[c]]) >= 2:
                                cur_code.pop(i)
                                cur_cov[neighbors[c]] -= 1
                                changed = True
                                break
                    if len(cur_code) < best_size:
                        best_code = cur_code[:]
                        best_size = len(cur_code)
                        # Update target_size for next generations?
                        # Yes, shrink the search space
                        target_size = best_size
                        # Resize population? 
                        # For simplicity, we can keep the GA running with the new target_size
                        # But we need to resize individuals.
                        # Let's just restart the GA loop with new target_size
                        # Or simply break and do a final local search
                        # Given time constraints, let's just update best and continue
                        # The fitness function will now penalize size implicitly? No, it's fixed size.
                        # If we reduce target_size, we need to resize pop.
                        # Let's just accept the improvement and continue searching for even smaller
                        # by reducing target_size further if we find another 0-uncovered solution.
                        # But since we are evolving fixed size, if we find a valid cover of size K,
                        # we set best_size=K. To find K-1, we need to evolve size K-1.
                        # So we should restart GA with new target_size.
                        
                        # Optimization: If we found a smaller valid code, restart GA with that size.
                        # This is a structural change.
                        # Let's break out of the current GA loop and restart with new params.
                        # We have time to spare.
                        
                        # Re-initialize population for new target_size
                        pop = []
                        for i in range(pop_size):
                            if i == 0:
                                ind = np.array(best_code, dtype=np.int32)
                                if len(ind) < target_size:
                                    pad = rng.integers(0, space_size, size=target_size - len(ind))
                                    ind = np.concatenate([ind, pad])
                                elif len(ind) > target_size:
                                    ind = ind[:target_size]
                                pop.append(ind)
                            else:
                                # Mutated best
                                ind = np.array(best_code, dtype=np.int32)
                                if len(ind) < target_size:
                                    pad = rng.integers(0, space_size, size=target_size - len(ind))
                                    ind = np.concatenate([ind, pad])
                                elif len(ind) > target_size:
                                    ind = ind[:target_size]
                                num_mut = rng.integers(1, max(2, target_size // 10))
                                if target_size > 0:
                                    mut_idx = rng.choice(target_size, size=min(num_mut, target_size), replace=False)
                                    new_vals = rng.integers(0, space_size, size=len(mut_idx))
                                    ind[mut_idx] = new_vals
                                pop.append(ind)
                        
                        scores = np.array([fitness(ind) for ind in pop])
                        best_idx = np.argmin(scores)
                        best_score = scores[best_idx]
                        best_ind = pop[best_idx].copy()
                        gen = 0 # Reset generation count for new size
                        continue
        
        # Stagnation check? 
        # If no improvement in last X generations, increase mutation or restart
        # For simplicity, we just keep going until time limit.
        
        # Small pause to check time
        if (time.time() - t_start) > ga_deadline:
            break

    # Phase 4: Final Local Search on the best GA result
    # The GA found a multiset of size target_size with low fitness.
    # If fitness < 1000, it's a valid cover.
    # Extract unique codewords and do final removal.
    
    # Re-evaluate best_ind to be sure
    # But we already tracked best_code and best_size.
    # Let's do one more aggressive removal pass on best_code.
    
    if best_code is not None:
        cur_code = best_code[:]
        cur_cov = build_cov(cur_code)
        changed = True
        while changed and (time.time() - t_start) < deadline - 1.0:
            changed = False
            for i in range(len(cur_code)):
                c = cur_code[i]
                if np.min(cur_cov[neighbors[c]]) >= 2:
                    cur_code.pop(i)
                    cur_cov[neighbors[c]] -= 1
                    changed = True
                    break
        if len(cur_code) < best_size:
            best_code = cur_code[:]
            best_size = len(cur_code)

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
