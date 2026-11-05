import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0

    def get_direct_sum():
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [[0] * n]

    if q != 3 or n != 6 or radius != 1:
        return get_direct_sum()

    total = 729
    powers = [1, 3, 9, 27, 81, 243]
    coords = [[(i // p) % 3 for p in powers] for i in range(total)]

    # Build neighbor list (including self)
    nbr = np.empty((total, 13), dtype=np.int32)
    for i in range(total):
        nbr[i, 0] = i
        k = 1
        for p in range(6):
            base = (i // powers[p]) % 3
            for alt in range(3):
                if alt != base:
                    nbr[i, k] = i + (alt - base) * powers[p]
                    k += 1

    # ---- Phase 1: Greedy set cover ----
    uncovered = np.ones(total, dtype=np.int8)
    code = []
    while uncovered.sum() > 0:
        if time.time() > deadline:
            break
        counts = np.sum(uncovered[nbr], axis=1)
        best = int(np.argmax(counts))
        if counts[best] == 0:
            break
        code.append(best)
        uncovered[nbr[best]] = 0

    def try_reduce(cw):
        if not cw:
            return cw
        cov = np.zeros(total, dtype=np.int32)
        for idx in cw:
            cov[nbr[idx]] += 1
        changed = True
        while changed:
            changed = False
            for i in range(len(cw) - 1, -1, -1):
                idx = cw[i]
                if np.min(cov[nbr[idx]]) >= 2:
                    cov[nbr[idx]] -= 1
                    cw.pop(i)
                    changed = True
                    break
        return cw

    code = try_reduce(code)
    best_code = code[:]
    best_size = len(code)

    # ---- Phase 2: Local search (remove & repair) with incremental coverage ----
    rng = np.random.default_rng(42)
    no_improve = 0
    max_no_improve = 5000

    # Build coverage incrementally
    cov = np.zeros(total, dtype=np.int32)
    for idx in code:
        cov[nbr[idx]] += 1
    in_code = np.zeros(total, dtype=np.int8)
    for idx in code:
        in_code[idx] = 1

    while time.time() < deadline - 3.0 and no_improve < max_no_improve:
        if not code:
            break

        i = int(rng.integers(len(code)))
        c_remove = code[i]
        code.pop(i)
        in_code[c_remove] = 0
        cov[nbr[c_remove]] -= 1

        uncovered_mask = (cov == 0)
        n_uncovered = int(np.count_nonzero(uncovered_mask))

        if n_uncovered == 0:
            # Try to remove more
            changed = True
            while changed:
                changed = False
                for j in range(len(code) - 1, -1, -1):
                    idx = code[j]
                    if np.min(cov[nbr[idx]]) >= 2:
                        cov[nbr[idx]] -= 1
                        in_code[idx] = 0
                        code.pop(j)
                        changed = True
                        break
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                no_improve = 0
            else:
                no_improve += 1
            continue

        # Repair: find best replacement among candidates
        # Use only words not already in code, weighted by uncovered coverage
        counts_add = np.sum(uncovered_mask[nbr], axis=1)
        # Mask out words already in code
        counts_add[in_code == 1] = -1
        best_add = int(np.argmax(counts_add))

        if counts_add[best_add] <= 0:
            # Can't repair; revert
            code.append(c_remove)
            in_code[c_remove] = 1
            cov[nbr[c_remove]] += 1
            no_improve += 1
        else:
            code.append(best_add)
            in_code[best_add] = 1
            cov[nbr[best_add]] += 1
            # Try to remove redundant codewords
            changed = True
            while changed:
                changed = False
                for j in range(len(code) - 1, -1, -1):
                    idx = code[j]
                    if np.min(cov[nbr[idx]]) >= 2:
                        cov[nbr[idx]] -= 1
                        in_code[idx] = 0
                        code.pop(j)
                        changed = True
                        break
            if len(code) < best_size:
                best_size = len(code)
                best_code = code[:]
                no_improve = 0
            else:
                no_improve += 1

    # ---- Phase 3: Multiple random restarts with incremental coverage ----
    if time.time() < deadline - 10.0:
        for restart in range(8):
            if time.time() > deadline - 20.0:
                break
            # Perturb: swap 2-4 random codewords
            perturbed = best_code[:]
            n_swaps = int(rng.integers(2, 5))
            for _ in range(n_swaps):
                if len(perturbed) < 2:
                    break
                a = int(rng.integers(len(perturbed)))
                new_cw = int(rng.integers(total))
                if new_cw not in perturbed:
                    perturbed[a] = new_cw
            # Verify validity and fix
            cov_ps = np.zeros(total, dtype=np.int8)
            in_ps = np.zeros(total, dtype=np.int8)
            for idx in perturbed:
                cov_ps[nbr[idx]] = 1
                in_ps[idx] = 1
            if not np.all(cov_ps == 1):
                uncovered_mask = (cov_ps == 0)
                while np.any(uncovered_mask):
                    if time.time() > deadline - 20.0:
                        break
                    counts = np.sum(uncovered_mask[nbr], axis=1)
                    counts[in_ps == 1] = -1
                    best = int(np.argmax(counts))
                    if counts[best] <= 0:
                        break
                    perturbed.append(best)
                    in_ps[best] = 1
                    cov_ps[nbr[best]] = 1
                if not np.all(cov_ps == 1):
                    continue
            # Local search on perturbed with incremental coverage
            code_ps = perturbed[:]
            cov_ps2 = np.zeros(total, dtype=np.int32)
            in_ps2 = np.zeros(total, dtype=np.int8)
            for idx in code_ps:
                cov_ps2[nbr[idx]] += 1
                in_ps2[idx] = 1
            no_imp_ps = 0
            while time.time() < deadline - 20.0 and no_imp_ps < 800:
                if not code_ps:
                    break
                idx_rm = int(rng.integers(len(code_ps)))
                cr = code_ps[idx_rm]
                code_ps.pop(idx_rm)
                in_ps2[cr] = 0
                cov_ps2[nbr[cr]] -= 1
                um = (cov_ps2 == 0)
                if not np.any(um):
                    changed = True
                    while changed:
                        changed = False
                        for j in range(len(code_ps) - 1, -1, -1):
                            idx = code_ps[j]
                            if np.min(cov_ps2[nbr[idx]]) >= 2:
                                cov_ps2[nbr[idx]] -= 1
                                in_ps2[idx] = 0
                                code_ps.pop(j)
                                changed = True
                                break
                    if len(code_ps) < best_size:
                        best_size = len(code_ps)
                        best_code = code_ps[:]
                        no_improve = 0
                    no_imp_ps += 1
                    continue
                ca = np.sum(um[nbr], axis=1)
                ca[in_ps2 == 1] = -1
                ba = int(np.argmax(ca))
                if ca[ba] <= 0:
                    code_ps.append(cr)
                    in_ps2[cr] = 1
                    cov_ps2[nbr[cr]] += 1
                    no_imp_ps += 1
                else:
                    code_ps.append(ba)
                    in_ps2[ba] = 1
                    cov_ps2[nbr[ba]] += 1
                    changed = True
                    while changed:
                        changed = False
                        for j in range(len(code_ps) - 1, -1, -1):
                            idx = code_ps[j]
                            if np.min(cov_ps2[nbr[idx]]) >= 2:
                                cov_ps2[nbr[idx]] -= 1
                                in_ps2[idx] = 0
                                code_ps.pop(j)
                                changed = True
                                break
                    if len(code_ps) < best_size:
                        best_size = len(code_ps)
                        best_code = code_ps[:]
                        no_improve = 0
                    no_imp_ps += 1

    # ---- Phase 4: Genetic algorithm on population of valid codes ----
    if time.time() < deadline - 15.0:
        def compute_fitness(cw_list, cov_arr):
            """Sum of squared coverage counts - lower is better (more uniform)"""
            return int(np.sum(cov_arr[cov_arr > 0] ** 2))

        def remove_redundant(cw_list, cov_arr, in_arr):
            """Remove codewords where all neighbors have coverage >= 2"""
            changed = True
            while changed:
                changed = False
                for j in range(len(cw_list) - 1, -1, -1):
                    idx = cw_list[j]
                    if np.min(cov_arr[nbr[idx]]) >= 2:
                        cov_arr[nbr[idx]] -= 1
                        in_arr[idx] = 0
                        cw_list.pop(j)
                        changed = True
                        break
            return cw_list

        def crossover_preserve(cw_a, cw_b, rng, nbr_arr, total_sz):
            """Crossover that preserves Hamming ball structure:
            Take a random subset of A's codewords, then greedily add from B
            to cover any gaps, maintaining validity."""
            if not cw_a or not cw_b:
                return cw_a[:] if cw_a else cw_b[:]
            
            # Select ~60-80% of codewords from A
            frac = rng.uniform(0.6, 0.8)
            n_keep = max(1, int(len(cw_a) * frac))
            keep_indices = rng.choice(len(cw_a), size=n_keep, replace=False)
            child = [cw_a[i] for i in keep_indices]
            
            # Build coverage of child
            cov_c = np.zeros(total_sz, dtype=np.int32)
            in_c = np.zeros(total_sz, dtype=np.int8)
            for idx in child:
                cov_c[nbr_arr[idx]] += 1
                in_c[idx] = 1
            
            # Find uncovered words
            uncovered = (cov_c == 0)
            
            # Greedily add from B (or best candidates) to cover gaps
            while np.any(uncovered):
                if time.time() > deadline - 10.0:
                    break
                # Prefer adding codewords from B that cover uncovered words
                counts_b = np.sum(uncovered[nbr_arr], axis=1)
                # Weight toward B's codewords
                b_set = set(cw_b)
                for idx in b_set:
                    if not in_c[idx]:
                        counts_b[idx] += 5  # bonus for being in B
                counts_b[in_c == 1] = -1
                best = int(np.argmax(counts_b))
                if counts_b[best] <= 0:
                    break
                child.append(best)
                in_c[best] = 1
                cov_c[nbr_arr[best]] += 1
                uncovered = (cov_c == 0)
            
            return child

        def mutate_code(cw_list, rng, nbr_arr, total_sz):
            """Mutation: swap 1-3 random codewords with new ones, repair if needed"""
            if not cw_list:
                return cw_list[:]
            mutated = cw_list[:]
            n_mut = int(rng.integers(1, 4))
            for _ in range(n_mut):
                if len(mutated) < 2:
                    break
                pos = int(rng.integers(len(mutated)))
                new_cw = int(rng.integers(total_sz))
                if new_cw not in mutated:
                    mutated[pos] = new_cw
            
            # Verify and repair
            cov_m = np.zeros(total_sz, dtype=np.int8)
            in_m = np.zeros(total_sz, dtype=np.int8)
            for idx in mutated:
                cov_m[nbr_arr[idx]] = 1
                in_m[idx] = 1
            
            if not np.all(cov_m == 1):
                uncovered_mask = (cov_m == 0)
                while np.any(uncovered_mask):
                    if time.time() > deadline - 10.0:
                        break
                    counts = np.sum(uncovered_mask[nbr_arr], axis=1)
                    counts[in_m == 1] = -1
                    best = int(np.argmax(counts))
                    if counts[best] <= 0:
                        break
                    mutated.append(best)
                    in_m[best] = 1
                    cov_m[nbr_arr[best]] = 1
                if not np.all(cov_m == 1):
                    return cw_list[:]  # revert to parent
            return mutated

        # Initialize population from best_code
        pop_size = 6
        population = []
        base_cov = np.zeros(total, dtype=np.int32)
        base_in = np.zeros(total, dtype=np.int8)
        for idx in best_code:
            base_cov[nbr[idx]] += 1
            base_in[idx] = 1
        population.append(best_code[:])
        
        # Add diverse individuals
        for _ in range(pop_size - 1):
            if time.time() > deadline - 15.0:
                break
            # Create variant by removing some redundant and adding others
            variant = best_code[:]
            cov_v = base_cov.copy()
            in_v = base_in.copy()
            
            # Remove 1-3 random non-critical codewords
            n_rm = int(rng.integers(1, 4))
            for _ in range(n_rm):
                if not variant:
                    break
                rm_idx = int(rng.integers(len(variant)))
                c_rm = variant[rm_idx]
                variant.pop(rm_idx)
                cov_v[nbr[c_rm]] -= 1
                in_v[c_rm] = 0
            
            # Repair
            uncovered = (cov_v == 0)
            while np.any(uncovered):
                if time.time() > deadline - 15.0:
                    break
                counts = np.sum(uncovered[nbr], axis=1)
                counts[in_v == 1] = -1
                best = int(np.argmax(counts))
                if counts[best] <= 0:
                    break
                variant.append(best)
                in_v[best] = 1
                cov_v[nbr[best]] += 1
                uncovered = (cov_v == 0)
            
            if np.all(cov_v == 1):
                # Local search polish
                remove_redundant(variant, cov_v, in_v)
                population.append(variant[:])
        
        # Genetic algorithm loop
        ga_generations = 0
        ga_max_gen = 30
        ga_no_improve = 0
        ga_max_no_improve = 20
        
        while ga_generations < ga_max_gen and ga_no_improve < ga_max_no_improve and time.time() < deadline - 12.0:
            ga_generations += 1
            
            # Evaluate fitness of population
            fitnesses = []
            for ind in population:
                if not ind:
                    fitnesses.append(float('inf'))
                    continue
                cov_ind = np.zeros(total, dtype=np.int32)
                for idx in ind:
                    cov_ind[nbr[idx]] += 1
                # Check validity
                if np.any(cov_ind == 0):
                    fitnesses.append(float('inf'))
                else:
                    # Fitness: size + penalty for uneven coverage
                    fitnesses.append(len(ind) + 0.01 * int(np.sum(cov_ind[cov_ind > 0] ** 2)))
            
            # Sort by fitness
            sorted_idx = sorted(range(len(population)), key=lambda i: fitnesses[i])
            
            # Check for improvement
            best_fit_idx = sorted_idx[0]
            if len(population[best_fit_idx]) < best_size:
                best_size = len(population[best_fit_idx])
                best_code = population[best_fit_idx][:]
                ga_no_improve = 0
            else:
                ga_no_improve += 1
            
            # Selection: keep top half
            half = max(1, len(population) // 2)
            survivors = [population[sorted_idx[i]] for i in range(half)]
            
            # Generate offspring
            new_pop = survivors[:]
            while len(new_pop) < pop_size:
                if time.time() > deadline - 12.0:
                    break
                # Pick parents (tournament of 2)
                p1_idx = sorted_idx[int(rng.integers(0, half))]
                p2_idx = sorted_idx[int(rng.integers(0, half))]
                p1 = population[p1_idx]
                p2 = population[p2_idx]
                
                # Crossover
                if rng.random() < 0.7:
                    child = crossover_preserve(p1, p2, rng, nbr, total)
                else:
                    child = mutate_code(p1, rng, nbr, total)
                
                if child:
                    # Local search polish on child
                    cov_ch = np.zeros(total, dtype=np.int32)
                    in_ch = np.zeros(total, dtype=np.int8)
                    for idx in child:
                        cov_ch[nbr[idx]] += 1
                        in_ch[idx] = 1
                    
                    # Remove redundant
                    remove_redundant(child, cov_ch, in_ch)
                    
                    # Quick local search (a few iterations)
                    for _ in range(3):
                        if time.time() > deadline - 12.0 or not child:
                            break
                        ri = int(rng.integers(len(child)))
                        cr = child[ri]
                        child.pop(ri)
                        in_ch[cr] = 0
                        cov_ch[nbr[cr]] -= 1
                        um = (cov_ch == 0)
                        if not np.any(um):
                            remove_redundant(child, cov_ch, in_ch)
                        else:
                            ca = np.sum(um[nbr], axis=1)
                            ca[in_ch == 1] = -1
                            ba = int(np.argmax(ca))
                            if ca[ba] > 0:
                                child.append(ba)
                                in_ch[ba] = 1
                                cov_ch[nbr[ba]] += 1
                                remove_redundant(child, cov_ch, in_ch)
                            else:
                                child.append(cr)
                                in_ch[cr] = 1
                                cov_ch[nbr[cr]] += 1
                    
                    # Verify
                    if child:
                        cov_fin = np.zeros(total, dtype=np.int8)
                        for idx in child:
                            cov_fin[nbr[idx]] = 1
                        if np.all(cov_fin == 1):
                            new_pop.append(child)
                            if len(child) < best_size:
                                best_size = len(child)
                                best_code = child[:]
                                ga_no_improve = 0
            
            population = new_pop[:pop_size]
            # Pad if needed
            while len(population) < pop_size:
                population.append(best_code[:])

    # ---- Final verification and output ----
    def to_words(idx_list):
        return [list(coords[idx]) for idx in idx_list]

    # Safety: ensure we return within deadline
    if time.time() > deadline - 2.0:
        if best_code:
            fc = np.zeros(total, dtype=np.int8)
            for idx in best_code:
                fc[nbr[idx]] = 1
            if np.all(fc == 1):
                ds = get_direct_sum()
                if len(best_code) <= len(ds):
                    return to_words(best_code)
                return ds
        return get_direct_sum()

        return [list(coords[idx]) for idx in idx_list]

    if best_code:
        fc = np.zeros(total, dtype=np.int8)
        for idx in best_code:
            fc[nbr[idx]] = 1
        if np.all(fc == 1):
            ds = get_direct_sum()
            if len(best_code) <= len(ds):
                return to_words(best_code)
            return ds
    # Fallback
    if code:
        fc = np.zeros(total, dtype=np.int8)
        for idx in code:
            fc[nbr[idx]] = 1
        if np.all(fc == 1):
            ds = get_direct_sum()
            if len(code) <= len(ds):
                return to_words(code)
    return get_direct_sum()
