import time
import numpy as np
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill

    # Fallback: (1 + x + ... + x^11)^2 has 21 non-zero terms
    fallback = [1] * 12
    fallback_score = 21

    best_score = fallback_score
    best_candidate = fallback

    def compute_score_fast(exps, coeffs):
        """
        Fast computation of nonzero terms in P(x)^2.
        exps: list of 12 exponents (sorted)
        coeffs: list of 12 coefficients
        Returns count of nonzero terms, or inf if invalid.
        """
        if not exps or not coeffs:
            return float('inf')

        max_deg = max(exps)
        if max_deg > 40:
            return float('inf')
        if any(abs(c) > 64 for c in coeffs):
            return float('inf')
        if any(c == 0 for c in coeffs):
            return float('inf')
        if len(set(exps)) != 12:
            return float('inf')

        # Build dense array
        p = np.zeros(max_deg + 1, dtype=np.int64)
        for e, c in zip(exps, coeffs):
            p[e] = c

        q = np.convolve(p, p)
        count = int(np.count_nonzero(q))
        return count

    def _update_best(exps, coeffs, score):
        """Update the global best candidate if improved."""
        nonlocal best_score, best_candidate
        if score < best_score:
            max_deg = max(exps)
            dense = [0] * (max_deg + 1)
            for e, c in zip(exps, coeffs):
                dense[e] = c
            if len(dense) <= 41 and dense[-1] != 0:
                best_score = score
                best_candidate = dense

    def crossover(parent1_exps, parent1_coeffs, parent2_exps, parent2_coeffs):
        """
        Uniform crossover: for each of the 12 positions, pick from parent1 or parent2.
        Returns (exps, coeffs) or None if invalid.
        """
        child_exps = []
        child_coeffs = []
        for i in range(12):
            if random.random() < 0.5:
                child_exps.append(parent1_exps[i])
                child_coeffs.append(parent1_coeffs[i])
            else:
                child_exps.append(parent2_exps[i])
                child_coeffs.append(parent2_coeffs[i])

        # Check validity
        if len(set(child_exps)) != 12:
            return None
        if max(child_exps) > 40:
            return None
        if any(abs(c) > 64 for c in child_coeffs):
            return None
        if any(c == 0 for c in child_coeffs):
            return None

        idx = np.argsort(child_exps)
        child_exps = [child_exps[i] for i in idx]
        child_coeffs = [child_coeffs[i] for i in idx]
        return child_exps, child_coeffs

    def mutate(exps, coeffs, mutation_rate=0.3):
        """
        Apply random mutations to a candidate.
        """
        new_exps = exps[:]
        new_coeffs = coeffs[:]

        for i in range(12):
            if random.random() < mutation_rate:
                move_type = random.random()
                if move_type < 0.3:
                    # Flip sign
                    new_coeffs[i] = -new_coeffs[i]
                elif move_type < 0.6:
                    # Shift exponent
                    delta = random.choice([-3, -2, -1, 1, 2, 3])
                    new_e = new_exps[i] + delta
                    if 0 <= new_e <= 40 and new_e not in new_exps:
                        new_exps[i] = new_e
                elif move_type < 0.8:
                    # Jump exponent
                    new_e = random.randint(0, 40)
                    if new_e not in new_exps:
                        new_exps[i] = new_e
                else:
                    # Change coefficient magnitude
                    new_c = random.choice([-3, -2, -1, 1, 2, 3, 4, 5, 8, 16])
                    if new_c != new_coeffs[i] and abs(new_c) <= 64:
                        new_coeffs[i] = new_c

        # Validate
        if len(set(new_exps)) != 12:
            return None
        if max(new_exps) > 40:
            return None
        if any(abs(c) > 64 for c in new_coeffs):
            return None
        if any(c == 0 for c in new_coeffs):
            return None

        idx = np.argsort(new_exps)
        new_exps = [new_exps[i] for i in idx]
        new_coeffs = [new_coeffs[i] for i in idx]
        return new_exps, new_coeffs

    def genetic_algorithm(pop_size=30, generations=50, elite_size=5):
        """
        Genetic algorithm with crossover and mutation.
        """
        nonlocal best_score, best_candidate

        # Initialize population
        population = []
        scores = []

        # Add some structured initial candidates
        initial_candidates = []

        # Consecutive exponents with various sign patterns
        for start in range(0, 6):
            exps = [start + i for i in range(12)]
            if max(exps) <= 40:
                patterns = [
                    [1] * 12,
                    [1 if i % 2 == 0 else -1 for i in range(12)],
                    [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
                    [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                    [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
                ]
                for pattern in patterns:
                    initial_candidates.append((exps[:], pattern[:]))

        # Gapped exponents
        gap_patterns = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14],
            [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15],
            [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
            [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        ]
        for exps in gap_patterns:
            for coeffs in [[1] * 12, [1 if i % 2 == 0 else -1 for i in range(12)]]:
                if max(exps) <= 40:
                    initial_candidates.append((exps[:], coeffs[:]))

        # Random sparse exponents
        for _ in range(10):
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            initial_candidates.append((exps, coeffs))

        # Clustered exponents
        for _ in range(5):
            cluster_centers = random.sample(range(0, 41), 3)
            exps = []
            for center in cluster_centers:
                for offset in random.sample([-2, -1, 0, 1, 2], 4):
                    e = center + offset
                    if 0 <= e <= 40 and e not in exps:
                        exps.append(e)
            while len(exps) < 12:
                e = random.randint(0, 40)
                if e not in exps:
                    exps.append(e)
            exps = sorted(exps[:12])
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            initial_candidates.append((exps, coeffs))

        # Evaluate initial candidates and add to population
        for exps, coeffs in initial_candidates:
            if len(population) >= pop_size:
                break
            score = compute_score_fast(exps, coeffs)
            if score < float('inf'):
                population.append((exps, coeffs))
                scores.append(score)
                _update_best(exps, coeffs, score)

        # Fill remaining population with random candidates
        while len(population) < pop_size:
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            score = compute_score_fast(exps, coeffs)
            if score < float('inf'):
                population.append((exps, coeffs))
                scores.append(score)
                _update_best(exps, coeffs, score)

        # Genetic algorithm loop
        for gen in range(generations):
            if time.time() > deadline - 5:
                break

            # Sort by score
            sorted_idx = np.argsort(scores)
            population = [population[i] for i in sorted_idx]
            scores = [scores[i] for i in sorted_idx]

            # Elitism: keep best elite_size
            new_population = []
            new_scores = []
            for i in range(elite_size):
                new_population.append(population[i])
                new_scores.append(scores[i])

            # Generate offspring
            while len(new_population) < pop_size:
                if time.time() > deadline - 5:
                    break

                # Tournament selection
                def tournament_select():
                    candidates = random.sample(range(len(population)), min(3, len(population)))
                    best = min(candidates, key=lambda i: scores[i])
                    return population[best]

                parent1 = tournament_select()
                parent2 = tournament_select()

                # Crossover
                child = crossover(parent1[0], parent1[1], parent2[0], parent2[1])
                if child is None:
                    # Fallback to mutation of a random parent
                    parent = random.choice([parent1, parent2])
                    child = mutate(parent[0], parent[1], mutation_rate=0.3)
                    if child is None:
                        continue

                # Mutation
                child_mutated = mutate(child[0], child[1], mutation_rate=0.2)
                if child_mutated is not None:
                    child = child_mutated
                else:
                    child = mutate(parent1[0], parent1[1], mutation_rate=0.3)
                    if child is None:
                        continue

                score = compute_score_fast(child[0], child[1])
                if score < float('inf'):
                    new_population.append(child)
                    new_scores.append(score)
                    _update_best(child[0], child[1], score)

            population = new_population
            scores = new_scores

            if best_score <= 12:
                break

    def simulated_annealing(exps, coeffs, max_iters=2000):
        """
        Simulated annealing with diverse moves.
        """
        nonlocal best_score, best_candidate

        # Ensure sorted
        idx = np.argsort(exps)
        exps = [exps[i] for i in idx]
        coeffs = [coeffs[i] for i in idx]

        current_score = compute_score_fast(exps, coeffs)
        _update_best(exps, coeffs, current_score)

        if current_score <= 12:
            return

        # Annealing parameters
        T = 5.0
        T_min = 0.1
        cooling = 0.995

        for it in range(max_iters):
            if time.time() > deadline - 3:
                break

            # Choose move type
            move_type = random.random()
            new_exps = exps[:]
            new_coeffs = coeffs[:]

            if move_type < 0.25:
                # Flip sign of one coefficient
                i = random.randint(0, 11)
                new_coeffs[i] = -new_coeffs[i]
            elif move_type < 0.5:
                # Shift one exponent by small amount
                i = random.randint(0, 11)
                delta = random.choice([-3, -2, -1, 1, 2, 3])
                new_e = new_exps[i] + delta
                if 0 <= new_e <= 40 and new_e not in new_exps:
                    new_exps[i] = new_e
                else:
                    continue
            elif move_type < 0.7:
                # Move two exponents together
                i, j = random.sample(range(12), 2)
                delta = random.choice([-3, -2, -1, 1, 2, 3])
                new_ei = new_exps[i] + delta
                new_ej = new_exps[j] + delta
                if 0 <= new_ei <= 40 and 0 <= new_ej <= 40:
                    temp = new_exps[:]
                    temp[i] = new_ei
                    temp[j] = new_ej
                    if len(set(temp)) == 12:
                        new_exps[i] = new_ei
                        new_exps[j] = new_ej
                    else:
                        continue
                else:
                    continue
            elif move_type < 0.85:
                # Jump: random reposition of one exponent
                i = random.randint(0, 11)
                new_e = random.randint(0, 40)
                if new_e not in new_exps:
                    new_exps[i] = new_e
                else:
                    continue
            else:
                # Change coefficient magnitude
                i = random.randint(0, 11)
                old_c = new_coeffs[i]
                new_c = random.choice([-3, -2, -1, 1, 2, 3, 4, 5, 8, 16])
                if new_c != old_c and abs(new_c) <= 64:
                    new_coeffs[i] = new_c
                else:
                    continue

            # Sort new state
            nidx = np.argsort(new_exps)
            new_exps = [new_exps[i] for i in nidx]
            new_coeffs = [new_coeffs[i] for i in nidx]

            new_score = compute_score_fast(new_exps, new_coeffs)

            # Accept or reject
            if new_score < current_score:
                exps = new_exps
                coeffs = new_coeffs
                current_score = new_score
                _update_best(exps, coeffs, current_score)
                if current_score <= 12:
                    break
            else:
                # Simulated annealing acceptance
                delta_score = new_score - current_score
                if T > T_min:
                    accept_prob = np.exp(-delta_score / T)
                    if random.random() < accept_prob:
                        exps = new_exps
                        coeffs = new_coeffs
                        current_score = new_score

            T = max(T * cooling, T_min)

        return exps, coeffs, current_score

    # --- Main search loop ---

    # Set random seed for reproducibility
    random.seed(42)

    # Phase 1: Genetic Algorithm
    genetic_algorithm(pop_size=30, generations=50, elite_size=5)

    if best_score <= 12:
        return best_candidate

    # Phase 2: Simulated Annealing on best candidate
    if time.time() < deadline - 10:
        best_exps = []
        best_coeffs = []
        for i, c in enumerate(best_candidate):
            if c != 0:
                best_exps.append(i)
                best_coeffs.append(c)

        if len(best_exps) == 12:
            simulated_annealing(best_exps, best_coeffs, max_iters=3000)

    # Phase 3: Random restarts if time remains
    if time.time() < deadline - 5 and best_score > 12:
        for _ in range(3):
            if time.time() > deadline - 3:
                break
            exps = sorted(random.sample(range(0, 41), 12))
            coeffs = [random.choice([-1, 1]) for _ in range(12)]
            simulated_annealing(exps, coeffs, max_iters=1500)
            if best_score <= 12:
                break

    return best_candidate

# EVOLVE-BLOCK-END
