import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin

    def evaluate(coeffs):
        sq = np.convolve(coeffs, coeffs)
        return int(np.count_nonzero(sq))

    def validate(coeffs):
        if len(coeffs) != 12:
            return False
        if coeffs[-1] == 0:
            return False
        if coeffs[0] == 0:
            return False
        if not all(isinstance(c, int) for c in coeffs):
            return False
        if not all(-64 <= c <= 64 for c in coeffs):
            return False
        if sum(1 for c in coeffs if c != 0) != 12:
            return False
        return True

    def compute_square(coeffs):
        sq = [0] * 23
        for i in range(12):
            ci = coeffs[i]
            if ci == 0:
                continue
            for j in range(12):
                cj = coeffs[j]
                if cj == 0:
                    continue
                sq[i + j] += ci * cj
        return sq

    def count_nz(sq):
        return sum(1 for x in sq if x != 0)

    def try_zero_out(coeffs, k, max_iter=100):
        """Try to zero out sq[k] by modifying one coefficient."""
        sq = compute_square(coeffs)
        if sq[k] == 0:
            return coeffs, count_nz(sq)
        
        current_score = count_nz(sq)
        
        for i in range(12):
            j = k - i
            if j < 0 or j >= 12:
                continue
            
            A = 1 if (2 * i == k) else 0
            B = 0
            C = 0
            
            for l in range(12):
                m = k - l
                if m < 0 or m >= 12:
                    continue
                if l == i:
                    continue
                if m == i:
                    continue
                C += coeffs[l] * coeffs[m]
                
            if i != j:
                B = 2 * coeffs[j]
            else:
                B = 0
            
            if A == 0:
                if B != 0:
                    new_val = -C // B
                    if B * new_val + C == 0:
                        if -64 <= new_val <= 64 and new_val != 0:
                            new_coeffs = coeffs.copy()
                            new_coeffs[i] = new_val
                            new_sq = compute_square(new_coeffs)
                            new_score = count_nz(new_sq)
                            if new_score < current_score:
                                return new_coeffs, new_score
            else:
                disc = B * B - 4 * A * C
                if disc >= 0:
                    sqrt_disc = int(np.sqrt(disc))
                    if sqrt_disc * sqrt_disc == disc:
                        for sign in [1, -1]:
                            num = -B + sign * sqrt_disc
                            den = 2 * A
                            if den != 0 and num % den == 0:
                                new_val = num // den
                                if -64 <= new_val <= 64 and new_val != 0:
                                    new_coeffs = coeffs.copy()
                                    new_coeffs[i] = new_val
                                    new_sq = compute_square(new_coeffs)
                                    new_score = count_nz(new_sq)
                                    if new_score < current_score:
                                        return new_coeffs, new_score
        return coeffs, current_score

    # === GENETIC ALGORITHM WITH BLOCK-BASED CROSSOVER ===
    
    # Define block structure: partition 12 positions into contiguous blocks
    # Block boundaries: [0,3,6,9,12) -> 4 blocks of size 3
    # Or: [0,2,5,8,12) -> blocks of sizes 2,3,3,4
    # We'll use multiple block structures for diversity
    
    BLOCK_STRUCTURES = [
        [0, 3, 6, 9, 12],      # 4 blocks of size 3
        [0, 2, 5, 8, 12],      # sizes 2,3,3,4
        [0, 4, 8, 12],         # 3 blocks of size 4
        [0, 2, 4, 6, 8, 10, 12],  # 6 blocks of size 2
        [0, 1, 3, 5, 7, 9, 11, 12],  # alternating sizes
    ]

    def get_blocks(structure):
        """Get list of (start, end) for each block."""
        blocks = []
        for i in range(len(structure) - 1):
            blocks.append((structure[i], structure[i+1]))
        return blocks

    def block_crossover(parent1, parent2, structure):
        """
        Crossover: for each block, randomly choose which parent contributes
        that entire block. This preserves structural collision patterns
        within each block while recombining between blocks.
        """
        blocks = get_blocks(structure)
        child = [0] * 12
        for start, end in blocks:
            if random.random() < 0.5:
                child[start:end] = parent1[start:end]
            else:
                child[start:end] = parent2[start:end]
        
        # Ensure first and last are nonzero
        if child[0] == 0:
            child[0] = random.choice([-1, 1])
        if child[11] == 0:
            child[11] = random.choice([-1, 1])
        
        # Clamp values
        for i in range(12):
            if child[i] > 64:
                child[i] = 64
            elif child[i] < -64:
                child[i] = -64
        
        return child

    def block_shift_mutation(individual, structure, shift_magnitude=2):
        """
        Mutation: apply a small integer shift to an entire block.
        This explores the local landscape of sum-cancellation structures
        without destroying the global support geometry.
        """
        blocks = get_blocks(structure)
        if not blocks:
            return individual
        
        # Pick 1-2 blocks to mutate
        num_mutations = random.choice([1, 1, 2])
        mutated = individual.copy()
        
        for _ in range(num_mutations):
            start, end = random.choice(blocks)
            shift = random.randint(-shift_magnitude, shift_magnitude)
            if shift == 0:
                continue
            
            for i in range(start, end):
                new_val = mutated[i] + shift
                # Clamp to valid range, avoid zero
                if new_val == 0:
                    new_val = 1 if shift > 0 else -1
                if new_val > 64:
                    new_val = 64
                elif new_val < -64:
                    new_val = -64
                mutated[i] = new_val
        
        # Ensure first and last nonzero
        if mutated[0] == 0:
            mutated[0] = 1
        if mutated[11] == 0:
            mutated[11] = 1
            
        return mutated

    def individual_mutation(individual):
        """Standard per-coefficient mutation for diversity."""
        mutated = individual.copy()
        num_mutations = random.choice([1, 1, 2, 3])
        for _ in range(num_mutations):
            idx = random.randint(0, 11)
            old_val = mutated[idx]
            if random.random() < 0.5:
                new_val = old_val + random.choice([-3, -2, -1, 1, 2, 3])
            else:
                new_val = random.choice([-64, -10, -3, -2, -1, 1, 2, 3, 10, 64])
            if new_val == 0:
                new_val = 1 if random.random() < 0.5 else -1
            if -64 <= new_val <= 64:
                mutated[idx] = new_val
        
        if mutated[0] == 0:
            mutated[0] = 1
        if mutated[11] == 0:
            mutated[11] = 1
        return mutated

    def run_genetic_algorithm(pop_size=40, generations=200, elite_size=4):
        nonlocal best_score, best_poly
        
        if time.time() > deadline - 15:
            return
        
        # Initialize population with diverse structures
        population = []
        
        # Seed with some known good patterns
        seed_patterns = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],  # Will be fixed
            [1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1],
            [1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2, 1],
            [1, -2, 1, 1, -2, 1, 1, -2, 1, 1, -2, 1],
            [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
            [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        ]
        
        for sp in seed_patterns:
            # Fix any zeros
            fixed = sp.copy()
            for i in range(12):
                if fixed[i] == 0:
                    fixed[i] = random.choice([-1, 1])
            population.append(fixed)
        
        # Fill rest with random
        while len(population) < pop_size:
            if random.random() < 0.3:
                # Symmetric pattern
                half = [random.choice([-2, -1, 1, 2]) for _ in range(6)]
                if random.random() < 0.5:
                    full = half + half
                else:
                    full = half + [-x for x in half]
            elif random.random() < 0.5:
                # Arithmetic-like
                base = random.choice([-2, -1, 1, 2])
                step = random.choice([-1, 1])
                full = [base + i * step for i in range(12)]
            else:
                full = [random.choice([-3, -2, -1, 1, 2, 3]) for _ in range(12)]
            
            # Fix zeros
            for i in range(12):
                if full[i] == 0:
                    full[i] = random.choice([-1, 1])
            population.append(full)
        
        # Evaluate initial population
        scores = [evaluate(ind) for ind in population]
        
        # Update best
        for i, score in enumerate(scores):
            if score < best_score:
                best_score = score
                best_poly = population[i].copy()
        
        # Sort by score
        pop_with_scores = sorted(zip(scores, population), key=lambda x: x[0])
        
        for gen in range(generations):
            if time.time() > deadline - 10:
                break
            
            new_population = []
            
            # Elitism: keep best individuals
            for i in range(min(elite_size, len(pop_with_scores))):
                new_population.append(pop_with_scores[i][1].copy())
            
            # Generate offspring
            while len(new_population) < pop_size:
                if time.time() > deadline - 8:
                    break
                
                # Selection: tournament
                p1_idx = random.choice(range(min(8, len(pop_with_scores))))
                p2_idx = random.choice(range(min(8, len(pop_with_scores))))
                p1 = pop_with_scores[p1_idx][1]
                p2 = pop_with_scores[p2_idx][1]
                
                # Choose operation
                op = random.random()
                
                if op < 0.4:
                    # Block crossover with random structure
                    structure = random.choice(BLOCK_STRUCTURES)
                    child = block_crossover(p1, p2, structure)
                elif op < 0.6:
                    # Block shift mutation of a good individual
                    structure = random.choice(BLOCK_STRUCTURES)
                    child = block_shift_mutation(p1, structure)
                elif op < 0.8:
                    # Standard mutation
                    child = individual_mutation(p1)
                else:
                    # Blend: crossover then mutation
                    structure = random.choice(BLOCK_STRUCTURES)
                    child = block_crossover(p1, p2, structure)
                    child = individual_mutation(child)
                
                new_population.append(child)
            
            # Evaluate new population
            new_scores = [evaluate(ind) for ind in new_population]
            
            # Update best
            for i, score in enumerate(new_scores):
                if score < best_score:
                    best_score = score
                    best_poly = new_population[i].copy()
            
            # Select for next generation
            pop_with_scores = sorted(zip(new_scores, new_population), key=lambda x: x[0])
        
        return

    # Run the genetic algorithm
    run_genetic_algorithm(pop_size=40, generations=200, elite_size=4)
    
    # Final polish: local search with targeted zeroing from best found
    if best_poly is not None and time.time() < deadline - 10:
        curr = best_poly.copy()
        curr_score = best_score
        
        no_improve = 0
        max_no_improve = 1000
        
        while time.time() < deadline - 3 and no_improve < max_no_improve:
            sq = compute_square(curr)
            nonzero_ks = [k for k in range(23) if sq[k] != 0]
            
            if not nonzero_ks:
                break
            
            k = random.choice(nonzero_ks)
            new_curr, new_score = try_zero_out(curr, k)
            
            if new_score < curr_score:
                curr = new_curr
                curr_score = new_score
                no_improve = 0
                if curr_score < best_score:
                    best_score = curr_score
                    best_poly = curr.copy()
            else:
                idx = random.randint(0, 11)
                old_val = curr[idx]
                new_val = old_val + random.choice([-3, -2, -1, 1, 2, 3])
                if -64 <= new_val <= 64 and new_val != 0:
                    curr[idx] = new_val
                    new_score = evaluate(curr)
                    if new_score < curr_score:
                        curr_score = new_score
                        no_improve = 0
                        if curr_score < best_score:
                            best_score = curr_score
                            best_poly = curr.copy()
                    else:
                        curr[idx] = old_val
                        no_improve += 1
                else:
                    no_improve += 1

    if best_poly is None:
        best_poly = [1] * 12

    if not validate(best_poly):
        best_poly = [1] * 12

    return best_poly
