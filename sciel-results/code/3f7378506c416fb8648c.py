"""Sparse squares of twelve-term integer polynomials - basis + genetic search."""
import time
import random
import numpy as np
from itertools import combinations, product

def construct():
    deadline = time.time() + 85.0
    random.seed(42)
    np.random.seed(42)

    # ---------- fast exact square via np.convolve on int64 ----------
    def square_count(poly):
        """Return (count_nonzero, square_array) for exact integer square."""
        n = len(poly)
        if n == 0:
            return 0, np.array([], dtype=np.int64)
        # pad to at least 1
        a = np.asarray(poly, dtype=np.int64)
        # np.convolve is exact for int64 when values are small enough
        sq = np.convolve(a, a, mode='full')
        cnt = int(np.count_nonzero(sq))
        return cnt, sq

    def trim(poly):
        while len(poly) > 1 and poly[-1] == 0:
            poly.pop()
        return poly

    def is_valid(poly):
        if len(poly) > 41:
            return False
        if len(poly) < 1 or poly[-1] == 0:
            return False
        if sum(1 for c in poly if c != 0) != 12:
            return False
        if any(c < -64 or c > 64 for c in poly):
            return False
        return True

    # ---------- precompute a set of "low-square" basis polynomials ----------
    # A monomial x^k has square = x^{2k}: 1 nonzero.
    # A binomial (a x^i + b x^j)^2 has 3 nonzeros unless a^2 + 2ab + b^2 collides.
    # A trinomial can have as few as 3 nonzeros in its square if terms cancel.
    #
    # Key insight: if we pick exponents in an arithmetic progression with
    # alternating signs, the square can have many cancellations.
    #
    # Precompute a pool of candidate exponent sets that are known to produce
    # sparse squares, then search over sign/coefficient assignments.

    # Build a diverse pool of exponent sets
    exponent_pool = []

    # 1) Arithmetic progressions (various steps, offsets)
    for step in range(1, 7):
        for offset in range(0, 42 - 11*step, max(1, step)):
            exponents = [offset + i*step for i in range(12)]
            if max(exponents) < 41:
                exponent_pool.append(tuple(exponents))

    # 2) Two-AP interleaved: split 12 terms into two groups of 6
    for step1 in range(1, 5):
        for step2 in range(1, 5):
            for off1 in range(0, 4):
                for off2 in range(0, 4):
                    try:
                        g1 = [off1 + i*step1 for i in range(6)]
                        g2 = [off2 + i*step2 for i in range(6)]
                        exponents = sorted(g1 + g2)
                        if len(exponents) == 12 and max(exponents) < 41 and len(set(exponents)) == 12:
                            exponent_pool.append(tuple(exponents))
                    except:
                        pass

    # 3) Fibonacci-like / quadratic growth
    for base in range(1, 4):
        exponents = [int(base * i * i) for i in range(12)]
        if max(exponents) < 41 and len(set(exponents)) == 12:
            exponent_pool.append(tuple(exponents))

    # 4) Random structured sets (seeded)
    rng = random.Random(12345)
    for _ in range(200):
        exponents = sorted(rng.sample(range(41), 12))
        exponent_pool.append(tuple(exponents))

    # Remove duplicates
    exponent_pool = list(set(exponent_pool))

    # ---------- score a candidate (exponents, coefficients) ----------
    def make_poly(exponents, coeffs):
        poly = [0] * 41
        for e, c in zip(exponents, coeffs):
            if e < 41:
                poly[e] = c
        return trim(poly)

    def evaluate(exponents, coeffs):
        poly = make_poly(exponents, coeffs)
        if not is_valid(poly):
            return 999
        cnt, _ = square_count(poly)
        return cnt

    # ---------- Phase 1: Exhaustive sign search on best AP families ----------
    best_count = 23
    best_poly = [1]*12

    # Pick the most promising exponent sets (shortest range = most overlap)
    exponent_pool.sort(key=lambda ex: max(ex) - min(ex))

    # For the top families, try all 2^11 sign patterns (fix first = +1)
    top_families = exponent_pool[:30]
    for exp_set in top_families:
        if time.time() > deadline:
            break
        # Try all sign patterns
        for signs in product([1, -1], repeat=11):
            if time.time() > deadline:
                break
            coeffs = [1] + list(signs)
            cnt = evaluate(exp_set, coeffs)
            if cnt < best_count:
                best_count = cnt
                best_poly = make_poly(exp_set, coeffs)

    # Also try with coefficients ±2 on some positions
    if time.time() < deadline:
        for exp_set in exponent_pool[:15]:
            if time.time() > deadline:
                break
            # Try: 1 position = ±2, rest ±1
            for special_idx in range(12):
                if time.time() > deadline:
                    break
                for special_val in [2, -2]:
                    if time.time() > deadline:
                        break
                    # Fix first coeff = +1 if special_idx != 0
                    # Generate sign patterns for the other 11 positions
                    # This is 2^11 = 2048 patterns per (exp_set, special_idx, special_val)
                    # Too many for all, so sample
                    for _ in range(200):
                        if time.time() > deadline:
                            break
                        coeffs = [1]*12
                        coeffs[special_idx] = special_val
                        # Random signs for non-special, non-first positions
                        for i in range(12):
                            if i == 0 or i == special_idx:
                                continue
                            if random.random() < 0.5:
                                coeffs[i] = -1
                        cnt = evaluate(exp_set, coeffs)
                        if cnt < best_count:
                            best_count = cnt
                            best_poly = make_poly(exp_set, coeffs)

    # ---------- Phase 2: Genetic algorithm over (exponents, coefficients) ----------
    # Chromosome: 12 exponents (0..40) + 12 coefficients (from small set)
    # Use a population of candidates, evolve over generations.

    coeff_choices = [1, -1, 2, -2, 3, -3]

    def decode(chromosome):
        exponents = list(chromosome[:12])
        coeffs = [coeff_choices[c % len(coeff_choices)] for c in chromosome[12:24]]
        # Ensure distinct exponents
        seen = set()
        valid = True
        for e in exponents:
            if e in seen or e > 40:
                valid = False
                break
            seen.add(e)
        if not valid:
            return None
        return exponents, coeffs

    def chromosome_from_poly(poly):
        # Extract 12 nonzero terms
        terms = [(i, c) for i, c in enumerate(poly) if c != 0]
        if len(terms) != 12:
            return None
        exponents = [t[0] for t in terms]
        coeffs = []
        for c in [t[1] for t in terms]:
            # Map coefficient to index in coeff_choices
            if c in coeff_choices:
                coeffs.append(coeff_choices.index(c))
            else:
                # Snap to nearest
                best = min(coeff_choices, key=lambda x: abs(x - c))
                coeffs.append(coeff_choices.index(best))
        return exponents + coeffs

    # Initialize population
    pop_size = 60
    population = []

    # Add best found so far
    if is_valid(best_poly):
        chrom = chromosome_from_poly(best_poly)
        if chrom:
            population.append(chrom)

    # Add random candidates
    while len(population) < pop_size:
        exps = sorted(random.sample(range(41), 12))
        coefs = [random.choice(range(len(coeff_choices))) for _ in range(12)]
        population.append(exps + coefs)

    # Evaluate initial population
    scores = []
    for chrom in population:
        decoded = decode(chrom)
        if decoded is None:
            scores.append(999)
        else:
            scores.append(evaluate(decoded[0], decoded[1]))

    # Track best
    best_idx = int(np.argmin(scores))
    best_chrom = population[best_idx][:]
    best_score = scores[best_idx]
    if best_score < best_count:
        best_count = best_score
        decoded = decode(best_chrom)
        if decoded:
            best_poly = make_poly(decoded[0], decoded[1])

    # Genetic loop
    generations = 0
    while time.time() < deadline:
        generations += 1
        new_population = []

        # Elitism: keep top 5
        order = sorted(range(pop_size), key=lambda i: scores[i])
        for i in order[:5]:
            new_population.append(population[i][:])

        # Crossover + mutation for the rest
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            # Tournament selection
            i1 = order[random.randint(0, min(9, pop_size-1))]
            i2 = order[random.randint(0, min(9, pop_size-1))]
            p1 = population[i1]
            p2 = population[i2]

            # Uniform crossover
            child = [p1[j] if random.random() < 0.5 else p2[j] for j in range(24)]

            # Mutation
            for j in range(24):
                if random.random() < 0.15:
                    if j < 12:
                        # Mutate exponent: random walk
                        child[j] = max(0, min(40, child[j] + random.randint(-3, 3)))
                    else:
                        # Mutate coefficient index
                        child[j] = random.randint(0, len(coeff_choices)-1)

            new_population.append(child)

        # Evaluate new population
        population = new_population
        scores = []
        for chrom in population:
            decoded = decode(chrom)
            if decoded is None:
                scores.append(999)
            else:
                scores.append(evaluate(decoded[0], decoded[1]))

        # Update best
        best_idx = int(np.argmin(scores))
        if scores[best_idx] < best_score:
            best_score = scores[best_idx]
            best_chrom = population[best_idx][:]
            decoded = decode(best_chrom)
            if decoded:
                candidate = make_poly(decoded[0], decoded[1])
                if is_valid(candidate):
                    best_count = best_score
                    best_poly = candidate

        # Early termination if we hit a very good score
        if best_count <= 13:
            break

    # ---------- Phase 3: Local refinement on best ----------
    if time.time() < deadline and is_valid(best_poly):
        work = best_poly[:]
        work_count = best_count
        max_deg = len(work)

        for _ in range(3000):
            if time.time() > deadline:
                break
            # Mutate 1-2 positions
            n_mut = 1 if random.random() < 0.6 else 2
            positions = random.sample(range(max_deg), n_mut)
            originals = [work[p] for p in positions]

            for p in positions:
                work[p] = random.choice([0, 1, -1, 2, -2, 3, -3])

            if not is_valid(work):
                for p, ov in zip(positions, originals):
                    work[p] = ov
                continue

            cnt, _ = square_count(work)
            if cnt < work_count:
                work_count = cnt
                if cnt < best_count:
                    best_count = cnt
                    best_poly = work[:]
            else:
                for p, ov in zip(positions, originals):
                    work[p] = ov

    # Final validation
    best_poly = trim(best_poly)
    if not is_valid(best_poly):
        return [1]*12
    return best_poly
