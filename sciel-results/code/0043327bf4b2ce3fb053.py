"""Genetic algorithm for kissing number in dimension 11."""
D = 11
import numpy as np
import time
import random


def construct():
    deadline = time.time() + 130  # safety margin under 150s

    # Build candidate pool: all vectors with small integer entries
    candidates = []
    # Weight-1: single nonzero coordinate
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s
            candidates.append(v)
    # Weight-2: two nonzero coordinates
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    candidates.append(v)
    # Weight-3: three nonzero coordinates (sample a subset to keep pool manageable)
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = [0] * D
                            v[i], v[j], v[k] = si, sj, sk
                            candidates.append(v)

    C = len(candidates)
    A = np.array(candidates, dtype=np.int64)
    # Precompute squared norms
    norms_sq = np.sum(A * A, axis=1)
    # Precompute pairwise squared distances
    # dist2[i,j] = ||A[i]-A[j]||^2
    # Use vectorized: dist2 = norm_i + norm_j - 2*A@A.T
    AA = A @ A.T
    dist2 = norms_sq[:, None] + norms_sq[None, :] - 2 * AA
    np.fill_diagonal(dist2, 0)
    # Symmetrize (should already be symmetric, but ensure)
    dist2 = (dist2 + dist2.T) // 2

    # A subset S is valid iff for all i in S: dist2[i,j] >= max(norms_sq[j] for j in S)
    # i.e., for all i,j in S: dist2[i,j] >= max_norm_sq

    def is_valid(mask):
        """Check if the subset indicated by boolean mask is valid."""
        if not np.any(mask):
            return True
        idx = np.where(mask)[0]
        if len(idx) <= 1:
            return True
        sub_norms = norms_sq[idx]
        max_n = int(sub_norms.max())
        sub_dist = dist2[np.ix_(idx, idx)]
        # All off-diagonal entries must be >= max_n
        return bool(np.all(sub_dist >= max_n))

    # Greedy seed: build a valid set greedily
    def greedy_build():
        mask = np.zeros(C, dtype=bool)
        # Order candidates by norm (smaller first tends to allow more)
        order = np.argsort(norms_sq)
        for ci in order:
            # Try adding ci
            mask[ci] = True
            if not is_valid(mask):
                mask[ci] = False
        return mask

    best_mask = greedy_build()
    best_size = int(best_mask.sum())

    # Genetic algorithm
    pop_size = 40
    population = [best_mask.copy() for _ in range(pop_size)]
    # Add some random initial members
    for _ in range(pop_size // 2):
        m = np.zeros(C, dtype=bool)
        # Randomly include some weight-1 and weight-2 vectors
        rand_idx = np.random.choice(C, size=min(30, C), replace=False)
        m[rand_idx] = True
        # Repair: remove violations
        for ci in rand_idx:
            if not is_valid(m):
                m[ci] = False
        population.append(m)
        if len(population) >= pop_size:
            break
    population = population[:pop_size]

    def fitness(mask):
        """Larger valid sets are better; invalid sets get penalty."""
        size = int(mask.sum())
        if is_valid(mask):
            return size
        return size - 1000  # penalty for invalid

    def repair(mask):
        """Remove violating vectors until valid."""
        changed = True
        while changed and is_valid(mask) is False:
            changed = False
            idx = np.where(mask)[0]
            if len(idx) == 0:
                break
            max_n = int(norms_sq[idx].max())
            # Find a vector that violates
            sub_dist = dist2[np.ix_(idx, idx)]
            viol = (sub_dist < max_n)
            if not np.any(viol):
                break
            # Pick a violating pair, remove one
            rows, cols = np.where(viol)
            if len(rows) == 0:
                break
            ri, ci = rows[0], cols[0]
            # Remove the one with larger norm to be safe
            if norms_sq[idx[ri]] >= norms_sq[idx[ci]]:
                mask[idx[ri]] = False
            else:
                mask[idx[ci]] = False
            changed = True
        return mask

    def crossover(m1, m2):
        # Uniform crossover
        r = np.random.rand(C) < 0.5
        child = np.where(r, m1, m2)
        return child

    def mutate(mask, rate=0.02):
        m = mask.copy()
        # Flip some bits
        flip_idx = np.where(np.random.rand(C) < rate)[0]
        for fi in flip_idx:
            m[fi] = not m[fi]
        # Repair
        m = repair(m)
        return m

    best_fitness = fitness(best_mask)

    while time.time() < deadline:
        new_pop = []
        # Elitism: keep best
        new_pop.append(best_mask.copy())

        for _ in range(pop_size - 1):
            # Tournament selection
            i1 = random.randrange(len(population))
            i2 = random.randrange(len(population))
            if fitness(population[i1]) >= fitness(population[i2]):
                p1 = population[i1]
            else:
                p1 = population[i2]
            i3 = random.randrange(len(population))
            i4 = random.randrange(len(population))
            if fitness(population[i3]) >= fitness(population[i4]):
                p2 = population[i3]
            else:
                p2 = population[i4]

            child = crossover(p1, p2)
            if np.random.rand() < 0.7:
                child = mutate(child)
            else:
                child = repair(child)
            new_pop.append(child)

        population = new_pop

        # Update best
        for m in population:
            f = fitness(m)
            if f > best_fitness:
                best_fitness = f
                best_mask = m.copy()
                best_size = int(best_mask.sum())

        # Inject diversity occasionally
        if len(population) > 1:
            random_idx = random.randrange(len(population))
            population[random_idx] = greedy_build()

    # Final: try to add more vectors to best_mask via local search
    order = np.argsort(norms_sq)
    for ci in order:
        if not best_mask[ci]:
            best_mask[ci] = True
            if not is_valid(best_mask):
                best_mask[ci] = False

    # Convert mask to list of vectors
    idx = np.where(best_mask)[0]
    result = A[idx]
    return result
