"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
TOTAL = N * N

# EVOLVE-BLOCK-START
import itertools
import random
import time
from collections import defaultdict

import numpy as np

_DEADLINE_MARGIN = 30.0  # seconds before hard limit
_START_TIME = None
_BEST_RESULT = None


def _init_time():
    global _START_TIME
    _START_TIME = time.time()


def _time_remaining():
    return 330.0 - (time.time() - _START_TIME) - _DEADLINE_MARGIN


def _point_to_idx(p):
    return p[0] * N + p[1]


def _idx_to_point(i):
    return (i // N, i % N)


def _precompute_distances():
    """Precompute squared distances between all pairs of grid points."""
    pts = list(itertools.product(range(N), repeat=2))
    dist2 = np.zeros((TOTAL, TOTAL), dtype=np.int64)
    for i, a in enumerate(pts):
        for j, b in enumerate(pts):
            dist2[i, j] = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
    return pts, dist2


def _is_isosceles(i, j, k, dist2):
    """Check if points i, j, k form an isosceles triangle."""
    d1 = dist2[i, j]
    d2 = dist2[i, k]
    d3 = dist2[j, k]
    return d1 == d2 or d1 == d3 or d2 == d3


def _check_no_isosceles(chosen_indices, dist2):
    """Check if the set of chosen indices has no isosceles triangles."""
    n = len(chosen_indices)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _is_isosceles(chosen_indices[i], chosen_indices[j], chosen_indices[k], dist2):
                    return False
    return True


def _get_blocked_points(chosen_indices, dist2):
    """For each point not in chosen, check if it's blocked (forms isosceles with some pair in chosen)."""
    chosen_set = set(chosen_indices)
    blocked = set()
    for p in range(TOTAL):
        if p in chosen_set:
            continue
        for i in range(len(chosen_indices)):
            for j in range(i + 1, len(chosen_indices)):
                if _is_isosceles(p, chosen_indices[i], chosen_indices[j], dist2):
                    blocked.add(p)
                    break
            if p in blocked:
                break
    return blocked


def _is_maximal(chosen_indices, dist2):
    """Check if the set is maximal (every outside point is blocked)."""
    blocked = _get_blocked_points(chosen_indices, dist2)
    chosen_set = set(chosen_indices)
    for p in range(TOTAL):
        if p not in chosen_set and p not in blocked:
            return False
    return True


def _fitness(chosen_indices, dist2):
    """Compute fitness: lower is better. Returns (penalty, size)."""
    if not _check_no_isosceles(chosen_indices, dist2):
        return (1000 + len(chosen_indices), len(chosen_indices))
    if not _is_maximal(chosen_indices, dist2):
        # Penalize non-maximal: count unblocked points
        blocked = _get_blocked_points(chosen_indices, dist2)
        chosen_set = set(chosen_indices)
        unblocked_count = sum(1 for p in range(TOTAL) if p not in chosen_set and p not in blocked)
        return (500 + unblocked_count + len(chosen_indices), len(chosen_indices))
    return (0, len(chosen_indices))


def _greedy_construct(dist2):
    """Greedy construction to get an initial feasible solution."""
    chosen = []
    for p in range(TOTAL):
        if all(not _is_isosceles(p, chosen[i], chosen[j], dist2)
               for i in range(len(chosen)) for j in range(i + 1, len(chosen))):
            chosen.append(p)
    return chosen


def _repair_no_isosceles(indices, dist2):
    """Remove points to ensure no isosceles triangles. Greedy removal of points involved in most violations."""
    indices = list(indices)
    changed = True
    while changed:
        changed = False
        n = len(indices)
        if n < 3:
            break
        # Find all isosceles triangles
        violations = defaultdict(set)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if _is_isosceles(indices[i], indices[j], indices[k], dist2):
                        violations[indices[i]].add((j, k))
                        violations[indices[j]].add((i, k))
                        violations[indices[k]].add((i, j))
        if not violations:
            break
        # Remove the point with most violations
        worst = max(violations.keys(), key=lambda x: len(violations[x]))
        indices.remove(worst)
        changed = True
    return indices


def _ga_crossover(parent1, parent2, dist2):
    """Repair-based crossover: combine parents, then repair."""
    combined = list(set(parent1) | set(parent2))
    # Repair: remove points that create isosceles triangles
    repaired = _repair_no_isosceles(combined, dist2)
    return repaired


def _ga_mutation(individual, dist2, mutation_rate=0.1):
    """Mutate: randomly add or remove points, then repair."""
    ind = list(individual)
    n = len(ind)
    num_mutations = max(1, int(n * mutation_rate))
    for _ in range(num_mutations):
        if random.random() < 0.5 and len(ind) > 0:
            # Remove a random point
            idx = random.randint(0, len(ind) - 1)
            ind.pop(idx)
        else:
            # Add a random point
            p = random.randint(0, TOTAL - 1)
            if p not in ind:
                ind.append(p)
    # Repair
    repaired = _repair_no_isosceles(ind, dist2)
    return repaired


def _select_parents(population, fitnesses, tournament_size=3):
    """Tournament selection."""
    selected = []
    for _ in range(2):
        tournament = random.sample(range(len(population)), min(tournament_size, len(population)))
        best = min(tournament, key=lambda i: fitnesses[i])
        selected.append(population[best])
    return selected


def construct():
    _init_time()
    pts, dist2 = _precompute_distances()
    
    # Initial population from greedy constructions with different orderings
    population = []
    fitnesses = []
    
    # Greedy in natural order
    pop1 = _greedy_construct(dist2)
    population.append(pop1)
    fitnesses.append(_fitness(pop1, dist2))
    
    # Greedy in random order
    for seed in range(5):
        random.seed(seed)
        order = list(range(TOTAL))
        random.shuffle(order)
        chosen = []
        for p in order:
            if all(not _is_isosceles(p, chosen[i], chosen[j], dist2)
                   for i in range(len(chosen)) for j in range(i + 1, len(chosen))):
                chosen.append(p)
        population.append(chosen)
        fitnesses.append(_fitness(chosen, dist2))
    
    # Keep best
    best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
    best_individual = list(population[best_idx])
    best_fitness = fitnesses[best_idx]
    
    # Genetic Algorithm
    pop_size = 20
    while _time_remaining() > 5.0:
        # Selection and crossover
        new_population = []
        for _ in range(pop_size):
            p1, p2 = _select_parents(population, fitnesses)
            child = _ga_crossover(p1, p2, dist2)
            new_population.append(child)
        
        # Mutation
        for i in range(len(new_population)):
            if random.random() < 0.7:
                new_population[i] = _ga_mutation(new_population[i], dist2)
        
        # Evaluate
        new_fitnesses = [_fitness(ind, dist2) for ind in new_population]
        
        # Elitism: keep best from old population
        best_in_old = min(range(len(population)), key=lambda i: fitnesses[i])
        
        # Combine and select top pop_size
        combined = population + new_population
        combined_fit = fitnesses + new_fitnesses
        combined_indices = list(range(len(combined)))
        combined_indices.sort(key=lambda i: combined_fit[i])
        
        # Keep best and fill rest
        population = [combined[i] for i in combined_indices[:pop_size]]
        fitnesses = [combined_fit[i] for i in combined_indices[:pop_size]]
        
        # Update best
        current_best = min(range(len(population)), key=lambda i: fitnesses[i])
        if fitnesses[current_best] < best_fitness:
            best_fitness = fitnesses[current_best]
            best_individual = list(population[current_best])
    
    # Convert to list of (x, y) tuples
    result = [_idx_to_point(i) for i in best_individual]
    return result
# EVOLVE-BLOCK-END
