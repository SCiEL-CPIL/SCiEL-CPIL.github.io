"""Genetic algorithm approach for covering design C(19,8,5)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

_CACHE = {}

def _get_precomputed(v, k, t):
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
        
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    all_kblocks = list(itertools.combinations(range(v), k))
    num_kblocks = len(all_kblocks)
    
    block_to_idx = {b: i for i, b in enumerate(all_kblocks)}
    idx_to_block = [list(b) for b in all_kblocks]
    
    block_coverages = []
    for b in all_kblocks:
        cov_indices = []
        for ts in itertools.combinations(b, t):
            cov_indices.append(tsubset_to_idx[ts])
        block_coverages.append(np.array(cov_indices, dtype=np.int32))
        
    _CACHE[key] = (all_tsubsets, num_tsubsets, tsubset_to_idx, block_to_idx, idx_to_block, block_coverages)
    return _CACHE[key]

def construct(v, k, t):
    all_tsubsets, num_tsubsets, tsubset_to_idx, block_to_idx, idx_to_block, block_coverages = _get_precomputed(v, k, t)
    
    population_size = 300
    max_generations = 100
    crossover_rate = 0.8
    mutation_rate = 0.3
    elite_size = 30
    tournament_size = 5
    time_limit = 100
    
    num_blocks_total = len(idx_to_block)

    def fitness(block_indices):
        cov = np.zeros(num_tsubsets, dtype=bool)
        for idx in block_indices:
            cov[block_coverages[idx]] = True
        uncovered_count = num_tsubsets - np.count_nonzero(cov)
        return uncovered_count * 10000 + len(block_indices)

    def random_block_idx():
        return random.randint(0, num_blocks_total - 1)

    def random_individual():
        num_blocks = random.randint(300, 400)
        blocks = set()
        while len(blocks) < num_blocks:
            blocks.add(random_block_idx())
        return list(blocks)

    def crossover(parent1, parent2):
        len1 = len(parent1)
        len2 = len(parent2)
        max_len = max(len1, len2)
        child = []
        for i in range(max_len):
            if i < len1 and i < len2:
                if random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            elif i < len1:
                child.append(parent1[i])
            elif i < len2:
                child.append(parent2[i])
        return list(set(child))

    def mutate(individual):
        if not individual:
            return individual
        n = len(individual)
        mutation_type = random.random()
        
        if mutation_type < 0.4:
            idx = random.randint(0, n - 1)
            individual[idx] = random_block_idx()
        elif mutation_type < 0.6:
            individual.append(random_block_idx())
        elif mutation_type < 0.8:
            idx = random.randint(0, n - 1)
            individual.pop(idx)
        else:
            individual.append(random_block_idx())
            if len(individual) > 1:
                individual.pop(random.randint(0, len(individual)-1))
        return individual

    population = [random_individual() for _ in range(population_size)]
    fitnesses = [fitness(ind) for ind in population]
    
    best_individual = min(population, key=lambda x: fitness(x))
    best_fitness = fitness(best_individual)
    
    start_time = time.time()
    
    for gen in range(max_generations):
        if time.time() - start_time > time_limit:
            break
        
        indexed_pop = sorted(zip(fitnesses, population), key=lambda x: x[0])
        
        if indexed_pop[0][0] < best_fitness:
            best_fitness = indexed_pop[0][0]
            best_individual = indexed_pop[0][1]
        
        new_population = []
        
        for i in range(elite_size):
            new_population.append(indexed_pop[i][1])
        
        while len(new_population) < population_size:
            parents = []
            for _ in range(2):
                candidates = random.sample(range(len(indexed_pop)), min(tournament_size, len(indexed_pop)))
                parent_idx = min(candidates, key=lambda i: indexed_pop[i][0])
                parents.append(indexed_pop[parent_idx][1])
            
            if random.random() < crossover_rate:
                child = crossover(parents[0], parents[1])
            else:
                child = list(parents[0])
            
            if random.random() < mutation_rate:
                child = mutate(child)
            
            child = list(set(child))
            new_population.append(child)
        
        fitnesses = [fitness(ind) for ind in new_population]
        population = new_population
    
    return [idx_to_block[i] for i in best_individual]

# EVOLVE-BLOCK-END
