"""Genetic algorithm approach for covering design C(19,8,5)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design using a genetic algorithm with multi-objective fitness.
    Uses efficient bitset coverage tracking and targeted mutation.
    """
    time_limit = 110  # seconds, leaving buffer for evaluation
    
    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute coverage mask for each possible block
    # C(19,8) = 75582 blocks, each covers C(8,5) = 56 t-subsets
    # Use a dictionary mapping block tuple to coverage bitset
    block_coverage_cache = {}
    
    def get_block_coverage(block):
        """Get or compute the coverage bitset for a block."""
        bt = tuple(block)
        if bt in block_coverage_cache:
            return block_coverage_cache[bt]
        cov = np.zeros(num_tsubsets, dtype=bool)
        for ts in itertools.combinations(bt, t):
            idx = tsubset_to_idx[ts]
            cov[idx] = True
        block_coverage_cache[bt] = cov
        return cov
    
    def compute_coverage(blocks):
        """Compute union coverage of all blocks using numpy."""
        cov = np.zeros(num_tsubsets, dtype=bool)
        for block in blocks:
            cov |= get_block_coverage(block)
        return cov
    
    def fitness(blocks):
        """
        Multi-objective fitness: lower is better.
        Primary: number of uncovered t-subsets
        Secondary: number of blocks
        """
        cov = compute_coverage(blocks)
        uncovered_count = num_tsubsets - np.count_nonzero(cov)
        return uncovered_count * 10000 + len(blocks)
    
    def greedy_cover():
        """
        Greedy set cover: repeatedly pick the block covering the most uncovered t-subsets.
        """
        uncovered_mask = np.ones(num_tsubsets, dtype=bool)
        blocks = []
        # Precompute all block coverages
        all_blocks = list(itertools.combinations(range(v), k))
        block_covs = [get_block_coverage(list(b)) for b in all_blocks]
        
        while np.any(uncovered_mask):
            # Find block with maximum new coverage
            best_idx = -1
            best_gain = -1
            for i, bc in enumerate(block_covs):
                gain = np.count_nonzero(bc & uncovered_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_gain <= 0:
                break
            blocks.append(list(all_blocks[best_idx]))
            uncovered_mask &= ~block_covs[best_idx]
        return blocks
    
    def random_individual(min_blocks=280, max_blocks=380):
        """Generate a random individual with some greedy blocks for quality."""
        # Start with a few greedy blocks
        greedy_blocks = greedy_cover()
        num_greedy = min(len(greedy_blocks), 30)
        blocks = greedy_blocks[:num_greedy]
        
        # Add random blocks to fill up
        target_size = random.randint(min_blocks, max_blocks)
        seen = set(tuple(b) for b in blocks)
        while len(blocks) < target_size:
            b = tuple(sorted(random.sample(range(v), k)))
            if b not in seen:
                blocks.append(list(b))
                seen.add(b)
        return blocks
    
    # Initialize population
    population_size = 150
    population = []
    
    # Start with a good greedy solution
    greedy_sol = greedy_cover()
    population.append(greedy_sol)
    
    # Add diverse random individuals
    for _ in range(population_size - 1):
        population.append(random_individual())
    
    # Evaluate initial population
    fitnesses = [fitness(ind) for ind in population]
    best_idx = int(np.argmin(fitnesses))
    best_individual = population[best_idx]
    best_fitness = fitnesses[best_idx]
    
    start_time = time.time()
    max_generations = 200
    
    # Genetic algorithm parameters
    elite_size = 15
    tournament_size = 5
    crossover_rate = 0.7
    mutation_rate = 0.3
    
    for gen in range(max_generations):
        if time.time() - start_time > time_limit:
            break
        
        # Sort by fitness
        indexed_pop = sorted(zip(fitnesses, population), key=lambda x: x[0])
        
        # Update best
        if indexed_pop[0][0] < best_fitness:
            best_fitness = indexed_pop[0][0]
            best_individual = indexed_pop[0][1]
        
        # Create new population
        new_population = []
        
        # Elitism
        for i in range(elite_size):
            new_population.append(indexed_pop[i][1])
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            parents = []
            for _ in range(2):
                candidates = random.sample(range(len(indexed_pop)), min(tournament_size, len(indexed_pop)))
                parent_idx = min(candidates, key=lambda i: indexed_pop[i][0])
                parents.append(indexed_pop[parent_idx][1])
            
            # Crossover: merge blocks from both parents, keeping unique
            if random.random() < crossover_rate:
                child = crossover(parents[0], parents[1])
            else:
                child = list(parents[0])
            
            # Mutation: targeted mutation focusing on uncovered areas
            if random.random() < mutation_rate:
                child = mutate(child)
            
            new_population.append(child)
        
        # Evaluate new population
        fitnesses = [fitness(ind) for ind in new_population]
        population = new_population
    
    return best_individual


def crossover(parent1, parent2):
    """
    Crossover: combine blocks from two parents, prioritizing blocks from the better parent.
    """
    # Assume parent1 is at least as good as parent2
    child = []
    seen = set()
    
    # Include most blocks from parent1
    for b in parent1:
        bt = tuple(b)
        if bt not in seen:
            child.append(b)
            seen.add(bt)
    
    # Add some blocks from parent2 for diversity
    parent2_blocks = [b for b in parent2 if tuple(b) not in seen]
    random.shuffle(parent2_blocks)
    num_add = min(len(parent2_blocks), len(parent1) // 4)
    for b in parent2_blocks[:num_add]:
        child.append(b)
        seen.add(tuple(b))
    
    random.shuffle(child)
    return child


def mutate(individual):
    """
    Mutation: replace some blocks with new random blocks.
    """
    n = len(individual)
    if n == 0:
        return individual
    
    num_mutations = max(1, n // 8)
    v = 19
    k = 8
    
    for _ in range(num_mutations):
        idx = random.randint(0, n - 1)
        new_block = sorted(random.sample(range(v), k))
        individual[idx] = new_block
    
    return individual
# EVOLVE-BLOCK-END
