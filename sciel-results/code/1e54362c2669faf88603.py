"""Genetic algorithm approach for covering design C(19,8,5)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design using a highly optimized genetic algorithm.
    Uses precomputed bitmasks for fast coverage calculation.
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # 1. Precomputation
    # Map each t-subset to an index
    all_tsubsets = list(itertools.combinations(range(v), t))
    num_tsubsets = len(all_tsubsets)
    tsubset_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Number of 64-bit words needed to represent num_tsubsets bits
    num_words = (num_tsubsets + 63) // 64
    
    # Precompute coverage bitmask for every possible block
    # A block is a tuple of k distinct integers
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # coverage_masks[b] is a numpy array of uint64, length num_words
    # coverage_masks[b][w] has bit i set if t-subset index (w*64 + i) is covered by block b
    coverage_masks = np.zeros((num_blocks, num_words), dtype=np.uint64)
    
    # Generate masks
    for b_idx, block in enumerate(all_blocks):
        block_set = set(block)
        # Iterate through all t-subsets of this block
        for ts in itertools.combinations(block, t):
            idx = tsubset_to_idx[ts]
            word_idx = idx // 64
            bit_idx = idx % 64
            coverage_masks[b_idx, word_idx] |= (1 << bit_idx)
            
    # Total number of bits to consider (mask out extra bits in the last word)
    last_word_mask = (1 << (num_tsubsets % 64)) - 1 if num_tsubsets % 64 != 0 else (2**64 - 1)
    if num_tsubsets % 64 == 0 and num_tsubsets > 0:
         last_word_mask = (1 << 64) - 1 # All bits valid if it divides evenly? No, if it divides evenly, we used exactly num_words.
         # Actually, if num_tsubsets is divisible by 64, the last word is fully used.
         # If not, we mask the last word.
    if num_tsubsets % 64 != 0:
        last_word_mask = (1 << (num_tsubsets % 64)) - 1
    else:
        last_word_mask = (1 << 64) - 1 if num_words > 0 else 0
        # Note: if num_tsubsets is 0, num_words is 0. But here num_tsubsets > 0.
        
    # Helper to calculate fitness (number of uncovered t-subsets)
    def get_fitness(individual):
        """
        individual: list of block indices
        Returns: number of uncovered t-subsets
        """
        if not individual:
            return num_tsubsets
            
        # Start with zero coverage
        current_cov = np.zeros(num_words, dtype=np.uint64)
        
        # OR all block masks
        # Using a loop is often faster in Python for small num_words if we avoid array creation overhead, 
        # but num_words is ~182. 
        # Vectorized approach: stack masks and reduce with OR.
        # However, individual size varies.
        # Let's use a simple loop for OR, it's fast enough for 182 words.
        for b_idx in individual:
            current_cov |= coverage_masks[b_idx]
            
        # Count uncovered bits
        # Invert the coverage bits, mask with valid bits, count 1s
        # Note: ~current_cov will have 1s in the extra bits of the last word if not masked.
        # So we mask the last word before counting.
        
        # Create a mask array for valid bits
        valid_mask = np.full(num_words, (1 << 64) - 1, dtype=np.uint64)
        if num_words > 0:
            valid_mask[-1] = last_word_mask
            
        uncovered_bits = ~current_cov & valid_mask
        uncovered_count = np.sum(np.bit_count(uncovered_bits))
        
        return uncovered_count

    # Initialize Population
    pop_size = 300
    population = []
    
    # Add some random individuals
    for _ in range(pop_size):
        # Random number of blocks between 280 and 400
        n_blocks = random.randint(280, 400)
        # Sample distinct block indices
        # To ensure distinct blocks in the individual, we sample without replacement
        # But if n_blocks > num_blocks, we cap it. num_blocks is 75k, so 400 is fine.
        indices = random.sample(range(num_blocks), n_blocks)
        population.append(indices)
        
    # Evaluate initial fitness
    fitnesses = [get_fitness(ind) for ind in population]
    
    best_fitness = min(fitnesses)
    best_individual = population[fitnesses.index(best_fitness)]
    
    # GA Parameters
    elite_size = 20
    crossover_rate = 0.8
    mutation_rate = 0.1
    max_generations = 10000 # Will break on time
    
    # Main GA Loop
    while time.time() - start_time < time_limit:
        # Sort population by fitness
        sorted_pop = sorted(zip(fitnesses, population), key=lambda x: x[0])
        
        # Update best
        if sorted_pop[0][0] < best_fitness:
            best_fitness = sorted_pop[0][0]
            best_individual = sorted_pop[0][1]
            
        # If we have a perfect cover (0 uncovered), we can stop
        if best_fitness == 0:
            break
            
        new_population = []
        
        # Elitism
        for i in range(elite_size):
            new_population.append(sorted_pop[i][1])
            
        # Generate Offspring
        while len(new_population) < pop_size:
            # Tournament Selection
            p1_idx = random.choice(range(min(10, len(sorted_pop))))
            p2_idx = random.choice(range(min(10, len(sorted_pop))))
            # Actually, better tournament:
            candidates = random.sample(range(len(sorted_pop)), 5)
            p1 = min(candidates, key=lambda i: sorted_pop[i][0])
            p2 = min(candidates, key=lambda i: sorted_pop[i][0])
            
            parent1 = sorted_pop[p1][1]
            parent2 = sorted_pop[p2][1]
            
            # Crossover
            if random.random() < crossover_rate:
                # Uniform crossover
                child = []
                len_p1 = len(parent1)
                len_p2 = len(parent2)
                # To keep size similar, we can sample from the union or just mix
                # Simple approach: for each position up to max(len1, len2), pick from p1 or p2
                # But individuals are sets of blocks (indices). Order doesn't matter for coverage, but for GA structure it does.
                # Let's treat them as lists.
                max_len = max(len_p1, len_p2)
                for i in range(max_len):
                    if random.random() < 0.5:
                        if i < len_p1:
                            child.append(parent1[i])
                    else:
                        if i < len_p2:
                            child.append(parent2[i])
                
                # Remove duplicates (block indices should be unique in an individual)
                child = list(set(child))
            else:
                child = list(parent1)
                
            # Mutation
            if random.random() < mutation_rate:
                # Replace some blocks with random ones
                num_mutations = max(1, len(child) // 20)
                for _ in range(num_mutations):
                    if child:
                        idx = random.randint(0, len(child) - 1)
                        child[idx] = random.randint(0, num_blocks - 1)
                        # Ensure no duplicates? 
                        # Duplicates don't hurt fitness (OR is idempotent), but they waste space.
                        # We can clean up after mutation or just allow it. 
                        # Allowing it is simpler and faster.
                        
            new_population.append(child)
            
        # Evaluate New Population
        fitnesses = [get_fitness(ind) for ind in new_population]
        population = new_population
        
    # Convert best individual (list of block indices) to list of blocks (lists of ints)
    result_blocks = [list(all_blocks[idx]) for idx in best_individual]
    return result_blocks

# EVOLVE-BLOCK-END
