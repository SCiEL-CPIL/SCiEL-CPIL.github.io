import itertools
import time
import numpy as np
import random

_DEADLINE = 115.0  # Safety margin for 120s limit

def construct(v, k, t):
    start_time = time.time()
    
    # 1. Precomputation
    # Map triples to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    n_t = len(all_t_subsets)
    ALL_ONES = (1 << n_t) - 1
    
    # Generate all k-subsets (blocks) and their coverage masks
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(all_k_subsets)
    
    block_masks = [0] * n_k
    for b_idx, block in enumerate(all_k_subsets):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[b_idx] = mask
        
    # Helper to count bits
    def popcount(x):
        return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

    # 2. Fast Greedy Initialization
    def greedy_construct():
        uncovered = ALL_ONES
        selected_mask = 0
        while uncovered:
            if time.time() - start_time > _DEADLINE * 0.2:
                break
            best_block = -1
            best_gain = 0
            # Iterate all blocks to find max gain
            # Optimization: If we have many blocks, this is O(n_k) per step.
            # 346k * 76 steps is ~26M iterations. Each is a bitwise AND and popcount.
            # This should be < 1-2 seconds in Python 3.10+.
            for b_idx in range(n_k):
                gain = popcount(uncovered & block_masks[b_idx])
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            if best_block == -1:
                break
            selected_mask |= (1 << best_block)
            uncovered &= ~block_masks[best_block]
        return selected_mask

    initial_sol = greedy_construct()
    
    # Check if initial solution is valid
    def get_fitness(sol_mask):
        if sol_mask == 0:
            return 10**9 # Invalid
        covered = 0
        num_blocks = 0
        # Iterate set bits
        m = sol_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            covered |= block_masks[idx]
            num_blocks += 1
            m ^= lsb
        uncovered_count = popcount(ALL_ONES ^ covered)
        if uncovered_count == 0:
            return num_blocks
        else:
            return 10**6 + num_blocks + uncovered_count # Penalize invalid heavily

    best_sol = initial_sol
    best_fit = get_fitness(best_sol)
    
    # 3. Genetic Algorithm
    POP_SIZE = 50
    # Initialize population
    population = [initial_sol]
    for _ in range(POP_SIZE - 1):
        # Mutate initial solution
        sol = initial_sol
        # Randomly flip some bits
        for _ in range(random.randint(5, 20)):
            if random.random() < 0.5:
                # Remove a block
                if sol:
                    # Pick random set bit
                    m = sol
                    idx = random.randrange(n_k)
                    # This is inefficient for picking random set bit. 
                    # Better: pick random index, if set, flip.
                    # Or just pick random index and flip.
                    pass 
            # Let's just use a simple mutation: flip random bits
            idx = random.randrange(n_k)
            sol ^= (1 << idx)
        population.append(sol)

    # Evaluate initial population
    fits = [get_fitness(s) for s in population]
    
    while time.time() < start_time + _DEADLINE - 5:
        # Sort by fitness (lower is better)
        sorted_idx = sorted(range(POP_SIZE), key=lambda i: fits[i])
        best_idx = sorted_idx[0]
        if fits[best_idx] < best_fit:
            best_fit = fits[best_idx]
            best_sol = population[best_idx]
        
        # If we have a valid solution, we can try to improve it
        # Stop condition: if best_fit is very low (e.g., < 70) and we've been stable?
        # No, just run until time.
        
        new_population = []
        
        # Elitism: keep top 2
        new_population.append(population[sorted_idx[0]])
        new_population.append(population[sorted_idx[1]])
        
        while len(new_population) < POP_SIZE:
            # Tournament selection
            def select():
                i1 = random.randrange(POP_SIZE)
                i2 = random.randrange(POP_SIZE)
                return i1 if fits[i1] < fits[i2] else i2
            
            p1 = select()
            p2 = select()
            
            # Crossover: Uniform
            child = 0
            m1 = population[p1]
            m2 = population[p2]
            # To speed up, we can iterate bits? No, just bitwise ops on big ints are fast.
            # Uniform crossover: for each bit, take from p1 or p2.
            # This is hard to do efficiently with big ints without iterating bits.
            # Alternative: "One-point" crossover on the list of set bits?
            # Or just use Mutation only? GA with only mutation is basically Simulated Annealing / Random Walk.
            # Let's try a different crossover: 
            # Take top half of bits from p1, bottom from p2? 
            # Since bits are indices 0..n_k-1, we can split the range.
            # But the "meaning" of the bit position is just the block ID.
            # A better crossover for set cover: 
            # Pick a random block index boundary? No.
            
            # Let's stick to Mutation-heavy approach or simple crossover.
            # Simple Crossover: 
            # child = (m1 & mask1) | (m2 & ~mask1) where mask1 has 1s in random positions.
            # Generating a random mask of n_k bits is expensive if done every time.
            
            # Let's just use Mutation. It's robust and simple.
            child = population[p1]
            
            # Mutation
            mut_rate = 0.1
            n_mutations = int(n_k * mut_rate)
            for _ in range(n_mutations):
                idx = random.randrange(n_k)
                child ^= (1 << idx)
                
            new_population.append(child)
            
        population = new_population
        fits = [get_fitness(s) for s in population]
        
        # Check if we should break early if we hit a theoretical lower bound? 
        # Lower bound is 69. If we get 69, we can stop.
        if best_fit <= 69:
            break

    # Convert best_sol mask to list of blocks
    result_blocks = []
    m = best_sol
    while m:
        lsb = m & -m
        idx = lsb.bit_length() - 1
        result_blocks.append(list(all_k_subsets[idx]))
        m ^= lsb
        
    # Final validity check and repair if needed (shouldn't happen if best_fit < 10^6)
    # If best_fit >= 10^6, it's invalid. We should return the greedy solution which was valid.
    if best_fit >= 10**6:
         m = initial_sol
         result_blocks = []
         while m:
             lsb = m & -m
             idx = lsb.bit_length() - 1
             result_blocks.append(list(all_k_subsets[idx]))
             m ^= lsb
             
    return result_blocks
