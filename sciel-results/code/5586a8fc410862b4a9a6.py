"""Genetic algorithm approach for covering design C(19,8,5)."""
import itertools
import random
import numpy as np
from collections import Counter
import time

def construct(v, k, t):
    """
    Construct a covering design using a genetic algorithm:
    1. Initialize population with greedy + random solutions
    2. Evolve via selection, crossover (combining blocks), mutation (swapping blocks)
    3. Return best solution found
    """
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute for each k-subset which t-subsets it covers
    # Use a bitset representation for speed
    # C(19,8) = 75582 blocks, C(19,5) = 11628 t-subsets
    # Each block covers 56 t-subsets
    
    # Generate all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map each block to its index
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # Precompute coverage: for each block, which t-subsets it covers
    # Use numpy bitsets for efficient operations
    block_coverage = np.zeros((num_blocks, num_t_subsets), dtype=np.uint8)
    for bi, block in enumerate(all_blocks):
        for ts in itertools.combinations(block, t):
            idx = t_subset_to_idx[ts]
            block_coverage[bi, idx] = 1
    
    # Greedy initialization function
    def greedy_cover():
        uncovered = np.ones(num_t_subsets, dtype=bool)
        blocks = []
        
        while uncovered.any():
            # Pick first uncovered t-subset
            target_idx = int(np.argmax(uncovered))
            target_ts = all_t_subsets[target_idx]
            
            # Find all blocks containing this t-subset
            target_set = set(target_ts)
            candidates = []
            for bi, block in enumerate(all_blocks):
                if target_set.issubset(set(block)):
                    candidates.append(bi)
            
            if not candidates:
                break
                
            # Pick block that covers most uncovered t-subsets
            best_bi = None
            best_count = -1
            for bi in candidates:
                count = int(np.sum(block_coverage[bi] & uncovered))
                if count > best_count:
                    best_count = count
                    best_bi = bi
            
            blocks.append(all_blocks[best_bi])
            uncovered = uncovered & ~block_coverage[best_bi].astype(bool)
        
        return blocks
    
    # Random initialization
    def random_cover():
        # Start with random blocks, then add more to cover gaps
        blocks = []
        covered = np.zeros(num_t_subsets, dtype=bool)
        
        # Add some random blocks
        num_initial = random.randint(200, 300)
        for _ in range(num_initial):
            bi = random.randint(0, num_blocks - 1)
            if bi not in [block_to_idx[b] for b in blocks]:
                blocks.append(all_blocks[bi])
                covered = covered | block_coverage[bi].astype(bool)
        
        # Fill gaps with greedy
        while not covered.all():
            uncovered = ~covered
            target_idx = int(np.argmax(uncovered))
            target_ts = all_t_subsets[target_idx]
            target_set = set(target_ts)
            
            candidates = []
            for bi, block in enumerate(all_blocks):
                if target_set.issubset(set(block)):
                    candidates.append(bi)
            
            if not candidates:
                break
                
            best_bi = None
            best_count = -1
            for bi in candidates:
                count = int(np.sum(block_coverage[bi] & uncovered))
                if count > best_count:
                    best_count = count
                    best_bi = bi
            
            blocks.append(all_blocks[best_bi])
            covered = covered | block_coverage[best_bi].astype(bool)
        
        return blocks
    
    # Evaluate a covering: returns (num_blocks, is_valid, uncovered_count)
    def evaluate(blocks):
        if not blocks:
            return (0, False, num_t_subsets)
        
        covered = np.zeros(num_t_subsets, dtype=bool)
        for block in blocks:
            bi = block_to_idx[tuple(block)]
            covered = covered | block_coverage[bi].astype(bool)
        
        uncovered_count = int(np.sum(~covered))
        is_valid = (uncovered_count == 0)
        return (len(blocks), is_valid, uncovered_count)
    
    # Crossover: combine blocks from two parents
    def crossover(parent1, parent2):
        # Take union of blocks from both parents, then trim
        combined = list(set(parent1) | set(parent2))
        
        # If too many blocks, remove redundant ones
        if len(combined) > max(len(parent1), len(parent2)):
            # Remove blocks that don't add coverage
            covered = np.zeros(num_t_subsets, dtype=bool)
            kept = []
            for block in combined:
                bi = block_to_idx[tuple(block)]
                if not (block_coverage[bi].astype(bool) & ~covered).any():
                    continue  # Skip if no new coverage
                kept.append(block)
                covered = covered | block_coverage[bi].astype(bool)
            combined = kept
        
        return combined
    
    # Mutation: swap a block with a better one
    def mutate(blocks, mutation_rate=0.3):
        new_blocks = list(blocks)
        num_mutations = int(len(new_blocks) * mutation_rate)
        
        for _ in range(num_mutations):
            if not new_blocks:
                break
            
            # Pick a block to replace
            i = random.randint(0, len(new_blocks) - 1)
            old_block = new_blocks[i]
            bi = block_to_idx[tuple(old_block)]
            
            # Find a replacement that adds coverage
            covered_without = np.zeros(num_t_subsets, dtype=bool)
            for j, block in enumerate(new_blocks):
                if j != i:
                    bj = block_to_idx[tuple(block)]
                    covered_without = covered_without | block_coverage[bj].astype(bool)
            
            # Find blocks that cover uncovered t-subsets
            uncovered = ~covered_without
            if not uncovered.any():
                continue
            
            target_idx = int(np.argmax(uncovered))
            target_ts = all_t_subsets[target_idx]
            target_set = set(target_ts)
            
            candidates = []
            for bi2, block in enumerate(all_blocks):
                if target_set.issubset(set(block)):
                    candidates.append(bi2)
            
            if not candidates:
                continue
            
            # Pick best candidate
            best_bi = None
            best_gain = -1
            for bi2 in candidates:
                gain = int(np.sum(block_coverage[bi2] & uncovered))
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi2
            
            if best_bi is not None and best_gain > 0:
                new_blocks[i] = all_blocks[best_bi]
        
        return new_blocks
    
    # Initialize population
    pop_size = 10
    population = []
    
    # Add greedy solution
    population.append(greedy_cover())
    
    # Add random solutions
    for _ in range(pop_size - 1):
        population.append(random_cover())
    
    # Track best solution
    best_blocks = None
    best_score = float('inf')
    
    # Evolution
    max_generations = 30
    time_limit = 100  # seconds
    
    start_time = time.time()
    
    for gen in range(max_generations):
        if time.time() - start_time > time_limit:
            break
        
        # Evaluate all individuals
        scored_pop = []
        for blocks in population:
            score, valid, uncov = evaluate(blocks)
            scored_pop.append((score, valid, uncov, blocks))
        
        # Update best
        for score, valid, uncov, blocks in scored_pop:
            if valid and score < best_score:
                best_score = score
                best_blocks = list(blocks)
        
        # Sort by fitness (fewer blocks is better, valid solutions preferred)
        scored_pop.sort(key=lambda x: (not x[1], x[0] + x[2] * 0.1))
        
        # Selection: keep top half
        elite = [x[3] for x in scored_pop[:pop_size // 2]]
        
        # Create next generation
        new_population = list(elite)
        
        while len(new_population) < pop_size:
            if time.time() - start_time > time_limit:
                break
            
            # Pick two parents from elite
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation
            if random.random() < 0.5:
                child = mutate(child)
            
            new_population.append(child)
        
        population = new_population
    
    # Final local search on best solution
    if best_blocks is not None:
        # Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(best_blocks)):
                block = best_blocks[i]
                bi = block_to_idx[tuple(block)]
                
                # Check if any t-subset in this block is not covered by others
                redundant = True
                for ts in itertools.combinations(block, t):
                    idx = t_subset_to_idx[ts]
                    # Check if covered by other blocks
                    covered_by_others = False
                    for j, other_block in enumerate(best_blocks):
                        if j == i:
                            continue
                        bj = block_to_idx[tuple(other_block)]
                        if block_coverage[bj, idx]:
                            covered_by_others = True
                            break
                    if not covered_by_others:
                        redundant = False
                        break
                
                if redundant:
                    best_blocks.pop(i)
                    changed = True
                    break
    
    # Return best solution found, or greedy if GA didn't find valid
    if best_blocks is not None:
        return best_blocks
    else:
        return greedy_cover()
