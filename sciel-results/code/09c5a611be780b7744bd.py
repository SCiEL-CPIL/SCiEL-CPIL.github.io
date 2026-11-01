"""Genetic Algorithm for covering design C(24,7,3)."""
import itertools
import random
import time
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START
import numpy as np

# Precomputation for v=24, t=3
V = 24
T = 3
K = 7
TOTAL_TSETS = 2024 # C(24, 3)

# Map t-set to index
_tset_to_idx = {}
_all_tsets = list(itertools.combinations(range(V), T))
for i, ts in enumerate(_all_tsets):
    _tset_to_idx[ts] = i

# Precompute for each block (7-subset) the list of t-set indices it covers
# There are C(24,7) = 346104 possible blocks.
# We don't precompute ALL, but we can compute on the fly or cache.
# Given the speed requirement, computing the 35 t-sets for a block is fast.

def _block_to_tset_indices(block):
    """Return list of t-set indices covered by a block (tuple of 7 ints)."""
    # block is a tuple of 7 distinct ints
    # We need to generate all C(7,3) = 35 combinations
    # and map them to indices
    idxs = []
    for ts in itertools.combinations(block, T):
        idxs.append(_tset_to_idx[ts])
    return idxs

class GAIndividual:
    __slots__ = ['blocks', 'block_set', 'coverage_counts', 'uncovered_count', 'block_count']
    
    def __init__(self, blocks):
        self.blocks = blocks
        self.block_set = set(blocks)
        self.block_count = len(blocks)
        # Compute coverage
        self.coverage_counts = np.zeros(TOTAL_TSETS, dtype=np.int32)
        for b in blocks:
            for idx in _block_to_tset_indices(b):
                self.coverage_counts[idx] += 1
        self.uncovered_count = np.sum(self.coverage_counts == 0)

    def is_valid(self):
        return self.uncovered_count == 0

    def fitness(self):
        # We want to minimize block_count, but validity is a hard constraint.
        # Fitness for selection: Lower is better.
        # If invalid, penalty is huge.
        if self.uncovered_count > 0:
            return 100000 + self.block_count + self.uncovered_count * 10
        return self.block_count

def _fast_greedy_init(rng, time_limit):
    """
    Fast greedy initialization to get a valid covering.
    Returns list of blocks (tuples).
    """
    deadline = time.time() + time_limit
    uncovered = set(range(TOTAL_TSETS))
    blocks = []
    
    # Precompute gain for each possible block? No, too big.
    # Greedy: pick block that covers max uncovered.
    # To speed up, we can sample or use a heuristic.
    # For v=24, C(24,7) is ~346k. Checking all is ~346k * 35 ops ~ 12M ops.
    # This is feasible in ~1s in Python if optimized? Maybe 2-3s.
    # Let's try a random greedy first for speed, then refine.
    
    # Random Greedy:
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-set
        first = rng.choice(list(uncovered))
        # Find a random block containing this t-set
        # Or better: pick a block that covers 'first' and many others.
        
        # Let's construct a block containing the t-set of 'first'
        tset = _all_tsets[first]
        block_list = list(tset)
        remaining_slots = K - T
        
        # Fill remaining slots with elements that cover most uncovered sets
        # Candidates: all elements not in block_list
        candidates = [x for x in range(V) if x not in block_list]
        
        for _ in range(remaining_slots):
            best_x = None
            best_gain = -1
            # Evaluate each candidate
            for x in candidates:
                # Gain: number of uncovered t-sets formed by x + 2 from block_list
                gain = 0
                # We can approximate or compute exactly
                # Exact:
                for S in itertools.combinations(block_list, T-1):
                    if x not in S:
                        tset_new = tuple(sorted(S + (x,)))
                        idx = _tset_to_idx[tset_new]
                        if idx in uncovered:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block_list.append(best_x)
            candidates.remove(best_x)
            
        block = tuple(sorted(block_list))
        blocks.append(block)
        
        # Update uncovered
        for idx in _block_to_tset_indices(block):
            uncovered.discard(idx)
            
    return blocks

def _repair_individual(ind, rng):
    """
    If individual is invalid, add blocks to cover missing t-sets.
    """
    if ind.is_valid():
        return ind
        
    blocks = list(ind.blocks)
    # Rebuild coverage counts for efficiency in loop
    counts = ind.coverage_counts.copy()
    uncovered = np.where(counts == 0)[0]
    
    while len(uncovered) > 0:
        # Pick a random uncovered t-set
        first_idx = rng.choice(uncovered)
        tset = _all_tsets[first_idx]
        
        # Construct a block covering this tset
        block_list = list(tset)
        remaining_slots = K - T
        candidates = [x for x in range(V) if x not in block_list]
        
        # Greedy fill to cover more uncovered sets
        for _ in range(remaining_slots):
            best_x = None
            best_gain = -1
            for x in candidates:
                gain = 0
                for S in itertools.combinations(block_list, T-1):
                    if x not in S:
                        tset_new = tuple(sorted(S + (x,)))
                        idx = _tset_to_idx[tset_new]
                        if counts[idx] == 0:
                            gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            if best_x is None:
                break
            block_list.append(best_x)
            candidates.remove(best_x)
            
        block = tuple(sorted(block_list))
        
        # Add block if not already present
        if block not in ind.block_set:
            blocks.append(block)
            ind.block_set.add(block)
            # Update counts
            for idx in _block_to_tset_indices(block):
                counts[idx] += 1
            # Update uncovered
            uncovered = np.where(counts == 0)[0]
        else:
            # If block is already there, it shouldn't be uncovered.
            # This implies a bug in logic or duplicate handling.
            # Just break to avoid infinite loop.
            break
            
    # Create new individual
    new_ind = GAIndividual(blocks)
    return new_ind

def _prune_individual(ind, rng):
    """
    Remove redundant blocks to reduce count.
    """
    if not ind.is_valid():
        return ind
        
    blocks = list(ind.blocks)
    counts = ind.coverage_counts.copy()
    n = len(blocks)
    
    improved = True
    while improved:
        improved = False
        # Shuffle to avoid bias
        order = list(range(len(blocks)))
        rng.shuffle(order)
        
        for i in order:
            if i >= len(blocks):
                continue
            b = blocks[i]
            # Check if removing b leaves any t-set uncovered
            # Temporarily decrement
            b_idxs = _block_to_tset_indices(b)
            safe = True
            old_vals = []
            for idx in b_idxs:
                old_vals.append(counts[idx])
                counts[idx] -= 1
                if counts[idx] == 0:
                    safe = False
                    break
            
            if safe:
                # Remove block
                blocks.pop(i)
                # Restore counts (they are already decremented, which is correct for removal)
                # Wait, if we remove, we keep the decremented counts.
                # If we don't remove, we restore.
                pass
            else:
                # Restore counts
                for idx, val in zip(b_idxs, old_vals):
                    counts[idx] = val
                
        # Update individual
        if len(blocks) < ind.block_count:
            improved = True
            new_ind = GAIndividual(blocks)
            # Check validity again (should be valid if pruning logic was correct)
            if new_ind.is_valid():
                ind = new_ind
            else:
                # Should not happen if logic is sound
                break
                
    return ind

def construct(v, k, t):
    start_time = time.time()
    total_budget = 145.0 # Safety margin for 150s limit
    
    if v != 24 or k != 7 or t != 3:
        # Fallback for other parameters (though task is specific)
        # Use simple greedy
        rng = random.Random(42)
        blocks = _fast_greedy_init(rng, 10.0)
        return [list(b) for b in blocks]

    rng = random.Random(12345)
    
    # Phase 1: Initialization
    # Generate a few diverse initial individuals using fast greedy with different seeds
    pop_size = 20
    population = []
    
    # One strong greedy
    init_blocks = _fast_greedy_init(rng, 5.0)
    population.append(GAIndividual(init_blocks))
    
    # Add some random valid coverings?
    # Or just perturb the greedy one.
    # Let's create a diverse population by random restarts of greedy (short time)
    for i in range(pop_size - 1):
        if time.time() > start_time + 10.0:
            break
        b = _fast_greedy_init(rng, 0.5)
        ind = GAIndividual(b)
        if ind.is_valid():
            population.append(ind)
        else:
            # Repair it
            ind = _repair_individual(ind, rng)
            population.append(ind)
            
    # Ensure we have a valid best
    best_ind = min(population, key=lambda x: x.fitness())
    
    # Phase 2: Genetic Algorithm
    deadline = start_time + total_budget - 5.0
    gen = 0
    stagnation = 0
    
    while time.time() < deadline:
        gen += 1
        # Selection: Tournament
        def select_tournament():
            # Pick 3 random, return best
            candidates = rng.sample(population, 3)
            return min(candidates, key=lambda x: x.fitness())
        
        new_population = []
        
        # Elitism: keep top 2
        sorted_pop = sorted(population, key=lambda x: x.fitness())
        new_population.extend(sorted_pop[:2])
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
                
            p1 = select_tournament()
            p2 = select_tournament()
            
            # Crossover: Merge blocks from p1 and p2
            # We want to create a child that has a subset of blocks from parents
            # such that it is likely valid or close to valid.
            
            # Strategy: Start with a random mix of blocks from p1 and p2
            # Then repair.
            
            # Simple Crossover: Uniform crossover on block sets
            # But we need to reduce size.
            
            # Let's try: Take all blocks from p1, then try to replace some with blocks from p2
            # if they cover the same critical sets better?
            
            # Alternative: "Mating" two valid coverings.
            # Child = Union of blocks? No, that increases size.
            # Child = Intersection? No, that loses coverage.
            
            # Better: "Crossover" by picking a random block from p1, and a random block from p2.
            # If they are "compatible" (cover similar areas), we might combine them.
            
            # Let's use a "Merge and Prune" strategy.
            # 1. Start with p1's blocks.
            # 2. Randomly replace some blocks with blocks from p2.
            # 3. Repair.
            # 4. Prune.
            
            child_blocks = list(p1.blocks)
            block_set = set(child_blocks)
            
            # Try to inject blocks from p2
            p2_blocks = list(p2.blocks)
            rng.shuffle(p2_blocks)
            
            # Replace up to 20% of blocks
            num_replacements = max(1, len(child_blocks) // 5)
            for _ in range(num_replacements):
                if not child_blocks:
                    break
                # Pick a random block from child to remove
                idx = rng.randrange(len(child_blocks))
                removed_block = child_blocks[idx]
                child_blocks.pop(idx)
                block_set.discard(removed_block)
                
                # Pick a random block from p2 to add
                added_block = p2_blocks[rng.randrange(len(p2_blocks))]
                if added_block not in block_set:
                    child_blocks.append(added_block)
                    block_set.add(added_block)
            
            child_ind = GAIndividual(child_blocks)
            
            # Repair
            child_ind = _repair_individual(child_ind, rng)
            
            # Prune
            child_ind = _prune_individual(child_ind, rng)
            
            new_population.append(child_ind)
            
            # Update best
            if child_ind.fitness() < best_ind.fitness():
                best_ind = child_ind
                stagnation = 0
            else:
                stagnation += 1
                
            # If we found a very good one, we might want to clone it
            if child_ind.is_valid() and child_ind.block_count <= best_ind.block_count:
                new_population.append(child_ind)
                
        population = new_population
        
        # If stagnation is high, do a random restart or mutation
        if stagnation > 5 * pop_size:
            # Inject a fresh random greedy
            if time.time() < deadline - 5.0:
                fresh_blocks = _fast_greedy_init(rng, 2.0)
                fresh_ind = GAIndividual(fresh_blocks)
                fresh_ind = _repair_individual(fresh_ind, rng)
                fresh_ind = _prune_individual(fresh_ind, rng)
                population.append(fresh_ind)
                if fresh_ind.fitness() < best_ind.fitness():
                    best_ind = fresh_ind
                stagnation = 0
                population = population[:pop_size] # Keep size fixed

    # Final Result
    # Ensure best_ind is valid
    if not best_ind.is_valid():
        # Fallback to first valid found or greedy
        valid_inds = [ind for ind in population if ind.is_valid()]
        if valid_inds:
            best_ind = min(valid_inds, key=lambda x: x.block_count)
        else:
            # Emergency greedy
            emergency_blocks = _fast_greedy_init(rng, 10.0)
            return [list(b) for b in emergency_blocks]
            
    return [list(b) for b in best_ind.blocks]

# EVOLVE-BLOCK-END
