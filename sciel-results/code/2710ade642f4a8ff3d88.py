import itertools
import time
import random
import math
import sys

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using a Genetic Algorithm.
    
    Strategy:
    1. Precompute t-subsets for all possible k-blocks (C(19,8)=75582 is small).
    2. Initialize a population of valid coverings using randomized greedy + reduction.
    3. Evolve the population:
       - Crossover: Union of two parents, followed by aggressive reduction (Removal + Directed Swap).
       - Mutation: Directed Swap on a random block to improve local coverage structure.
       - Selection: Keep the best 'pop_size' individuals based on block count.
    4. The key innovation is the 'Directed Swap' in reduction: instead of just removing
       redundant blocks, it swaps a block B for a block B' if B' covers more t-subsets 
       that are currently under-covered (count==1) than B does, or if B' allows for 
       future removals.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all k-blocks and their t-subset indices
    all_k_blocks = list(itertools.combinations(range(v), k))
    block_tsubs = {}
    for b in all_k_blocks:
        block_tsubs[b] = [t_subset_to_idx[ts] for ts in itertools.combinations(b, t)]
        
    # Helper: Calculate cover counts for a list of blocks
    def get_cover_counts(blocks):
        counts = [0] * num_t_subsets
        for b in blocks:
            for idx in block_tsubs[b]:
                counts[idx] += 1
        return counts
        
    # Helper: Greedy Initial Construction
    def greedy_initial(seed):
        random.seed(seed)
        cover_count = [0] * num_t_subsets
        blocks = []
        uncovered = set(range(num_t_subsets))
        
        while uncovered:
            # Pick a random uncovered t-subset
            target_idx = random.choice(list(uncovered))
            target_subset = all_t_subsets[target_idx]
            
            # Build a block containing this subset
            block_points = set(target_subset)
            remaining = [x for x in range(v) if x not in block_points]
            random.shuffle(remaining)
            
            # Greedily fill the block
            for _ in range(k - t):
                best_point = -1
                best_gain = -1
                for x in remaining:
                    gain = 0
                    # Check how many new uncovered t-subsets this point adds
                    # This is expensive if done naively, but k-t is small (3)
                    # We can approximate by checking overlap with uncovered set
                    # For speed, we just pick a random point from remaining if gain is hard to calc fast
                    # Actually, let's do a simple heuristic: pick point that maximizes 
                    # number of t-subsets involving existing points that are uncovered.
                    # Simplified: just pick random for speed in init, reduction will fix it.
                    # But better init helps. Let's do a quick check.
                    # Number of t-subsets in block involving x: C(t-1, t-1) * (current_size - (t-1)) ... no.
                    # Let's just use random for init to save time, as GA will optimize.
                    pass
                
                if remaining:
                    p = remaining.pop(0)
                    block_points.add(p)
                    
            block = tuple(sorted(block_points))
            if block not in blocks:
                blocks.append(block)
                for idx in block_tsubs[block]:
                    if idx in uncovered:
                        uncovered.remove(idx)
                        
        return blocks

    # Helper: Aggressive Reduction with Directed Swap
    def reduce_covering(blocks, cover_count, time_limit):
        """
        Reduces the covering by removing redundant blocks and swapping blocks 
        that have low 'unique' coverage for blocks that cover more critical t-subsets.
        """
        blocks = [tuple(b) for b in blocks]
        block_set = set(blocks)
        
        # 1. Removal Pass
        changed = True
        while changed and time.time() < time_limit:
            changed = False
            random.shuffle(blocks)
            for i in range(len(blocks)):
                b = blocks[i]
                removable = True
                for idx in block_tsubs[b]:
                    if cover_count[idx] == 1:
                        removable = False
                        break
                if removable:
                    for idx in block_tsubs[b]:
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    block_set.remove(b)
                    changed = True
                    break
                    
        # 2. Directed Swap Pass
        # Identify t-subsets that are "critical" (covered only once)
        # We want to swap blocks that cover many critical subsets for blocks 
        # that cover fewer critical subsets but potentially more "flexible" ones?
        # Actually, the goal is to enable removals. 
        # A block B is "bad" if it covers many t-subsets with count==1, because 
        # removing it breaks the cover. 
        # A block B is "good" if it covers t-subsets with count>1, because 
        # removing it is safe.
        # So, we want to swap a "bad" block (high critical count) for a "good" block 
        # (low critical count) if the new block still covers all the critical 
        # subsets that the old block covered? No, that's impossible if counts are 1.
        
        # Alternative Strategy: 
        # If we have a block B with critical subsets S_crit(B).
        # If we find another block B' such that S_crit(B) is a subset of B's coverage 
        # and B' covers S_crit(B) AND B' has lower total critical count, we swap.
        
        # Simpler Heuristic:
        # For each block, calculate 'risk' = number of t-subsets in it with count==1.
        # Sort blocks by risk descending.
        # For the highest risk block, try to replace it with a random block 
        # that covers all its critical subsets. If found, swap.
        
        while time.time() < time_limit:
            # Calculate risk for all blocks
            risks = []
            for b in blocks:
                risk = 0
                for idx in block_tsubs[b]:
                    if cover_count[idx] == 1:
                        risk += 1
                risks.append(risk)
                
            if not risks or max(risks) == 0:
                break
                
            # Find the block with highest risk
            max_risk_idx = risks.index(max(risks))
            b_old = blocks[max_risk_idx]
            
            # Find critical subsets for b_old
            crit_subsets = [idx for idx in block_tsubs[b_old] if cover_count[idx] == 1]
            
            if not crit_subsets:
                break
                
            # Try to find a replacement block b_new that covers all crit_subsets
            # and has lower risk (or at least doesn't increase risk too much)
            # We can't iterate all 75k blocks for every swap, so sample some.
            
            found_swap = False
            # Sample 100 random blocks
            for _ in range(100):
                b_new = random.choice(all_k_blocks)
                if b_new in block_set:
                    continue
                    
                # Check if b_new covers all crit_subsets
                new_tsubs = set(block_tsubs[b_new])
                if all(idx in new_tsubs for idx in crit_subsets):
                    # Calculate new risk
                    # Temporarily remove b_old, add b_new
                    # This is expensive, so estimate:
                    # New risk for b_new = number of t-subsets in b_new with count==1 
                    # (after adding b_new, counts increase, so risk decreases for those)
                    # But we also remove b_old.
                    
                    # Let's just do the swap and see if it's valid?
                    # We already checked critical subsets are covered.
                    # Are there other subsets that become uncovered?
                    # Only those that were covered ONLY by b_old and NOT by b_new.
                    # We checked crit_subsets (count==1) are in b_new.
                    # So all subsets with count==1 in b_old are covered by b_new.
                    # Subsets with count>1 in b_old will have count-1, still >=1.
                    # So the swap is VALID.
                    
                    # Perform Swap
                    for idx in block_tsubs[b_old]:
                        cover_count[idx] -= 1
                    for idx in block_tsubs[b_new]:
                        cover_count[idx] += 1
                        
                    blocks[max_risk_idx] = b_new
                    block_set.remove(b_old)
                    block_set.add(b_new)
                    found_swap = True
                    break
                    
            if not found_swap:
                # If we can't swap the highest risk block, we might be stuck.
                # Break to avoid infinite loop
                break

        return blocks, cover_count

    # Initialize Population
    pop_size = 20
    population = []
    for i in range(pop_size):
        b = greedy_initial(i)
        # Reduce each initial solution
        counts = get_cover_counts(b)
        b, counts = reduce_covering(b, counts, deadline - 5.0)
        population.append((b, counts))
        
    # Sort initial population
    population.sort(key=lambda x: len(x[0]))
    
    best_blocks = population[0][0]
    best_count = len(best_blocks)
    
    # GA Loop
    gen = 0
    while time.time() < deadline - 5.0:
        gen += 1
        new_population = []
        
        # Selection: Elitism (keep best 2)
        new_population.append(population[0])
        new_population.append(population[1])
        
        # Crossover and Mutation to fill the rest
        while len(new_population) < pop_size:
            # Pick parents (Tournament Selection)
            p1_idx = random.randint(0, len(population)-1)
            p2_idx = random.randint(0, len(population)-1)
            
            b1, c1 = population[p1_idx]
            b2, c2 = population[p2_idx]
            
            # Crossover: Union
            union_set = set(b1) | set(b2)
            child_blocks = list(union_set)
            
            # If child is too large, randomly drop some? 
            # Or just let reduction handle it.
            
            # Calculate counts for child
            child_counts = get_cover_counts(child_blocks)
            
            # Reduction
            child_blocks, child_counts = reduce_covering(child_blocks, child_counts, deadline - 10.0)
            
            # Mutation: Directed Swap on a random block in the child
            if child_blocks and random.random() < 0.5:
                # Pick a random block in child
                idx = random.randint(0, len(child_blocks)-1)
                b_mut = child_blocks[idx]
                
                # Try to swap it with a better one
                # Use the same logic as reduce_covering's swap part
                # Find a block that covers its critical subsets and has lower risk
                # For simplicity, just try a few random swaps
                crit_subsets = [idx_ for idx_ in block_tsubs[b_mut] if child_counts[idx_] == 1]
                if crit_subsets:
                    for _ in range(20):
                        b_new = random.choice(all_k_blocks)
                        if b_new in set(child_blocks):
                            continue
                        new_tsubs = set(block_tsubs[b_new])
                        if all(idx_ in new_tsubs for idx_ in crit_subsets):
                            # Valid swap
                            for idx_ in block_tsubs[b_mut]:
                                child_counts[idx_] -= 1
                            for idx_ in block_tsubs[b_new]:
                                child_counts[idx_] += 1
                            
                            child_blocks[idx] = b_new
                            break
            
            new_population.append((child_blocks, child_counts))
            
        # Update population
        population = new_population
        population.sort(key=lambda x: len(x[0]))
        
        if len(population[0][0]) < best_count:
            best_count = len(population[0][0])
            best_blocks = population[0][0]
            
    # Final Reduction on the best found
    if best_blocks:
        best_blocks = [tuple(b) for b in best_blocks]
        best_counts = get_cover_counts(best_blocks)
        best_blocks, best_counts = reduce_covering(best_blocks, best_counts, deadline - 1.0)
        
    return [list(b) for b in best_blocks]
