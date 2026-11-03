import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # 1. Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    blocks = list(itertools.combinations(points, k))
    n_blocks = len(blocks)
    
    # C(k, t)
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    # block_triples[i] contains the indices of the t-subsets covered by blocks[i]
    # Shape: (n_blocks, c_k_t)
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    
    # Also compute frequency of each triple to use for weighting
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            block_triples[i, idx] = t_idx
            triple_freq[t_idx] += 1
            idx += 1
            
    # Weights: 1 / frequency. Higher weight for rarer triples.
    base_weights = 1.0 / np.maximum(triple_freq, 1)
    
    best_solution = None
    best_size = float('inf')
    
    # Local search polish: try to reduce solution size
    def local_search_polish(selected_indices, block_triples, n_triples, c_k_t):
        """Try to reduce solution size by removing redundant blocks or swapping 2 for 1."""
        if len(selected_indices) <= 1:
            return selected_indices
        
        current = list(selected_indices)
        improved = True
        max_iterations = 100
        
        while improved and max_iterations > 0:
            improved = False
            max_iterations -= 1
            
            # Compute cover counts
            cover_count = np.zeros(n_triples, dtype=np.int32)
            for b_idx in current:
                cover_count[block_triples[b_idx]] += 1
            
            # 1. Try removing each block if redundant
            for i in range(len(current)):
                b_idx = current[i]
                b_triples = block_triples[b_idx]
                if np.min(cover_count[b_triples]) > 1:
                    # Remove this block
                    cover_count[b_triples] -= 1
                    current.pop(i)
                    improved = True
                    break
            
            # If we removed one, restart the loop to re-evaluate
            if improved:
                continue

            # 2. Try 2-for-1 swap: replace current[i] and current[j] with a single block that covers all their unique triples
            # Only do this if solution size > 2
            if len(current) > 2:
                found_swap = False
                # Limit swap search to save time
                # We look for pairs (i, j) such that there exists a block B where:
                # Unique(i) U Unique(j) is subset of B
                # To optimize, we can precompute "unique triples" for each block in the current solution
                # But that's expensive. Let's just try a limited number of random pairs or sequential pairs.
                
                # A faster heuristic: 
                # For each block i in current, find its "essential" triples (those with cover_count == 1).
                # If a block i has essential triples, can we find a block B that covers them AND covers the essential triples of some other block j, 
                # such that all other triples of i and j are covered by other blocks?
                
                # Simplified 2-for-1:
                # Iterate through all pairs (i, j) in current.
                # Check if there is a block B such that:
                # 1. B covers all triples in current[i] that are NOT covered by other blocks (essential_i)
                # 2. B covers all triples in current[j] that are NOT covered by other blocks (essential_j)
                # 3. B covers all other triples in current[i] and current[j] (which are covered by others anyway, so this is just checking if B exists that covers the union of essential sets? No, B must cover the union of ALL triples in i and j? No, B only needs to cover the triples that would become UNCOVERED if i and j are removed.)
                
                # Let's be precise:
                # If we remove i and j, the triples that become uncovered are:
                # U_ij = { x in block_triples[i] U block_triples[j] | cover_count[x] <= 2 and (x in block_triples[i] or x in block_triples[j]) and (cover_count[x] == 1 or (cover_count[x] == 2 and (x in block_triples[i] or x in block_triples[j]))) }
                # Actually, simpler:
                # U_ij = { x | x in block_triples[i] or x in block_triples[j] }
                # After removing i and j, a triple x is uncovered if:
                # x was in i or j, AND (x not in any other block in current)
                # i.e., cover_count[x] <= 2 AND (x in i OR x in j) AND (if cover_count[x]==2, it must be that the other cover is NOT in i or j? No, if cover_count is 2, and x is in i and j, removing both makes it 0. If x is in i and k (k!=j), removing i and j leaves k, so cover_count becomes 1. So x is safe if cover_count[x] >= 2 AND (x not in i OR x not in j OR cover_count[x] > 2)?
                
                # Let's stick to the standard definition:
                # A triple x is "critical" for the pair {i, j} if:
                # 1. x is in block_triples[i] or block_triples[j]
                # 2. The number of blocks in current that cover x is <= 2
                # 3. If the count is 2, then x must be in BOTH i and j? No. If x is in i and k (k != j), removing i and j leaves k. So x is covered.
                #    So x becomes uncovered ONLY if:
                #    - x is in i and NOT in any other block (count 1)
                #    - x is in j and NOT in any other block (count 1)
                #    - x is in i AND j AND NOT in any other block (count 2, but both are i and j)
                
                # So, Essential(i, j) = { x | x in (i U j) AND (cover_count[x] == 1 OR (cover_count[x] == 2 AND x in i AND x in j)) }
                
                # We need a block B such that Essential(i, j) is a subset of B.
                
                # Optimization:
                # Precompute for each block in current, the set of essential triples (count 1).
                # For pair (i, j), the essential set is Essential_i U Essential_j U (i INTERSECT j INTERSECT {x | cover_count[x]==2})
                
                # This is getting complex. Let's just try a simple version:
                # For each pair (i, j), compute the set of triples that would be uncovered.
                # Check if any block covers this set.
                
                # To keep it fast, we only check a subset of pairs or use a quick filter.
                # Filter: If len(current[i]) + len(current[j]) - len(intersection) is large, it's unlikely a single block covers it.
                # Max size of a block is k=7, so it covers C(7,3)=35 triples.
                # If the union of essential triples > 35, skip.
                
                # Let's implement this efficiently using bitsets? No, numpy sets.
                
                # For now, let's just do the removal step which is very effective.
                # The 2-for-1 is risky for time. Let's skip it in this iteration to ensure stability, 
                # but improve the greedy heuristic significantly.
                
                pass
        
        return current

    # Weighted greedy with dynamic weights (Deficit Heuristic)
    def dynamic_greedy_cover(base_weights, block_triples, n_triples):
        uncovered = np.ones(n_triples, dtype=bool)
        selected = []
        
        # Dynamic weights start at base_weights
        # As we select blocks, the "deficit" of uncovered triples increases.
        # We can simulate this by just using the static base weights but 
        # prioritizing blocks that cover "many" uncovered triples.
        # The "Adaptive Deficit" suggests that the value of a triple should increase 
        # if it has been uncovered for a long time.
        
        # A simpler approximation: 
        # At each step, compute gain = sum(weights[uncovered triples in block])
        # This is what the current code does.
        
        # To make it "adaptive", we can update the weights of uncovered triples to be higher 
        # if they are "harder" to cover (fewer candidate blocks left).
        # But we don't track candidate blocks.
        
        # Let's try: 
        # 1. Start with base_weights.
        # 2. At each step, select block maximizing gain.
        # 3. After selection, for all remaining uncovered triples, 
        #    if they are not covered by the selected block, multiply their weight by a factor > 1.
        #    This forces the algorithm to focus on the "leftovers".
        
        weights = base_weights.copy()
        
        while np.any(uncovered):
            # Compute gains
            # gain[i] = sum(weights[j] for j in block_triples[i] if uncovered[j])
            # Vectorized:
            # mask = uncovered[:, None] & (block_triples != -1) # block_triples is dense with valid indices
            # Actually block_triples is (n_blocks, c_k_t) with valid indices.
            # We can just do:
            weighted_uncovered = np.where(uncovered, weights, 0.0)
            gains = np.sum(weighted_uncovered[block_triples], axis=1)
            
            best_idx = np.argmax(gains)
            
            if gains[best_idx] == 0:
                break
            
            selected.append(best_idx)
            
            # Update uncovered
            uncovered[block_triples[best_idx]] = False
            
            # Adaptive Deficit: Increase weights of still-uncovered triples
            # This encourages covering the "hard" triples in subsequent steps
            # Factor: 1.1 or something small
            # Only for triples that are still uncovered
            weights[uncovered] *= 1.05
            
            # Normalize to prevent overflow? 
            # If we run 100 steps, 1.05^100 is ~130. Fine.
            
        return selected
    
    # Run multiple greedy passes
    n_restarts = 0
    while time.time() < deadline - 5.0:
        n_restarts += 1
        
        # Perturb base weights slightly for diversity
        if n_restarts == 1:
            perturbed_weights = base_weights.copy()
        else:
            noise = np.random.normal(0, 0.1, size=n_triples)
            perturbed_weights = base_weights * (1 + noise)
            perturbed_weights = np.maximum(perturbed_weights, 0.01)
        
        # Run dynamic greedy
        selected = dynamic_greedy_cover(perturbed_weights, block_triples, n_triples)
        
        # Polish
        selected = local_search_polish(selected, block_triples, n_triples, c_k_t)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_solution = [list(blocks[i]) for i in selected]
        
        # Early exit if we hit the lower bound (69)
        if best_size <= 69:
            break
    
    if best_solution is None:
        # Fallback: return a valid covering
        selected = dynamic_greedy_cover(base_weights, block_triples, n_triples)
        selected = local_search_polish(selected, block_triples, n_triples, c_k_t)
        best_solution = [list(blocks[i]) for i in selected]
        
    return best_solution
