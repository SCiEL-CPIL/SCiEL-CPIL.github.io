import itertools
import random
import time
import math
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # Precompute all t-subsets (triples for t=3)
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_index = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute all possible k-blocks and their covered subset indices as bitmasks
    # For v=24, k=7, C(24,7) = 346,104 blocks.
    # Each block covers C(7,3) = 35 subsets.
    # Using bitmasks for fast intersection and union operations.
    
    block_list = []
    block_masks = []
    
    # Optimization: Precompute the bit for each triple to avoid shifting in the inner loop
    triple_bit = {}
    for i, sub in enumerate(all_subsets):
        triple_bit[sub] = 1 << i
        
    # Generate all blocks and their masks
    # This is the most expensive part of precomputation.
    # 346k blocks * 35 triples = ~12M operations.
    # In Python, this might take 5-10 seconds.
    # We need to ensure this fits within the deadline.
    
    # If v, k, t are such that C(v,k) is too large, we might timeout.
    # For the specific task C(24,7,3), it is feasible.
    # We'll add a check to abort if precomputation takes too long, but it's unlikely.
    
    # To speed up generation, we can use itertools.combinations.
    for block in itertools.combinations(range(v), k):
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= triple_bit[sub]
        block_list.append(block)
        block_masks.append(mask)
        
    n_blocks = len(block_list)
    full_mask = (1 << n_subsets) - 1
    
    # Check if precomputation took too long
    if time.time() > deadline - 10.0:
        # Fallback to a simple greedy if precomputation ate too much time
        # But for v=24, it should be fine.
        pass

    def greedy_initialization(rng):
        """
        Generate an initial cover using randomized greedy.
        Returns a list of block indices.
        """
        uncovered_mask = full_mask
        selected_indices = []
        used_indices = set()
        
        while uncovered_mask:
            best_gain = 0
            best_candidates = []
            
            # Iterate through all blocks
            for i in range(n_blocks):
                if i in used_indices:
                    continue
                gain = (uncovered_mask & block_masks[i]).bit_count()
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [i]
                elif gain == best_gain and gain > 0:
                    best_candidates.append(i)
                    
            if not best_candidates:
                # Should not happen if a covering exists
                break
                
            # Randomly pick one to add diversity
            chosen_idx = rng.choice(best_candidates)
            
            selected_indices.append(chosen_idx)
            used_indices.add(chosen_idx)
            uncovered_mask &= ~block_masks[chosen_idx]
            
        return selected_indices

    def remove_redundant_blocks(indices):
        """
        Remove any block that is not essential for coverage.
        Returns a new list of indices.
        """
        if not indices:
            return []
            
        # Compute coverage counts for each triple
        coverage_counts = [0] * n_subsets
        
        # First pass: count coverage
        for idx in indices:
            block = block_list[idx]
            for sub in itertools.combinations(block, t):
                coverage_counts[subset_index[sub]] += 1
                
        # Second pass: filter out redundant blocks
        essential_indices = []
        for idx in indices:
            block = block_list[idx]
            is_essential = False
            for sub in itertools.combinations(block, t):
                if coverage_counts[subset_index[sub]] == 1:
                    is_essential = True
                    break
            if is_essential:
                essential_indices.append(idx)
                
        return essential_indices

    def local_search(indices, rng, max_iterations=5000):
        """
        Variable Neighborhood Descent (VND) / Local Search.
        Tries to swap out blocks for better ones to reduce the total count.
        
        Returns a list of block indices.
        """
        current_indices = list(indices)
        current_size = len(current_indices)
        
        # Build coverage counts and mapping of triple -> list of block indices covering it
        # This allows us to quickly find redundant blocks.
        
        # For efficiency, we maintain:
        # 1. coverage_counts: array of size n_subsets
        # 2. block_in_set: boolean array or set of indices in current solution
        
        coverage_counts = [0] * n_subsets
        in_set = [False] * n_blocks
        
        for idx in current_indices:
            in_set[idx] = True
            block = block_list[idx]
            for sub in itertools.combinations(block, t):
                coverage_counts[subset_index[sub]] += 1
        
        # Helper to get list of blocks covering a triple
        # This is expensive to build dynamically. Instead, we iterate.
        
        # VND Strategy:
        # 1. Find a redundant block (one where all its triples have count > 1).
        # 2. Try to remove it.
        # 3. If removing it creates uncovered triples, try to add a block that covers them.
        #    Actually, standard 1-swap is: Remove A, Add B.
        #    If we remove A, some triples might become uncovered (count drops to 0).
        #    We need B to cover ALL triples that became uncovered.
        #    And B must be distinct from A and not already in the set.
        
        # Better approach for speed:
        # 1. Identify all "critical" triples (count == 1) and "safe" triples (count > 1).
        # 2. A block is removable if it does NOT cover any critical triple.
        # 3. If we remove a block A, it becomes a candidate for removal.
        #    But wait, if A covers only safe triples, removing it is safe.
        #    However, we want to *reduce* the number of blocks.
        #    Simply removing a redundant block reduces size by 1.
        #    But is it possible to remove a block that covers critical triples and replace it with another?
        #    Yes. If A covers critical triple T, we must add B that also covers T.
        #    But B might cover other triples, making them safe.
        
        # Let's implement a simple 1-1 swap (remove one, add one) and 1-0 (remove one).
        
        # Optimization: Precompute for each block the list of triples it covers?
        # We already have `block_list` and `subset_index`.
        
        # Let's try a simpler heuristic first:
        # Just try to remove redundant blocks repeatedly until none remain.
        # Then try 1-1 swaps.
        
        # Phase 1: Remove redundant blocks
        changed = True
        while changed and time.time() < deadline - 2.0:
            changed = False
            # Find a redundant block
            # A block is redundant if it doesn't cover any triple with count == 1.
            # We can iterate through current blocks.
            for i, idx in enumerate(current_indices):
                block = block_list[idx]
                is_redundant = True
                for sub in itertools.combinations(block, t):
                    if coverage_counts[subset_index[sub]] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    # Remove it
                    in_set[idx] = False
                    for sub in itertools.combinations(block, t):
                        coverage_counts[subset_index[sub]] -= 1
                    current_indices.pop(i)
                    current_size -= 1
                    changed = True
                    break # Restart scan from beginning to catch new redundancies
            
        # Phase 2: 1-1 Swaps
        # Try to replace a block A with a block B such that the new set is smaller?
        # No, 1-1 swap keeps size same.
        # To reduce size, we need 2-1 swaps (remove 2, add 1) or 1-0 (remove 1).
        # 1-0 is handled in Phase 1.
        # 2-1 is complex.
        
        # Alternative: Randomized Restart with Greedy + Removal.
        # The current program does multiple restarts.
        # My "Materially Different" strategy:
        # Instead of just greedy + removal, use a **Simulated Annealing** or **Tabu Search** 
        # on the set of blocks? 
        # Or, more simply, a **Greedy with Lookahead**.
        
        # Let's try a different angle: **Iterated Local Search (ILS)**
        # 1. Get a good solution via Greedy + Removal.
        # 2. Perturb it (randomly swap some blocks).
        # 3. Local Search (Remove Redundant + 1-1 Swaps to improve quality? No, size is fixed in 1-1).
        #    Actually, for minimizing size, 1-1 swaps don't help unless they enable a subsequent 1-0 removal.
        
        # Let's implement a "Swap to Enable Removal" heuristic.
        # 1. Find a block A that is "almost" redundant (covers only 1 critical triple).
        # 2. Find a block B that covers that critical triple and is not in the set.
        # 3. Swap A for B.
        # 4. Check if any block becomes redundant.
        
        # This is getting complex. Let's stick to a robust **Multi-Start Greedy with Aggressive Pruning**.
        # The key insight from inspirations is that simple greedy clusters.
        # **Diverse Greedy**: When selecting the next block, penalize blocks that share many elements with already selected blocks?
        # Or, **Scarcity Weighted**: Prioritize blocks that cover rare triples.
        
        # Let's implement **Scarcity Weighted Greedy**.
        # In each step, for each candidate block, calculate gain.
        # But also consider the "scarcity" of the triples it covers.
        # Actually, the standard greedy already maximizes gain.
        
        # Let's try **Reverse Greedy (Minimum Last)**.
        # No, that's for finding a minimal cover, not minimum.
        
        # Let's go with **Iterated Local Search** where the local search is **Redundancy Removal** 
        # and the perturbation is **Random Block Swap**.
        
        # But wait, if I just swap random blocks, I might break the cover.
        # I must ensure the cover remains valid after perturbation.
        
        # Revised Local Search:
        # 1. Start with a valid cover.
        # 2. Loop:
        #    a. Try to remove a redundant block. If successful, continue.
        #    b. If no redundant block, try a 1-1 swap that might lead to a reduction?
        #       Hard to predict.
        #    c. If stuck, apply a perturbation:
        #       - Pick a random block A in the set.
        #       - Pick a random block B not in the set.
        #       - Replace A with B.
        #       - If the new set is a valid cover, keep it and go to (a).
        #       - If not, revert.
        #    d. If many failed perturbations, restart.
        
        # This is a valid strategy.
        
        # Implementation details for speed:
        # Checking validity of a set of blocks is O(N * 35) if we recompute counts.
        # We can maintain counts incrementally.
        
        # Let's implement this ILS.
        
        # We already did Phase 1 (Redundancy Removal) above.
        # Now, let's do Perturbation + Redundancy Removal.
        
        iterations = 0
        while time.time() < deadline - 2.0 and iterations < 10000:
            iterations += 1
            
            # Try to find a redundant block again (in case swaps created some)
            # We already did this in a while loop above, but let's do one pass.
            
            # Pick a random block to swap out
            if not current_indices:
                break
                
            idx_out = rng.choice(current_indices)
            
            # Pick a random block to swap in
            # To make it smart, pick a block that covers many uncovered triples?
            # No, all triples are covered.
            # Pick a block that is "diverse" from current set?
            # Let's just pick randomly from all blocks not in set.
            # To avoid bias, we can pick from a random subset.
            
            # For efficiency, let's just pick a random index from 0 to n_blocks-1
            # and check if it's in the set.
            
            attempts = 0
            while attempts < 100:
                idx_in = rng.randint(0, n_blocks - 1)
                if not in_set[idx_in]:
                    break
                attempts += 1
            else:
                continue # Couldn't find a block to swap in? Unlikely.
                
            # Perform swap
            # Remove idx_out
            in_set[idx_out] = False
            block_out = block_list[idx_out]
            for sub in itertools.combinations(block_out, t):
                coverage_counts[subset_index[sub]] -= 1
                
            # Add idx_in
            in_set[idx_in] = True
            block_in = block_list[idx_in]
            for sub in itertools.combinations(block_in, t):
                coverage_counts[subset_index[sub]] += 1
            
            # Check validity
            # A valid cover requires all coverage_counts > 0.
            # We can check if min(coverage_counts) > 0.
            # This is O(n_subsets) = 2024. Very fast.
            
            if min(coverage_counts) > 0:
                # Swap successful. Update current_indices
                # Find idx_out in current_indices and replace with idx_in
                for i, idx in enumerate(current_indices):
                    if idx == idx_out:
                        current_indices[i] = idx_in
                        break
                        
                # Try to remove redundant blocks
                # We can call a helper that does one pass of removal
                # Or just let the outer loop handle it?
                # Let's do a quick removal pass here.
                changed = True
                while changed:
                    changed = False
                    for i, idx in enumerate(current_indices):
                        block = block_list[idx]
                        is_redundant = True
                        for sub in itertools.combinations(block, t):
                            if coverage_counts[subset_index[sub]] == 1:
                                is_redundant = False
                                break
                        if is_redundant:
                            in_set[idx] = False
                            for sub in itertools.combinations(block, t):
                                coverage_counts[subset_index[sub]] -= 1
                            current_indices.pop(i)
                            current_size -= 1
                            changed = True
                            break
            else:
                # Revert swap
                # Add back idx_out
                in_set[idx_out] = True
                for sub in itertools.combinations(block_out, t):
                    coverage_counts[subset_index[sub]] += 1
                    
                # Remove idx_in
                in_set[idx_in] = False
                for sub in itertools.combinations(block_in, t):
                    coverage_counts[subset_index[sub]] -= 1
                    
                # Revert current_indices
                for i, idx in enumerate(current_indices):
                    if idx == idx_in:
                        current_indices[i] = idx_out
                        break
                        
        return current_indices

    # Main Search Loop
    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    
    # Run multiple restarts
    # Each restart: Greedy Init -> Local Search
    while time.time() < deadline - 5.0:
        # 1. Greedy Initialization
        initial_indices = greedy_initialization(rng)
        
        # 2. Local Search (ILS with Redundancy Removal and Swaps)
        refined_indices = local_search(initial_indices, rng)
        
        # 3. Final Redundancy Removal (just in case)
        final_indices = remove_redundant_blocks(refined_indices)
        
        current_count = len(final_indices)
        
        if current_count < best_count:
            best_count = current_count
            best_blocks = [block_list[i] for i in final_indices]
            
        # Optional: Check if we hit the lower bound?
        # Lower bound is 69. If we hit 69, we can stop.
        if best_count <= 69:
            break

    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        # Fallback
        return []

if __name__ == "__main__":
    # Test with specific parameters
    # Note: In the evaluator, construct(v, k, t) is called.
    # We just define the function.
    pass
