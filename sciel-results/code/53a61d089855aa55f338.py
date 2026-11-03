import itertools
import random
import time
import math
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Precompute all t-subsets (triples for t=3)
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_index = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute all possible k-blocks and their covered subset indices (masks)
    # For v=24, k=7, this is C(24,7) = 346,104 blocks.
    # Each block covers C(7,3) = 35 subsets.
    
    block_list = []
    block_masks = []
    block_triples = [] # Store list of triple indices for faster redundancy checks later? Maybe not needed if we use masks.
    
    # Optimization: Precompute bit for each triple
    triple_bit = {}
    for i, sub in enumerate(all_subsets):
        triple_bit[sub] = 1 << i
        
    # Generate all blocks and their masks
    # This is the most expensive part. ~346k * 35 ops.
    # We can speed this up by noting that many blocks share triples? No, just iterate.
    # To make it faster, we can precompute the mask for all C(7,3) subsets of a 7-set?
    # No, the indices are global.
    
    # Let's just do the loop.
    # We can potentially parallelize if we had multiprocessing, but let's stick to single thread for simplicity and safety.
    # 346k iterations is small.
    
    for block in itertools.combinations(range(v), k):
        mask = 0
        for sub in itertools.combinations(block, t):
            mask |= triple_bit[sub]
        block_list.append(block)
        block_masks.append(mask)
        
    n_blocks = len(block_list)
    
    # --- Search Strategy: Weighted Greedy + Simulated Annealing ---
    
    # 1. Initial Solution: Weighted Greedy
    # Standard greedy picks max gain.
    # Weighted greedy (or "Logarithmic Heuristic" variant) might pick based on gain / cost?
    # Here cost is 1 for all. So it's just max gain.
    # However, we can diversify the initial solution by using random tie-breaking or slight perturbations.
    
    def get_greedy_solution(rng_seed):
        rng = random.Random(rng_seed)
        uncovered_mask = (1 << n_subsets) - 1
        selected_indices = []
        used_indices = set()
        
        while uncovered_mask:
            best_gain = 0
            best_candidates = []
            
            # Optimization: Iterate all blocks.
            # To speed up, we can maintain a list of available blocks?
            # But removing from a list is O(N).
            # Checking `if i in used_indices` is O(1).
            # 346k checks per step * ~70 steps = 24M checks.
            # Each check: `gain = (uncovered_mask & block_masks[i]).bit_count()`
            # This is fast.
            
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
                break
                
            # Random tie-breaking for diversity
            chosen_idx = rng.choice(best_candidates)
            
            selected_indices.append(chosen_idx)
            used_indices.add(chosen_idx)
            uncovered_mask &= ~block_masks[chosen_idx]
            
        return selected_indices

    # 2. Local Search: Simulated Annealing
    # Moves:
    # a) Remove a redundant block (if exists).
    # b) Swap: Remove one block, add another to maintain coverage.
    #    This is complex to check validity quickly.
    # c) Add a block (if it reduces some other metric? No, we want to minimize count).
    #    Adding a block increases count, so only do if it allows removing 2+ blocks later?
    #    Standard SA for set cover usually allows adding/removing/swapping.
    
    # Let's implement a simpler local search:
    # "Greedy Deletion" followed by "Greedy Addition" (which is just re-greedy on the remaining uncovered? No).
    
    # Better Local Search for Set Cover:
    # 1. Check for redundancy: If a block is the *only* one covering a triple, it's essential.
    #    If no triple has count 1, the block is redundant? No, that's not sufficient.
    #    A block is redundant if removing it doesn't uncover any triple.
    #    i.e., for all triples in block, there is another block covering it.
    
    def is_redundant(idx, current_blocks_set):
        # current_blocks_set: set of indices of selected blocks
        # Check if any triple in block `idx` is covered ONLY by `idx`.
        mask = block_masks[idx]
        # We need to know if (mask & union_of_others) == mask?
        # i.e., is mask a subset of the union of all other blocks' masks?
        
        # Efficient way:
        # Compute union of all other blocks? Expensive.
        # Alternative: For each triple in `idx`, check if it's in any other block.
        # We can precompute for each triple, the list of blocks covering it.
        # triple_to_blocks: list of lists.
        # triple_to_blocks[i] = list of block indices covering triple i.
        
        # Let's build this once.
        # It takes memory but speeds up local search.
        # 2024 triples * ~35 blocks/triple? Average overlap.
        # Total entries: 346k * 35 = 12M.
        # This is feasible.
        
        # Actually, building this might take time.
        # Let's stick to the mask approach for simplicity first, or use the counts.
        
        # Let's use a simpler heuristic for local search:
        # Try removing each block. If valid, remove it.
        # To check validity fast:
        # We can maintain a `coverage_count` array for the current solution.
        
        return False # Placeholder, we will implement properly below

    # Let's build `triple_to_blocks` to enable fast redundancy checks and swaps.
    # triple_to_blocks[i] is a list of block indices that cover triple i.
    triple_to_blocks = [[] for _ in range(n_subsets)]
    
    for i, mask in enumerate(block_masks):
        # Iterate through set bits in mask
        # Or just iterate through the 35 triples of the block?
        # We have `block_list[i]`.
        for sub in itertools.combinations(block_list[i], t):
            idx = subset_index[sub]
            triple_to_blocks[idx].append(i)
            
    # Now we can do local search efficiently.
    
    def local_search(initial_blocks, rng, time_limit):
        # initial_blocks: list of block indices
        # Returns list of block indices
        
        current_blocks = list(initial_blocks)
        current_set = set(current_blocks)
        
        # Compute coverage counts for current solution
        # coverage_count[i] = number of blocks in current_set covering triple i
        coverage_count = [0] * n_subsets
        for b_idx in current_blocks:
            for sub in itertools.combinations(block_list[b_idx], t):
                coverage_count[subset_index[sub]] += 1
        
        # Check validity (should be valid if initial was)
        # valid = all(c > 0 for c in coverage_count)
        
        best_count = len(current_blocks)
        best_blocks = list(current_blocks)
        
        # Simulated Annealing parameters
        T = 10.0 # Initial temperature
        T_min = 0.1
        cooling_rate = 0.995
        
        while T > T_min and time.time() < time_limit:
            # Pick a random move
            # Move types:
            # 1. Delete: Remove a block. (Accept if still valid)
            # 2. Swap: Remove one, add one. (Accept if still valid and size doesn't increase? Or if size decreases?)
            #    If we swap 1 for 1, size is same. We want to reduce size.
            #    So Swap is only useful if we can then delete?
            #    Actually, standard set cover SA often allows "Add" and "Delete" and "Swap".
            #    If we add a block, size increases. We hope it enables 2 deletes later.
            #    This is complex.
            
            # Let's stick to:
            # 1. Try to delete a random block. If valid, accept (size decreases).
            # 2. Try to swap a random block with another random block. If valid, accept (size same, but might enable future deletions).
            #    Actually, if size is same, why accept? Only if it improves "potential" for deletion?
            #    Simpler: Just try deletions. If stuck, try a "2-opt" like move?
            #    Or just restart greedy with different seeds?
            
            # Let's try:
            # 1. Random Deletion: Pick random block from current_set.
            #    Check if removal keeps all triples covered (coverage_count[sub] > 1 for all sub in block).
            #    If yes, remove it. Update counts. Accept (improvement).
            # 2. Random Swap: Pick random block A from current_set, random block B from NOT current_set.
            #    Remove A, Add B. Check validity.
            #    If valid, accept? (Size unchanged).
            #    This might help escape local minima where a specific block is "blocking" a deletion.
            
            move_type = rng.choice(['delete', 'swap'])
            
            if move_type == 'delete':
                if not current_blocks: continue
                b_idx = rng.choice(current_blocks)
                
                # Check if b_idx is redundant
                # i.e., for all triples in b_idx, coverage_count > 1
                is_redundant = True
                for sub in itertools.combinations(block_list[b_idx], t):
                    if coverage_count[subset_index[sub]] == 1:
                        is_redundant = False
                        break
                        
                if is_redundant:
                    # Remove it
                    current_blocks.remove(b_idx)
                    current_set.remove(b_idx)
                    for sub in itertools.combinations(block_list[b_idx], t):
                        coverage_count[subset_index[sub]] -= 1
                    # Update best
                    if len(current_blocks) < best_count:
                        best_count = len(current_blocks)
                        best_blocks = list(current_blocks)
                    # Since it's an improvement, we don't need probabilistic acceptance
                else:
                    # Probabilistic acceptance? No, it's worse (invalid or same size but "worse" structure?).
                    # If invalid, reject.
                    # If valid but not redundant? That means removing it makes some triple count 1?
                    # No, if it's not redundant, removing it makes some triple count 0 (invalid) or 1?
                    # Wait, coverage_count is for the CURRENT set.
                    # If we remove b_idx, the new count for triple s is coverage_count[s] - 1.
                    # If coverage_count[s] == 1, new count is 0 -> Invalid.
                    # If coverage_count[s] > 1, new count >= 1 -> Valid.
                    # So "is_redundant" check above is exactly the validity check for deletion.
                    # So if not redundant, deletion is invalid. Reject.
                    pass

            elif move_type == 'swap':
                if not current_blocks: continue
                b_remove = rng.choice(current_blocks)
                # Pick b_add from all blocks not in current_set
                # To avoid O(N) scan for random pick, we can precompute list of all blocks and filter?
                # Or just pick a random index and check if it's in current_set?
                # N=346k. current_set size ~70.
                # Probability of collision is low.
                
                attempts = 0
                while attempts < 10:
                    b_add = rng.randrange(n_blocks)
                    if b_add not in current_set:
                        break
                    attempts += 1
                else:
                    continue # Failed to pick a valid b_add
                
                # Check if swap (remove b_remove, add b_add) is valid.
                # New coverage for triple s:
                # old_count = coverage_count[s]
                # delta = (1 if s in b_add else 0) - (1 if s in b_remove else 0)
                # new_count = old_count + delta
                # Valid if new_count > 0 for all s.
                
                # We only need to check triples affected by b_remove or b_add.
                # Triples in b_remove: count decreases by 1. Must be > 1 originally (so new > 0).
                # Triples in b_add: count increases by 1. Always > 0 (since old >= 0).
                # Triples in both: count unchanged.
                
                # So we only need to check:
                # For all triples in b_remove, coverage_count[s] > 1.
                
                valid_swap = True
                for sub in itertools.combinations(block_list[b_remove], t):
                    if coverage_count[subset_index[sub]] == 1:
                        valid_swap = False
                        break
                        
                if valid_swap:
                    # Perform swap
                    current_blocks.remove(b_remove)
                    current_set.remove(b_remove)
                    for sub in itertools.combinations(block_list[b_remove], t):
                        coverage_count[subset_index[sub]] -= 1
                        
                    current_blocks.append(b_add)
                    current_set.add(b_add)
                    for sub in itertools.combinations(block_list[b_add], t):
                        coverage_count[subset_index[sub]] += 1
                        
                    # Size is same. Do we accept?
                    # In SA, we accept worse moves with probability exp(-delta/T).
                    # Here delta = 0 (size same). But structure changed.
                    # Let's accept with some probability to explore.
                    # Or just accept always? It might lead to cycles.
                    # Let's accept with probability 0.5?
                    # Or better: Only accept if it enables a future deletion? Hard to know.
                    # Let's accept if rng.random() < 0.5.
                    if rng.random() < 0.5:
                        # Keep swap
                        pass
                    else:
                        # Revert swap
                        current_blocks.remove(b_add)
                        current_set.remove(b_add)
                        for sub in itertools.combinations(block_list[b_add], t):
                            coverage_count[subset_index[sub]] -= 1
                        
                        current_blocks.append(b_remove)
                        current_set.add(b_remove)
                        for sub in itertools.combinations(block_list[b_remove], t):
                            coverage_count[subset_index[sub]] += 1
                
            T *= cooling_rate
            
        return best_blocks

    # Main Loop
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple restarts of Greedy + Local Search
    restart_id = 0
    while time.time() < deadline - 10.0:
        # 1. Generate greedy solution
        init_blocks = get_greedy_solution(restart_id)
        
        # 2. Local search
        ls_start = time.time()
        ls_limit = min(ls_start + 5.0, deadline - 2.0) # Give 5s to local search per restart
        refined_blocks = local_search(init_blocks, random.Random(restart_id), ls_limit)
        
        count = len(refined_blocks)
        if count < best_count:
            best_count = count
            best_blocks = [block_list[i] for i in refined_blocks]
            
        restart_id += 1
        # If we found a very good solution, we can reduce time per restart?
        # Or just keep going.
        
    if best_blocks:
        return [list(b) for b in best_blocks]
    else:
        # Fallback
        return []
