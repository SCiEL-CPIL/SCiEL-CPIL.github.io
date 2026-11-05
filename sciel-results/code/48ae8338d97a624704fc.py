import itertools
import random
import time
import math

def construct(v, k, t):
    # Precomputation
    all_points = range(v)
    all_t_subsets = list(itertools.combinations(all_points, t))
    num_t_subsets = len(all_t_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Generate all possible k-blocks
    all_k_blocks = list(itertools.combinations(all_points, k))
    num_k_blocks = len(all_k_blocks)
    
    # For each k-block, find which t-subsets it covers
    # block_covered_subsets[i] = list of subset ids covered by block i
    block_covered_subsets = []
    for b in all_k_blocks:
        b_set = set(b)
        covered = []
        for sid, s in enumerate(all_t_subsets):
            if all(x in b_set for x in s):
                covered.append(sid)
        block_covered_subsets.append(covered)
        
    # Greedy initialization
    def greedy_init():
        uncovered = set(range(num_t_subsets))
        blocks = []
        # Sort blocks by number of covered subsets? No, dynamic greedy.
        # To speed up, we can just pick random blocks that cover max uncovered.
        # But for init, let's do a simple greedy.
        # We need to know for each uncovered subset, which blocks cover it.
        # subset_to_blocks[sid] = list of block indices covering sid
        subset_to_blocks = [[] for _ in range(num_t_subsets)]
        for b_idx, covered in enumerate(block_covered_subsets):
            for sid in covered:
                subset_to_blocks[sid].append(b_idx)
                
        while uncovered:
            # Pick a random uncovered subset
            target_sid = random.choice(list(uncovered))
            # Pick a random block that covers it
            candidates = subset_to_blocks[target_sid]
            if not candidates:
                # Should not happen
                break
            # Choose the block that covers the most other uncovered subsets
            best_b = -1
            best_gain = -1
            # Randomize a bit to allow different inits
            random.shuffle(candidates)
            for b_idx in candidates:
                gain = 0
                for sid in block_covered_subsets[b_idx]:
                    if sid in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b = b_idx
                    
            if best_b != -1:
                blocks.append(best_b)
                for sid in block_covered_subsets[best_b]:
                    if sid in uncovered:
                        uncovered.remove(sid)
                        
        return blocks

    # Simulated Annealing
    def simulated_annealing(initial_blocks, time_limit=110):
        # State: list of block indices
        current_blocks = list(initial_blocks)
        # counts[sid] = number of blocks covering sid
        counts = [0] * num_t_subsets
        for b_idx in current_blocks:
            for sid in block_covered_subsets[b_idx]:
                counts[sid] += 1
        
        def get_energy(blocks, counts):
            uncovered = sum(1 for c in counts if c == 0)
            # Penalty for uncovered subsets should be high enough to prioritize coverage
            # but we want to minimize blocks.
            # If we allow uncovered, we might end up with invalid solution.
            # We must ensure final solution is valid.
            # So during SA, we should penalize uncovered heavily, or only accept valid states?
            # If we only accept valid states, we can only remove blocks if they don't uncover anything.
            # This is "Local Search on Valid Solutions".
            # Move: Remove block. If valid, accept. Else reject (or try to repair?).
            # Simpler: Only allow moves that keep the solution valid.
            # This is a strict local search.
            
            # Let's use a hybrid: Allow invalid states but with high penalty, 
            # and ensure we return a valid one at the end.
            # Actually, if we start with a valid solution, and we only remove blocks 
            # that are redundant, we stay valid.
            # If we add blocks, we stay valid.
            # So we can restrict moves to:
            # 1. Remove a block (if it doesn't uncover any subset).
            # 2. Add a block (always valid, but increases block count).
            # 3. Swap a block with another (check validity).
            
            # Let's stick to: 
            # - Try to remove a random block. If it causes uncovered > 0, reject.
            # - Try to add a random block. Accept if it allows removing another block? 
            #   This is complex.
            
            # Let's use the "Energy" with penalty approach but force validity at the end.
            # Penalty = 1000 * uncovered_count.
            return len(blocks) + 1000 * uncovered

        current_energy = get_energy(current_blocks, counts)
        best_blocks = list(current_blocks)
        best_energy = current_energy
        
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        start_time = time.time()
        
        # Pre-compute for speed?
        # We need to check if removing a block keeps counts > 0 for all its subsets.
        
        while time.time() - start_time < time_limit:
            # Generate move
            if random.random() < 0.7:
                # Try to remove a block
                if not current_blocks:
                    continue
                b_idx = random.choice(current_blocks)
                # Check if removal is valid
                # A block is redundant if for all sid in block_covered_subsets[b_idx], counts[sid] > 1
                redundant = True
                for sid in block_covered_subsets[b_idx]:
                    if counts[sid] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Apply move
                    current_blocks.remove(b_idx)
                    for sid in block_covered_subsets[b_idx]:
                        counts[sid] -= 1
                    new_energy = get_energy(current_blocks, counts)
                    # Always accept if valid and energy decreases (which it does by 1)
                    if new_energy < best_energy:
                        best_energy = new_energy
                        best_blocks = list(current_blocks)
                else:
                    # Try to swap? Or just reject?
                    # Let's try to swap with a random block that might cover the critical subsets
                    # For simplicity, reject for now. Or try to add a block?
                    pass
            else:
                # Try to add a block
                # This increases block count by 1, but might allow removal of others later?
                # In a strict "remove only" search, we don't add.
                # But if we get stuck, adding a block might allow a different removal path?
                # No, adding a block never helps remove another block directly in a set cover.
                # It only helps if we are doing a swap.
                
                # Let's implement a SWAP move.
                # Pick a block B1 in current. Pick a block B2 not in current.
                # Check if (Current - B1 + B2) is valid.
                if current_blocks:
                    b1 = random.choice(current_blocks)
                    b2 = random.choice(range(num_k_blocks))
                    if b2 == b1 or b2 in current_blocks:
                        continue
                    # Check validity of swap
                    # We need to ensure that for all sid covered by B1 but not B2, 
                    # there is another block covering it.
                    # And for all sid covered by B2, it's fine.
                    
                    # Simulate removal of B1
                    # counts_sim = counts - cover(B1)
                    # Then add B2: counts_sim + cover(B2)
                    # Check if any count becomes 0.
                    
                    # Optimization: 
                    # Critical subsets are those covered by B1 with count 1.
                    # If B2 does not cover them, swap is invalid.
                    critical = []
                    for sid in block_covered_subsets[b1]:
                        if counts[sid] == 1:
                            critical.append(sid)
                    
                    if critical:
                        # Check if B2 covers all critical subsets
                        b2_covers = set(block_covered_subsets[b2])
                        valid_swap = True
                        for sid in critical:
                            if sid not in b2_covers:
                                valid_swap = False
                                break
                        if not valid_swap:
                            continue
                    
                    # Apply swap
                    current_blocks.remove(b1)
                    for sid in block_covered_subsets[b1]:
                        counts[sid] -= 1
                    current_blocks.append(b2)
                    for sid in block_covered_subsets[b2]:
                        counts[sid] += 1
                    
                    new_energy = get_energy(current_blocks, counts)
                    delta = new_energy - current_energy
                    # Accept if better or with prob exp(-delta/T)
                    if delta < 0 or random.random() < math.exp(-delta / T):
                        current_energy = new_energy
                        if new_energy < best_energy:
                            best_energy = new_energy
                            best_blocks = list(current_blocks)
                    else:
                        # Revert
                        current_blocks.remove(b2)
                        for sid in block_covered_subsets[b2]:
                            counts[sid] -= 1
                        current_blocks.append(b1)
                        for sid in block_covered_subsets[b1]:
                            counts[sid] += 1
                        
            T *= cooling
            if T < T_min:
                T = T_min
                
        return best_blocks

    # Run
    best_blocks = None
    best_count = float('inf')
    
    start_time = time.time()
    # Do a few greedy inits
    for i in range(5):
        random.seed(i)
        blocks = greedy_init()
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks
            
    elapsed = time.time() - start_time
    remaining = 110 - elapsed
    
    if remaining > 0:
        # Run SA on the best greedy solution
        sa_blocks = simulated_annealing(best_blocks, time_limit=remaining)
        if len(sa_blocks) < best_count:
            best_count = len(sa_blocks)
            best_blocks = sa_blocks
            
    # Final cleanup: remove any redundant blocks
    final_blocks = list(best_blocks)
    # Rebuild counts
    counts = [0] * num_t_subsets
    for b_idx in final_blocks:
        for sid in block_covered_subsets[b_idx]:
            counts[sid] += 1
            
    changed = True
    while changed:
        changed = False
        for i in range(len(final_blocks)):
            b_idx = final_blocks[i]
            redundant = True
            for sid in block_covered_subsets[b_idx]:
                if counts[sid] <= 1:
                    redundant = False
                    break
            if redundant:
                for sid in block_covered_subsets[b_idx]:
                    counts[sid] -= 1
                final_blocks.pop(i)
                changed = True
                break
                
    # Convert indices to lists
    return [list(all_k_blocks[b_idx]) for b_idx in final_blocks]
