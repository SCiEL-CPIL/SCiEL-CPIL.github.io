import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute the list of t-subset indices covered by each block
    # This is critical for fast local search updates
    # Memory: 346104 * 35 * 4 bytes ~ 48MB. Acceptable.
    block_t_indices = []
    for block in all_blocks:
        idxs = np.array([t_subset_to_idx[ts] for ts in itertools.combinations(block, t)], dtype=np.int32)
        block_t_indices.append(idxs)

    # 2. Fast Greedy Initialization
    # We need a valid starting point.
    # Use the vectorized greedy from the parent, but simplified for speed.
    n_uint64s = (n_t + 63) // 64
    coverage_masks = np.zeros((n_blocks, n_uint64s), dtype=np.uint64)
    
    # Fill masks
    for i, block in enumerate(all_blocks):
        # Optimization: use the precomputed indices to set bits
        # But we need to set bits in uint64 array.
        # It's faster to just iterate the 35 triples for each block during init
        # 346k * 35 = 12M ops. ~1-2s in Python.
        # Let's try to vectorize mask creation if possible, or just accept the time.
        # Actually, we can skip full mask creation if we do greedy using indices?
        # Greedy needs "gain" = count of new triples covered.
        # If we maintain cover_counts, gain = len(block_t_indices[i]) - count(already covered in block).
        # This is O(35) per block check. 346k * 35 = 12M ops per greedy step.
        # Greedy takes ~70 steps. 70 * 12M = 840M ops. Too slow in Python loop.
        
        # So we stick to the bitmask approach for the initial greedy.
        pass

    # Efficient Mask Creation
    # We can create masks faster by iterating blocks and setting bits.
    # Let's just do it.
    for i in range(n_blocks):
        idxs = block_t_indices[i]
        for idx in idxs:
            word_idx = idx // 64
            bit_idx = idx % 64
            coverage_masks[i, word_idx] |= (1 << bit_idx)
            
    if time.time() > deadline - 20.0:
        # Fallback: simple greedy without local search
        covered_mask = np.zeros(n_uint64s, dtype=np.uint64)
        selected_blocks = []
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            if np.bitwise_count(covered_mask).sum() == n_t:
                break
            not_covered = ~covered_mask
            intersections = coverage_masks & not_covered
            gains = np.bitwise_count(intersections).sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_blocks.append(best_idx)
            covered_mask |= coverage_masks[best_idx]
        return [list(all_blocks[i]) for i in selected_blocks]

    # Run Greedy
    covered_mask = np.zeros(n_uint64s, dtype=np.uint64)
    selected_blocks = []
    for _ in range(200):
        if time.time() > deadline - 20.0:
            break
        if np.bitwise_count(covered_mask).sum() == n_t:
            break
        not_covered = ~covered_mask
        intersections = coverage_masks & not_covered
        gains = np.bitwise_count(intersections).sum(axis=1)
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        selected_blocks.append(best_idx)
        covered_mask |= coverage_masks[best_idx]
        
    # State for Local Search
    in_solution = np.zeros(n_blocks, dtype=bool)
    for b_idx in selected_blocks:
        in_solution[b_idx] = True
        
    cover_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in selected_blocks:
        cover_counts[block_t_indices[b_idx]] += 1
        
    n_uncovered = np.sum(cover_counts == 0)
    current_size = len(selected_blocks)
    
    # Objective: Minimize size + penalty * uncovered
    # We want to keep size low. If uncovered > 0, we must fix it.
    # Let's use a large penalty for uncovered triples.
    PENALTY = 1000
    
    def get_energy(size, uncovered):
        return size + PENALTY * uncovered
        
    current_energy = get_energy(current_size, n_uncovered)
    
    # Simulated Annealing
    # T starts high, decays.
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.99
    
    rng = random.Random(42)
    
    # Precompute list of blocks in solution for fast removal sampling
    # We can just use np.where(in_solution)[0] but that's slow.
    # Let's maintain a list.
    sol_list = list(selected_blocks)
    
    # To speed up random selection of a block IN solution or NOT in solution:
    # We can just pick a random index 0..n_blocks-1 and check in_solution.
    # Expected 1.5 tries.
    
    steps = 0
    while time.time() < deadline - 5.0:
        if T < T_min:
            break
            
        steps += 1
        
        # Choose a move: Add or Remove
        # Balance based on current state?
        # If uncovered > 0, bias towards Add.
        # If size is high, bias towards Remove.
        
        # Simple: 50% Add, 50% Remove
        is_add = (rng.random() < 0.5)
        
        # Pick a random block
        b_idx = rng.randint(0, n_blocks - 1)
        
        # If we want to add but it's already in, or remove but it's not, skip or flip?
        # Flipping is a valid move (Swap-like effect if we consider sequence, but here it's just toggle).
        # Actually, "Toggle" is a valid move.
        # Let's just toggle.
        
        if is_add and in_solution[b_idx]:
            # It's a remove move effectively
            # Or just skip?
            # Let's allow toggle.
            pass
            
        if (not is_add) and (not in_solution[b_idx]):
            # It's an add move effectively
            pass
            
        # Apply Move
        t_idxs = block_t_indices[b_idx]
        
        if in_solution[b_idx]:
            # Remove
            cover_counts[t_idxs] -= 1
            in_solution[b_idx] = False
            sol_list.remove(b_idx) # O(N) but N is ~70. Fast enough.
            new_size = current_size - 1
        else:
            # Add
            cover_counts[t_idxs] += 1
            in_solution[b_idx] = True
            sol_list.append(b_idx)
            new_size = current_size + 1
            
        new_uncovered = np.sum(cover_counts == 0)
        new_energy = get_energy(new_size, new_uncovered)
        
        delta = new_energy - current_energy
        
        # Accept?
        if delta <= 0:
            accept = True
        else:
            # Boltzmann probability
            # Avoid overflow in exp
            if delta / T > 50:
                accept = False
            else:
                accept = (rng.random() < np.exp(-delta / T))
                
        if accept:
            current_size = new_size
            n_uncovered = new_uncovered
            current_energy = new_energy
        else:
            # Revert
            if in_solution[b_idx]:
                # It was added, so revert remove
                cover_counts[t_idxs] -= 1
                in_solution[b_idx] = False
                if b_idx in sol_list:
                    sol_list.remove(b_idx)
            else:
                # It was removed, so revert add
                cover_counts[t_idxs] += 1
                in_solution[b_idx] = True
                sol_list.append(b_idx)
                
        T *= cooling_rate
        
        # If we have a valid solution (uncovered == 0), we can try to be more aggressive with removals
        # Or just let the energy function handle it.
        
    # Final Solution
    final_blocks = [list(all_blocks[i]) for i in sol_list]
    
    # Verify and Fix if invalid (should not happen if we started valid and accepted moves carefully, 
    # but SA can end in invalid state if T is too low and we got stuck in a local min of invalid states?
    # No, we started valid. We only accept worse moves with prob. 
    # If we end up with uncovered > 0, we need to repair.
    
    if n_uncovered > 0:
        # Repair: Greedily add blocks to cover uncovered
        # Find uncovered triples
        uncovered_idxs = np.where(cover_counts == 0)[0]
        
        # For each uncovered triple, find a block that covers it and is not in solution
        # Or just re-run a quick greedy repair on the current cover_counts?
        # Simple repair:
        # While there are uncovered triples:
        #   Pick a random uncovered triple
        #   Pick a random block that covers it
        #   Add it
        
        while n_uncovered > 0 and time.time() < deadline - 2.0:
            # Pick a random uncovered triple
            u_idx = rng.choice(uncovered_idxs)
            
            # Find blocks covering u_idx
            # We didn't build the inverse index. 
            # We can scan all blocks? Too slow.
            # Let's build a simple inverse index for the uncovered triples?
            # Or just pick a random block and check if it covers u_idx?
            # Probability a random block covers a specific triple:
            # C(21, 4) / C(24, 7) ? No.
            # # blocks containing a specific triple = C(v-t, k-t) = C(21, 4) = 5985.
            # Total blocks = 346104.
            # Prob ~ 1.7%.
            # So we need ~60 tries to find a block.
            # 60 * 35 ops = 2100 ops. Fast.
            
            found = False
            for _ in range(200): # Cap tries
                b_idx = rng.randint(0, n_blocks - 1)
                if in_solution[b_idx]:
                    continue
                # Check if b_idx covers u_idx
                # u_idx is an integer index.
                # We need to check if u_idx is in block_t_indices[b_idx]
                # Since block_t_indices[b_idx] is a numpy array of 35 ints, 
                # we can do a binary search if sorted, or just check.
                # It's not sorted.
                # Use a set? No, creating set is slow.
                # Just iterate? 35 ops.
                if u_idx in block_t_indices[b_idx]:
                    # Add it
                    cover_counts[block_t_indices[b_idx]] += 1
                    in_solution[b_idx] = True
                    sol_list.append(b_idx)
                    n_uncovered = np.sum(cover_counts == 0)
                    found = True
                    break
                    
            if not found:
                # Should not happen if covering exists
                break
                
            # Update uncovered_idxs
            if n_uncovered > 0:
                uncovered_idxs = np.where(cover_counts == 0)[0]
            else:
                break

    # Final Pruning Pass (Greedy Removal)
    # Try to remove any redundant blocks
    changed = True
    while changed and time.time() < deadline - 1.0:
        changed = False
        for i in range(len(sol_list) - 1, -1, -1):
            b_idx = sol_list[i]
            t_idxs = block_t_indices[b_idx]
            # Check if all triples in this block have count > 1
            if np.min(cover_counts[t_idxs]) > 1:
                cover_counts[t_idxs] -= 1
                in_solution[b_idx] = False
                sol_list.pop(i)
                changed = True
                break # Restart loop to check new possibilities
                
    final_blocks = [list(all_blocks[i]) for i in sol_list]
    
    # Final Validation
    test_counts = np.zeros(n_t, dtype=np.int32)
    for b_idx in sol_list:
        test_counts[block_t_indices[b_idx]] += 1
        
    if np.any(test_counts == 0):
        # Fallback to greedy if repair failed
        covered_mask = np.zeros(n_uint64s, dtype=np.uint64)
        valid_blocks = []
        for _ in range(200):
            if time.time() > deadline - 1.0:
                break
            if np.bitwise_count(covered_mask).sum() == n_t:
                break
            not_covered = ~covered_mask
            intersections = coverage_masks & not_covered
            gains = np.bitwise_count(intersections).sum(axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            valid_blocks.append(best_idx)
            covered_mask |= coverage_masks[best_idx]
        return [list(all_blocks[i]) for i in valid_blocks]

    return final_blocks
