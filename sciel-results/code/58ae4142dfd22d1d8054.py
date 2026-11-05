import itertools
import random
import time
import math

def construct(v, k, t):
    start = time.time()
    deadline = start + 115  # Safety margin for 120s limit
    
    rng = random.Random(1234567890)

    # Precompute all k-subsets (blocks) and t-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    t_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(t_subsets)
    sub_idx = {s: i for i, s in enumerate(t_subsets)}

    # Precompute block masks (bitsets of t-subsets covered)
    # For v=19, k=7, t=4: n_blocks = 50388, n_subsets = 3876
    # Python integers are arbitrary precision, so bitsets are efficient enough for bitwise ops.
    block_masks = [0] * n_blocks
    for bi, blk in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(blk, t):
            si = sub_idx[sub]
            mask |= 1 << si
        block_masks[bi] = mask

    # Precompute list of blocks covering each subset (for potential future use, or quick checks)
    # Not strictly needed for SA if we use masks, but useful for initialization heuristics.
    
    # Initialize a good starting solution using Greedy Set Cover
    def greedy_init():
        uncovered = (1 << n_subsets) - 1
        chosen = []
        chosen_set = set()
        while uncovered:
            best_bi = -1
            best_gain = 0
            best_mask = 0
            for bi in range(n_blocks):
                if bi in chosen_set:
                    continue
                gain = (block_masks[bi] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_bi = bi
                    best_mask = block_masks[bi]
                    if best_gain == uncovered.bit_count():
                        break
            if best_bi < 0:
                break
            chosen.append(best_bi)
            chosen_set.add(best_bi)
            uncovered &= ~best_mask
        return chosen

    # Prune redundant blocks from a valid covering
    def prune(blocks_list):
        chosen = list(blocks_list)
        chosen_set = set(chosen)
        counts = [0] * n_subsets
        for bi in chosen:
            mask = block_masks[bi]
            while mask:
                low = mask & -mask
                si = low.bit_length() - 1
                counts[si] += 1
                mask ^= low
        
        changed = True
        while changed:
            changed = False
            for i, bi in enumerate(chosen):
                mask = block_masks[bi]
                redundant = True
                while mask:
                    low = mask & -mask
                    si = low.bit_length() - 1
                    if counts[si] == 1:
                        redundant = False
                        break
                    mask ^= low
                if redundant:
                    mask = block_masks[bi]
                    while mask:
                        low = mask & -mask
                        si = low.bit_length() - 1
                        counts[si] -= 1
                        mask ^= low
                    chosen.pop(i)
                    chosen_set.discard(bi)
                    changed = True
                    break
        return chosen

    # Simulated Annealing
    # State: set of block indices
    # Fitness: -len(chosen) - penalty * (number of uncovered subsets)
    # We want to minimize len(chosen) while ensuring coverage.
    # If coverage is lost, penalty should be high enough to discourage it, 
    # but SA allows temporary invalid states to escape local minima.
    
    def simulate_annealing(initial_blocks, time_budget):
        current_blocks = list(initial_blocks)
        current_set = set(current_blocks)
        
        # Calculate initial coverage counts
        counts = [0] * n_subsets
        for bi in current_blocks:
            mask = block_masks[bi]
            while mask:
                low = mask & -mask
                si = low.bit_length() - 1
                counts[si] += 1
                mask ^= low
        
        uncovered_count = sum(1 for c in counts if c == 0)
        
        # Fitness function: we want to minimize size, but coverage is a hard constraint 
        # in the final result. For SA, we use a soft constraint.
        # Score = size + lambda * uncovered_count
        # We want to minimize this.
        def score(counts, size):
            unc = sum(1 for c in counts if c == 0)
            # If covered, score is size. If not, size + large penalty.
            # However, for SA to explore, we need a continuous landscape.
            # Let's use: size + 100 * unc
            return size + 100 * unc
        
        current_score = score(counts, len(current_blocks))
        best_blocks = list(current_blocks)
        best_score = current_score
        
        t_start = time.time()
        temp = 10.0
        cooling = 0.995
        
        while time.time() - t_start < time_budget:
            if temp < 0.01:
                break
                
            # Generate neighbor
            # 1. Remove a random block
            # 2. Add a random block
            # 3. Swap a random block for another
            
            move_type = rng.choice([1, 2, 3])
            
            if move_type == 1: # Remove
                if not current_blocks:
                    continue
                idx = rng.randrange(len(current_blocks))
                bi_out = current_blocks[idx]
                
                # Apply removal
                mask = block_masks[bi_out]
                while mask:
                    low = mask & -mask
                    si = low.bit_length() - 1
                    counts[si] -= 1
                    mask ^= low
                
                current_blocks.pop(idx)
                current_set.discard(bi_out)
                new_score = score(counts, len(current_blocks))
                
                # Acceptance
                delta = new_score - current_score
                if delta < 0 or rng.random() < math.exp(-delta / temp):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_blocks = list(current_blocks)
                else:
                    # Revert
                    current_blocks.insert(idx, bi_out)
                    current_set.add(bi_out)
                    mask = block_masks[bi_out]
                    while mask:
                        low = mask & -mask
                        si = low.bit_length() - 1
                        counts[si] += 1
                        mask ^= low
                        
            elif move_type == 2: # Add
                bi_in = rng.randrange(n_blocks)
                if bi_in in current_set:
                    continue
                
                # Apply addition
                mask = block_masks[bi_in]
                while mask:
                    low = mask & -mask
                    si = low.bit_length() - 1
                    counts[si] += 1
                    mask ^= low
                
                current_blocks.append(bi_in)
                current_set.add(bi_in)
                new_score = score(counts, len(current_blocks))
                
                # Acceptance
                delta = new_score - current_score
                if delta < 0 or rng.random() < math.exp(-delta / temp):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_blocks = list(current_blocks)
                else:
                    # Revert
                    current_blocks.pop()
                    current_set.discard(bi_in)
                    mask = block_masks[bi_in]
                    while mask:
                        low = mask & -mask
                        si = low.bit_length() - 1
                        counts[si] -= 1
                        mask ^= low

            else: # Swap
                if not current_blocks:
                    continue
                idx = rng.randrange(len(current_blocks))
                bi_out = current_blocks[idx]
                bi_in = rng.randrange(n_blocks)
                if bi_in in current_set:
                    continue
                
                # Apply swap
                mask_out = block_masks[bi_out]
                while mask_out:
                    low = mask_out & -mask_out
                    si = low.bit_length() - 1
                    counts[si] -= 1
                    mask_out ^= low
                current_set.discard(bi_out)
                
                mask_in = block_masks[bi_in]
                while mask_in:
                    low = mask_in & -mask_in
                    si = low.bit_length() - 1
                    counts[si] += 1
                    mask_in ^= low
                current_set.add(bi_in)
                
                current_blocks[idx] = bi_in
                new_score = score(counts, len(current_blocks))
                
                # Acceptance
                delta = new_score - current_score
                if delta < 0 or rng.random() < math.exp(-delta / temp):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_blocks = list(current_blocks)
                else:
                    # Revert
                    mask_in = block_masks[bi_in]
                    while mask_in:
                        low = mask_in & -mask_in
                        si = low.bit_length() - 1
                        counts[si] -= 1
                        mask_in ^= low
                    current_set.discard(bi_in)
                    
                    mask_out = block_masks[bi_out]
                    while mask_out:
                        low = mask_out & -mask_out
                        si = low.bit_length() - 1
                        counts[si] += 1
                        mask_out ^= low
                    current_set.add(bi_out)
                    
                    current_blocks[idx] = bi_out

            temp *= cooling
            
        # Ensure best_blocks is valid
        # Re-verify coverage of best_blocks
        counts_best = [0] * n_subsets
        for bi in best_blocks:
            mask = block_masks[bi]
            while mask:
                low = mask & -mask
                si = low.bit_length() - 1
                counts_best[si] += 1
                mask ^= low
        if any(c == 0 for c in counts_best):
            # If best found was invalid, we need to repair it or fall back to current if valid
            # For safety, let's return the pruned greedy if SA fails to find valid
            # But usually SA finds valid solutions quickly.
            # Let's try to repair best_blocks by adding blocks greedily
            uncovered = (1 << n_subsets) - 1
            for bi in best_blocks:
                uncovered &= ~block_masks[bi]
            while uncovered:
                best_bi = -1
                best_gain = 0
                for bi in range(n_blocks):
                    if bi in best_blocks:
                        continue
                    gain = (block_masks[bi] & uncovered).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_bi = bi
                if best_bi < 0:
                    break
                best_blocks.append(best_bi)
                uncovered &= ~block_masks[best_bi]
                
        return best_blocks

    # Main Search
    # 1. Greedy Init
    init_blocks = greedy_init()
    pruned_init = prune(init_blocks)
    
    best_solution = pruned_init
    best_size = len(pruned_init)
    
    # 2. Run SA for remaining time
    # Use most of the time for SA
    if time.time() < deadline - 5:
        sa_result = simulate_annealing(pruned_init, time_budget=deadline - time.time() - 5)
        pruned_sa = prune(sa_result)
        if len(pruned_sa) < best_size:
            best_solution = pruned_sa
            best_size = len(pruned_sa)
    
    # 3. Final Prune
    final_solution = prune(best_solution)
    
    return [list(all_blocks[bi]) for bi in final_solution]
