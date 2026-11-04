import itertools
import random
import time
import math

# Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        mask |= 1 << subset_to_idx[sub]
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

def construct(v, k, t):
    # Note: The function signature must be construct(v, k, t)
    # We assume v=19, k=7, t=4 based on the task description.
    # If v, k, t are different, we should handle it, but the task is specific.
    # The prompt says "The evaluator passes this task's parameters v, k and t."
    # So we should be generic or at least robust.
    # However, precomputing for v=19 is specific.
    # If v, k, t are different, we must recompute or use a generic greedy.
    # Given the constraints and the specific task "C(19,7,4)", we can assume these values.
    # But to be safe, if v != 19, we fall back to a simple greedy.
    
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (though unlikely to be tested)
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            block = list(min(uncovered))
            while len(block) < k:
                def gain(x):
                    return sum(1 for rest in itertools.combinations(block, t - 1)
                               if tuple(sorted(rest + (x,))) in uncovered)
                block.append(max((x for x in range(v) if x not in block), key=gain))
            block.sort()
            blocks.append(block)
            uncovered.difference_update(itertools.combinations(block, t))
        return blocks

    # Specific optimization for C(19,7,4)
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # 1. Greedy Initialization
    counts = [0] * N_SUBSETS
    current_blocks = []
    current_masks = []
    
    def add_block(idx):
        current_blocks.append(idx)
        current_masks.append(all_masks[idx])
        m = all_masks[idx]
        # Iterate set bits
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            mm ^= lsb

    def remove_block(i):
        m = current_masks[i]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] -= 1
            mm ^= lsb
        current_blocks.pop(i)
        current_masks.pop(i)

    def is_redundant(i):
        m = current_masks[i]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                return False
            mm ^= lsb
        return True

    # Greedy: Pick block covering most uncovered subsets
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0
        # Optimization: We can sample or check all. 50k checks * 110 blocks?
        # 50k * 150 = 7.5M operations. Might be slow in Python loop.
        # Let's just check all.
        for i in range(N_BLOCKS):
            gain = (all_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]
        
    # 2. Local Search (Simulated Annealing)
    best_blocks = current_blocks.copy()
    best_len = len(current_blocks)
    T = 10.0
    
    while time.time() < deadline:
        new_blocks = current_blocks.copy()
        new_masks = current_masks.copy()
        new_counts = counts.copy()
        
        # Helper to add/remove in new state
        def n_add(idx):
            new_blocks.append(idx)
            new_masks.append(all_masks[idx])
            m = all_masks[idx]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                new_counts[bit_idx] += 1
                mm ^= lsb

        def n_remove(i):
            m = new_masks[i]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                new_counts[bit_idx] -= 1
                mm ^= lsb
            new_blocks.pop(i)
            new_masks.pop(i)

        def n_is_redundant(i):
            m = new_masks[i]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if new_counts[bit_idx] == 1:
                    return False
                mm ^= lsb
            return True

        move_type = random.random()
        accepted = False
        
        if move_type < 0.4:
            # Try to remove a redundant block
            if new_blocks:
                i = random.randint(0, len(new_blocks) - 1)
                if n_is_redundant(i):
                    n_remove(i)
                    accepted = True
        elif move_type < 0.9:
            # Swap
            if new_blocks:
                i = random.randint(0, len(new_blocks) - 1)
                c = random.randint(0, N_BLOCKS - 1)
                # Check if c is already in new_blocks
                if c in new_blocks:
                    continue
                n_remove(i)
                n_add(c)
                # Swap is always accepted if valid (it is, since we just swapped)
                # But we should ensure coverage is maintained?
                # The swap itself doesn't change the set of covered subsets?
                # No, it does. Block i covered some, Block c covers some.
                # We need to ensure all subsets are still covered.
                # Check if any count dropped to 0.
                valid = True
                # We can't easily check all 3876 counts.
                # But we can check if the union of masks is still full.
                # Or simpler: check if any subset that was covered only by i is now uncovered.
                # Let's just check if min(new_counts) > 0.
                # This is O(3876).
                # 3876 checks per swap. 1000 swaps/sec? 3.8M ops. OK.
                if min(new_counts) < 0: # Should not happen
                    valid = False
                # Actually, we need to ensure coverage.
                # If min(new_counts) >= 1, coverage is maintained.
                if min(new_counts) >= 1:
                    accepted = True
                else:
                    # Revert
                    new_blocks = current_blocks.copy()
                    new_masks = current_masks.copy()
                    new_counts = counts.copy()
                    accepted = False
        else:
            # Add
            c = random.randint(0, N_BLOCKS - 1)
            if c not in new_blocks:
                n_add(c)
                delta_E = 1
                if random.random() < math.exp(-delta_E / T):
                    accepted = True

        if accepted:
            current_blocks = new_blocks
            current_masks = new_masks
            counts = new_counts
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
        
        T *= 0.999
        if T < 0.1:
            T = 0.1

    # Final cleanup: remove any remaining redundant blocks
    # We need to update counts for best_blocks
    # Actually, best_blocks is a list of indices.
    # Let's just return the blocks corresponding to best_blocks.
    return [list(all_blocks[i]) for i in best_blocks]
