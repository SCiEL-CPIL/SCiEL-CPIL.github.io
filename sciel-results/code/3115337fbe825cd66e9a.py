import itertools
import random
import time
import math

# Global Precomputation
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
blocks_containing_subset = [[] for _ in range(N_SUBSETS)]

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback
        return [] # Or simple greedy
        
    start_time = time.time()
    deadline = start_time + 115 # 5s margin
    
    # 1. Greedy Initialization
    counts = [0] * N_SUBSETS
    current_blocks = []
    current_masks = []
    
    def add_block(idx):
        current_blocks.append(idx)
        current_masks.append(all_masks[idx])
        m = all_masks[idx]
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
        # Remove from lists
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

    # Greedy
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0
        # Optimization: Randomize order or use heuristic
        # To speed up, we can sample or just check all.
        # 50k checks * ~150 iterations = 7.5M bit_counts. Might be slow.
        # Let's try to speed up greedy.
        # Actually, greedy is only done once.
        for i in range(N_BLOCKS):
            gain = (all_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]
        
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()
    
    # 2. Local Search
    # We will use a loop that tries to remove blocks, and if stuck, swaps.
    
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        # Try to remove redundant blocks
        removed_any = False
        # Randomize order of checks
        indices = list(range(len(current_blocks)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline: break
            if is_redundant(i):
                remove_block(i)
                removed_any = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                # After removal, indices might be invalid if we removed
                # from middle? No, we removed from list, so indices shift.
                # It's safer to restart the shuffle loop or just break and restart.
                # Since removal is rare (only if redundant), and we shuffle,
                # let's just break and restart the inner loop to get fresh indices.
                break
        
        if removed_any:
            no_improve_count = 0
            continue
            
        # If no redundant block found, try a swap
        if len(current_blocks) == 0: break
        
        # Pick a random block to remove
        i = random.randint(0, len(current_blocks) - 1)
        
        # Tentatively remove
        m_i = current_masks[i]
        # Find uncovered subsets
        uncovered_bits = []
        mm = m_i
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                uncovered_bits.append(bit_idx)
            mm ^= lsb
            
        if not uncovered_bits:
            # It was redundant, why didn't we catch it above?
            # Maybe is_redundant was slow or we missed it due to index shift?
            # If counts[bit] == 1 for all bits in m_i, it is redundant.
            # The loop above checks if ANY count is 1.
            # If uncovered_bits is empty, it means all counts in m_i were > 1.
            # So it IS redundant.
            remove_block(i)
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            no_improve_count = 0
            continue
            
        # We need to cover uncovered_bits
        # If too many, revert
        if len(uncovered_bits) > 3:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Perturbation: Add a random block, then try to prune
                # This helps escape local minima
                c = random.randint(0, N_BLOCKS - 1)
                if c not in current_blocks:
                    add_block(c)
                    no_improve_count = 0
                continue
        else:
            # Find a block that covers all uncovered_bits
            # Intersect lists
            # Start with list for first bit
            candidates = blocks_containing_subset[uncovered_bits[0]]
            # Intersect with others
            # To speed up, pick the smallest list first
            uncovered_bits.sort(key=lambda b: len(blocks_containing_subset[b]))
            candidates = blocks_containing_subset[uncovered_bits[0]]
            
            for b in uncovered_bits[1:]:
                if not candidates: break
                # Intersect candidates with blocks_containing_subset[b]
                # Use set intersection for speed if lists are large?
                # Lists are size ~1000-2000.
                set_cands = set(candidates)
                set_b = set(blocks_containing_subset[b])
                candidates = list(set_cands.intersection(set_b))
                
            if candidates:
                # Pick a random candidate
                j = random.choice(candidates)
                # Ensure j is not i (it shouldn't be, since i doesn't cover uncovered bits? 
                # Wait, i covered the bits where count was 1. 
                # If j covers those bits, j is different from i? 
                # Not necessarily. But i is being removed. 
                # If j == i, we can't add it back as a swap, that's a no-op.
                # But if j == i, it means i covers all uncovered bits. 
                # Which it does, by definition. 
                # So we must pick j != i.
                valid_candidates = [c for c in candidates if c != i]
                if valid_candidates:
                    j = random.choice(valid_candidates)
                    # Perform swap: remove i, add j
                    remove_block(i)
                    add_block(j)
                    # Swap is valid by construction (coverage maintained)
                    # Size remains same. 
                    # This changes the structure.
                    no_improve_count = 0
                    continue
            else:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    # Perturbation
                    c = random.randint(0, N_BLOCKS - 1)
                    if c not in current_blocks:
                        add_block(c)
                        no_improve_count = 0
                    continue

    # Final cleanup: remove any remaining redundant blocks
    # Re-evaluate redundancy on best_blocks
    # We need to restore state to best_blocks to return them
    # Actually, we just return the list of blocks corresponding to best_blocks indices
    
    return [list(all_blocks[i]) for i in best_blocks]
