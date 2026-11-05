import itertools
import random
import time
import math
import sys

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
subset_frequencies = [0] * N_SUBSETS

for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        idx = subset_to_idx[sub]
        mask |= 1 << idx
        blocks_containing_subset[idx].append(len(all_blocks))
        subset_frequencies[idx] += 1
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

# Precompute per-block list of covered subset indices (35 each) for fast iteration
block_bit_indices = []
for mask in all_masks:
    bits = []
    mm = mask
    while mm:
        lsb = mm & -mm
        bits.append(lsb.bit_length() - 1)
        mm ^= lsb
    block_bit_indices.append(bits)

# Weighted greedy weights: rare subsets get higher priority
weights = [1.0 / f for f in subset_frequencies]

# For fast intersection of candidate sets in 2-for-1 moves, we might need 
# to precompute intersections of blocks_containing_subset for pairs of subsets? 
# That would be N_SUBSETS^2 which is ~3.7M * 35, too big. 
# Instead, we compute intersections on the fly using sets, which is fast enough 
# for the small number of uncovered subsets (usually < 10).

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        all_k_subsets = list(itertools.combinations(range(v), k))
        random.shuffle(all_k_subsets)
        for b in all_k_subsets:
            if not uncovered:
                break
            covered = set(itertools.combinations(b, t))
            if uncovered & covered:
                blocks.append(list(b))
                uncovered -= covered
        return blocks

    start_time = time.time()
    deadline = start_time + 115  # Safety margin below 120s kill

    def weighted_greedy():
        """Weighted greedy: pick block maximizing sum of weights over uncovered subsets it covers."""
        uncovered_mask = (1 << N_SUBSETS) - 1
        chosen = []
        while uncovered_mask:
            best_idx = -1
            best_gain = 0.0
            for i in range(N_BLOCKS):
                gain = 0.0
                for bit in block_bit_indices[i]:
                    if (uncovered_mask >> bit) & 1:
                        gain += weights[bit]
                        if gain <= best_gain:
                            break
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx < 0 or best_gain == 0.0:
                break
            chosen.append(best_idx)
            uncovered_mask &= ~all_masks[best_idx]
        return chosen

    # 1. Initialization
    # Run a few greedy restarts to find a good starting point
    best_init = None
    best_init_len = float('inf')
    
    # Run 5 greedy restarts with slight randomization in tie-breaking
    for _ in range(5):
        # Shuffle blocks to break ties randomly? 
        # The greedy above is deterministic based on weights. 
        # To add randomness, we can perturb weights slightly.
        perturbed_weights = [w * (1 + random.uniform(-0.1, 0.1)) for w in weights]
        
        # Redefine greedy with perturbed weights
        def weighted_greedy_perturbed():
            uncovered_mask = (1 << N_SUBSETS) - 1
            chosen = []
            while uncovered_mask:
                best_idx = -1
                best_gain = 0.0
                for i in range(N_BLOCKS):
                    gain = 0.0
                    for bit in block_bit_indices[i]:
                        if (uncovered_mask >> bit) & 1:
                            gain += perturbed_weights[bit]
                            if gain <= best_gain:
                                break
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx < 0 or best_gain == 0.0:
                    break
                chosen.append(best_idx)
                uncovered_mask &= ~all_masks[best_idx]
            return chosen

        cand = weighted_greedy_perturbed()
        if len(cand) < best_init_len:
            best_init_len = len(cand)
            best_init = cand

    current_blocks = best_init
    best_blocks = current_blocks.copy()
    best_len = len(current_blocks)

    # Build counts
    counts = [0] * N_SUBSETS
    for idx in current_blocks:
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def add_block(idx):
        current_blocks.append(idx)
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def remove_block(pos):
        idx = current_blocks[pos]
        for bit in block_bit_indices[idx]:
            counts[bit] -= 1
        current_blocks.pop(pos)

    def is_redundant(pos):
        idx = current_blocks[pos]
        for bit in block_bit_indices[idx]:
            if counts[bit] == 1:
                return False
        return True

    def find_redundant():
        for pos in range(len(current_blocks)):
            if is_redundant(pos):
                return pos
        return -1

    def try_2_for_1():
        """
        Attempt to find two blocks that can be replaced by one.
        Returns True if a swap was made.
        """
        if len(current_blocks) < 2:
            return False
        
        # Randomly select pairs of blocks to try
        # To avoid O(N^2) checks every step, we can try a random subset of pairs
        n = len(current_blocks)
        # Try up to 20 random pairs
        for _ in range(20):
            i, j = random.sample(range(n), 2)
            idx_i = current_blocks[i]
            idx_j = current_blocks[j]
            
            # Find subsets that would become uncovered if both i and j are removed
            # We need to check counts. 
            # A subset is uncovered if count == 1 and it is in block i or j.
            # Wait, if count == 2 and both are in i and j, removing them makes count 0.
            # So we need subsets s such that count(s) <= 2 and s is in (block_i U block_j).
            # Specifically:
            # If s is in i only: count(s) must be 1.
            # If s is in j only: count(s) must be 1.
            # If s is in both i and j: count(s) must be 2.
            
            uncovered_bits = []
            # Iterate through bits of i
            for bit in block_bit_indices[idx_i]:
                if counts[bit] == 1:
                    uncovered_bits.append(bit)
                elif counts[bit] == 2:
                    # Check if j also covers this bit
                    # We need to know if bit is in j. 
                    # We don't have a fast "is bit in block" check without scanning or a set.
                    # Let's use a set for block_j bits for this check? 
                    # Or we can just collect candidates and verify later.
                    # Actually, if counts[bit] == 2 and bit is in i, and bit is in j, 
                    # then removing i and j makes it 0. 
                    # If bit is in i but not j, and counts[bit]==2, removing i makes it 1. OK.
                    # So we only need to worry about bits where count becomes 0.
                    # Let's find bits in i that are also in j.
                    pass
            
            # Better approach: 
            # 1. Identify bits in i that have count 1. These MUST be covered by the new block.
            # 2. Identify bits in j that have count 1. These MUST be covered by the new block.
            # 3. Identify bits in (i U j) that have count 2 AND are in both i and j. These MUST be covered by the new block.
            
            must_cover = set()
            
            # Bits in i
            for bit in block_bit_indices[idx_i]:
                if counts[bit] == 1:
                    must_cover.add(bit)
            
            # Bits in j
            for bit in block_bit_indices[idx_j]:
                if counts[bit] == 1:
                    must_cover.add(bit)
            
            # Bits in both i and j with count 2
            # We need to check intersection of bits in i and j
            # Since blocks are small (35 bits), we can use sets
            set_i = set(block_bit_indices[idx_i])
            set_j = set(block_bit_indices[idx_j])
            common_bits = set_i.intersection(set_j)
            for bit in common_bits:
                if counts[bit] == 2:
                    must_cover.add(bit)
            
            if not must_cover:
                # Both blocks are redundant? 
                # If we remove them, nothing becomes uncovered.
                # We can just remove both.
                # But we need to be careful: remove one, then the other.
                # Actually, if must_cover is empty, it means no subset becomes uncovered.
                # So we can remove both.
                # Let's remove i and j.
                # Note: removing i might change counts for j, but we already calculated must_cover based on current counts.
                # If we remove i, counts decrease. 
                # If a bit was count 2 and in both, it becomes 1 after removing i. 
                # Then removing j makes it 0. 
                # Our logic above handles this: if count==2 and in both, it's in must_cover.
                # If count==1 and in i, it's in must_cover.
                # So if must_cover is empty, it is safe to remove both.
                
                # Remove i and j
                # We need to remove by index. Since we pop, indices shift.
                # Let's remove the larger index first.
                if i > j:
                    remove_block(i)
                    remove_block(j)
                else:
                    remove_block(j)
                    remove_block(i)
                
                if len(current_blocks) < best_len:
                    nonlocal best_len, best_blocks
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                return True
            
            # Find a block that covers all bits in must_cover
            # Intersect candidate lists
            cands = None
            for bit in must_cover:
                lst = blocks_containing_subset[bit]
                if cands is None:
                    cands = set(lst)
                else:
                    cands.intersection_update(lst)
                if not cands:
                    break
            
            if cands:
                # Exclude i and j
                cands.discard(idx_i)
                cands.discard(idx_j)
                
                if cands:
                    # Pick a random candidate
                    c = random.choice(list(cands))
                    if c not in current_blocks:
                        # Perform swap: remove i, j, add c
                        # Remove i and j first
                        if i > j:
                            remove_block(i)
                            remove_block(j)
                        else:
                            remove_block(j)
                            remove_block(i)
                        
                        add_block(c)
                        
                        if len(current_blocks) < best_len:
                            nonlocal best_len, best_blocks
                            best_len = len(current_blocks)
                            best_blocks = current_blocks.copy()
                        return True
        
        return False

    def try_1_for_2():
        """
        Attempt to remove one block and add two blocks.
        This is a perturbation move to escape local minima.
        """
        if len(current_blocks) == 0:
            return False
            
        i = random.randint(0, len(current_blocks) - 1)
        idx_i = current_blocks[i]
        
        # Find subsets that would become uncovered if we remove this block
        uncovered_if_removed = []
        for bit in block_bit_indices[idx_i]:
            if counts[bit] == 1:
                uncovered_if_removed.append(bit)
        
        if not uncovered_if_removed:
            # Redundant
            remove_block(i)
            if len(current_blocks) < best_len:
                nonlocal best_len, best_blocks
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
            return True
        
        # We need to find two blocks c1, c2 such that they cover all uncovered_if_removed
        # and they are not in current_blocks (or at least not i).
        # This is more complex. 
        # Simple heuristic: Pick a random block c1 that covers some, then c2 that covers the rest.
        # Or: Find all blocks covering the first uncovered bit, pick one, then find blocks covering the rest.
        
        if len(uncovered_if_removed) > 10:
            return False # Too hard to cover with 2 blocks
        
        # Pick a random block that covers the first uncovered bit
        first_bit = uncovered_if_removed[0]
        candidates_c1 = blocks_containing_subset[first_bit]
        candidates_c1 = [c for c in candidates_c1 if c != idx_i]
        
        if not candidates_c1:
            return False
            
        c1 = random.choice(candidates_c1)
        
        # Now we need c2 to cover the remaining bits
        # Bits covered by c1:
        bits_c1 = set(block_bit_indices[c1])
        remaining_bits = [b for b in uncovered_if_removed if b not in bits_c1]
        
        if not remaining_bits:
            # c1 covers everything. We just need to add c1. 
            # But this is a 1-for-1 move, which we handle elsewhere.
            # If we remove i and add c1, it's a swap.
            if c1 not in current_blocks:
                remove_block(i)
                add_block(c1)
                return True
            else:
                return False
        
        # Find c2 covering remaining_bits
        cands_c2 = None
        for bit in remaining_bits:
            lst = blocks_containing_subset[bit]
            if cands_c2 is None:
                cands_c2 = set(lst)
            else:
                cands_c2.intersection_update(lst)
            if not cands_c2:
                break
        
        if cands_c2:
            # Exclude i and c1
            cands_c2.discard(idx_i)
            cands_c2.discard(c1)
            if cands_c2:
                c2 = random.choice(list(cands_c2))
                if c2 not in current_blocks and c1 not in current_blocks:
                    # Remove i, add c1, add c2
                    remove_block(i)
                    add_block(c1)
                    add_block(c2)
                    return True
        return False

    # 2. Local Search Loop
    # We will alternate between 2-for-1 (improvement) and 1-for-2 (perturbation)
    # and redundant removal.
    
    no_improvement_steps = 0
    max_no_improvement = 2000
    
    while time.time() < deadline and no_improvement_steps < max_no_improvement:
        # Phase A: Prune redundant
        progress = True
        while progress:
            progress = False
            pos = find_redundant()
            if pos >= 0:
                remove_block(pos)
                progress = True
                if len(current_blocks) < best_len:
                    best_len = len(current_blocks)
                    best_blocks = current_blocks.copy()
                    no_improvement_steps = 0
                else:
                    no_improvement_steps += 1
        
        if len(current_blocks) == 0:
            break

        # Phase B: Try 2-for-1
        if try_2_for_1():
            no_improvement_steps = 0
            continue
        
        # Phase C: Try 1-for-1 swap (standard SA move)
        # Pick random block i
        i = random.randint(0, len(current_blocks) - 1)
        idx_i = current_blocks[i]
        
        uncovered_if_removed = []
        for bit in block_bit_indices[idx_i]:
            if counts[bit] == 1:
                uncovered_if_removed.append(bit)
        
        if not uncovered_if_removed:
            # Should have been caught by prune, but just in case
            remove_block(i)
            no_improvement_steps = 0
            continue
        
        if len(uncovered_if_removed) <= 4:
            cands = None
            for b in uncovered_if_removed:
                lst = blocks_containing_subset[b]
                if cands is None:
                    cands = set(lst)
                else:
                    cands.intersection_update(lst)
                if not cands:
                    break
            
            if cands:
                cands.discard(idx_i)
                if cands:
                    c = random.choice(list(cands))
                    if c not in current_blocks:
                        remove_block(i)
                        add_block(c)
                        # This is a neutral move (same size), but changes the solution.
                        # We accept it to explore.
                        # no_improvement_steps += 1 # It's a move, not an improvement, but it's progress in search
                        continue

        # Phase D: Perturbation (1-for-2 or random add)
        if random.random() < 0.5:
            if try_1_for_2():
                no_improvement_steps = 0
                continue
        else:
            # Add a random block
            c = random.randint(0, N_BLOCKS - 1)
            if c not in current_blocks:
                add_block(c)
                no_improvement_steps += 1

        no_improvement_steps += 1

    # 3. Final Polish
    # Run redundant removal again on best_blocks
    final_counts = [0] * N_SUBSETS
    for idx in best_blocks:
        for bit in block_bit_indices[idx]:
            final_counts[bit] += 1
    
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        for pos in range(len(best_blocks)):
            idx = best_blocks[pos]
            redundant = True
            for bit in block_bit_indices[idx]:
                if final_counts[bit] == 1:
                    redundant = False
                    break
            if redundant:
                for bit in block_bit_indices[idx]:
                    final_counts[bit] -= 1
                best_blocks.pop(pos)
                changed = True
                break

    return [list(all_blocks[i]) for i in best_blocks]
