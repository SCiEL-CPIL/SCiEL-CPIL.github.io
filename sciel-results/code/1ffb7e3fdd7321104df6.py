import itertools
import random
import time
import math
import numpy as np

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

# Precompute coverage matrix as numpy array for fast vectorized greedy
coverage_matrix = np.zeros((N_BLOCKS, N_SUBSETS), dtype=np.int8)
for i, bits in enumerate(block_bit_indices):
    for b in bits:
        coverage_matrix[i, b] = 1

# weights as numpy array
weights_np = np.array(weights, dtype=np.float64)

# Precompute alternative weightings for diverse greedy runs
weights_uniform = np.ones(N_SUBSETS, dtype=np.float64)
weights_sqrt_inv = np.array([1.0 / math.sqrt(f) for f in subset_frequencies], dtype=np.float64)
weights_inv_sqrt = np.array([1.0 / math.sqrt(f) for f in subset_frequencies], dtype=np.float64)

# Precompute block coverage counts for faster redundancy checks
block_covered_count = np.zeros(N_BLOCKS, dtype=np.int32)
for i, bits in enumerate(block_bit_indices):
    block_covered_count[i] = len(bits)

# Precompute pairwise overlap: overlap[i][j] = number of 4-subsets shared by blocks i and j
# This is used for 2-cover dominance pruning
# overlap[i][i] = 35 (self-overlap)
# For i != j, overlap[i][j] = |coverage(i) ∩ coverage(j)|
# Two blocks i,j are "compatible" if their union covers more than 35 subsets
# 2-cover dominance: block A is dominated if ∃ B,C such that coverage(A) ⊆ coverage(B) ∪ coverage(C)
# This is expensive to check exhaustively, so we use a heuristic:
# If |coverage(A) ∩ coverage(B)| + |coverage(A) ∩ coverage(C)| >= |coverage(A)| = 35
# and coverage(B) ∪ coverage(C) covers all of A's subsets, then A is 2-dominated.
# But checking the union is still expensive. Instead, we precompute a "dominance score"
# for each block: how many other blocks have high overlap with it.

# Precompute overlap matrix (N_BLOCKS x N_BLOCKS) - this is 77520^2 which is too large.
# Instead, for each block, store the top-50 blocks by overlap with it.
# Actually, let's think differently. The 2-cover dominance check in local search:
# Given a current set S, block i is 2-redundant if for every 4-subset in block i,
# there's some other block in S that covers it, AND no single other block covers all of i's subsets.
# But the standard redundancy check (count == 1) already handles the "no single block" case.
# The 2-cover dominance is: remove i if all of i's 35 subsets are covered by OTHER blocks in S.
# That's exactly the standard redundancy check! count[bit] >= 2 for all bits in block i.
# So the "2-cover dominance" in the context of a fixed set S is just the standard redundancy check.
# The pre-filtering idea is different: before running greedy, remove blocks that are 
# "structurally dominated" - i.e., there exist two OTHER blocks (not in any solution) whose 
# union covers all 35 subsets of the dominated block. This reduces the candidate pool.

# Let's compute for each block A, whether there exist B, C (B != C, B != A, C != A) such that
# coverage(A) ⊆ coverage(B) ∪ coverage(C).
# This is: for every 4-subset s in A, s is in B or s is in C.
# Equivalently: coverage(A) \ coverage(B) ⊆ coverage(C)
# So for a given A and B, we need C to cover coverage(A) \ coverage(B).
# Let missing = coverage(A) \ coverage(B), which has size 35 - overlap[A][B].
# We need a block C that covers all subsets in missing.
# A block C covers all of missing iff missing ⊆ coverage(C).
# Since |missing| can be up to 35, and each block covers 35 subsets, this is only possible
# if |missing| <= 35, which is always true. But C must cover ALL of missing.
# The number of blocks covering a specific set of m subsets is at most the number of blocks
# that contain all those subsets as 4-subsets within their 7-sets.
# For m = 35, C must be exactly A (since only A contains all 35 of A's 4-subsets... wait no,
# another 7-set could also contain some of A's 4-subsets). Actually, a 7-set contains exactly
# 35 4-subsets. If B != A, then coverage(A) \ coverage(B) has size 35 - overlap[A][B] > 0.
# For C to cover all of missing, C must contain each 4-subset in missing. Since C is a 7-set,
# it contains 35 4-subsets. The missing set has 35 - overlap[A][B] elements.
# This is feasible only if there exists a 7-set containing all those 4-subsets.
# In general, this is rare. Let's just do a lighter pre-filter:
# Remove block A if there exists B such that overlap[A][B] >= 34 (i.e., they share at least 34 of 35 subsets).
# Then any solution using A can be replaced by one using B (with possibly one extra block).
# Actually, this doesn't directly help. Let me think more carefully.

# A simpler and more effective pre-filter: for each block A, compute the number of blocks B
# that have overlap[A][B] >= 30. If this count is very high, A is "common" and likely replaceable.
# But this doesn't directly reduce the search space.

# Let me take a different approach: the key improvement should be in the LOCAL SEARCH phase.
# The current local search only does 1-for-1 swaps. Let's add:
# 1. A "2-for-1" improvement: remove two blocks and add one that covers all their unique subsets.
# 2. Better swap selection using the overlap structure.

# For now, let's focus on improving the local search with better swap heuristics
# and add a 2-for-1 move.

def construct(v, k, t):
    if v != 19 or k != 7 or t != 4:
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
    deadline = start_time + 112  # Safety margin below 120s kill

    def fast_greedy(weight_vector=None):
        """Vectorized weighted greedy using numpy for speed."""
        if weight_vector is None:
            weight_vector = weights_np
        uncovered = np.ones(N_SUBSETS, dtype=np.int8)
        chosen = []
        remaining = N_SUBSETS
        
        while remaining > 0:
            # Compute gain for each block: sum of weights of uncovered subsets it covers
            gains = coverage_matrix @ (weight_vector * uncovered)
            best_idx = int(np.argmax(gains))
            best_gain = gains[best_idx]
            
            if best_gain <= 0:
                break
            
            chosen.append(best_idx)
            
            # Update uncovered
            newly_covered = coverage_matrix[best_idx]
            still_uncovered = uncovered & (1 - newly_covered)
            remaining = int(np.sum(still_uncovered))
            uncovered = still_uncovered
        
        return chosen

    # 1. Try multiple greedy initializations with different weightings
    greedy_results = []
    for w_vec in [weights_np, weights_uniform, weights_sqrt_inv]:
        if time.time() > deadline - 5:
            break
        result = fast_greedy(w_vec)
        if result:
            greedy_results.append(result)
    
    # Pick the best greedy result
    if greedy_results:
        best_greedy = min(greedy_results, key=len)
    else:
        best_greedy = fast_greedy(weights_np)
    
    current_blocks = best_greedy
    best_len = len(current_blocks)
    best_blocks = current_blocks.copy()

    # Build counts (list of ints, fast)
    counts = [0] * N_SUBSETS
    for idx in current_blocks:
        for bit in block_bit_indices[idx]:
            counts[bit] += 1

    def add_block(idx, cnt, current, cset):
        current.append(idx)
        for bit in block_bit_indices[idx]:
            cnt[bit] += 1
        cset.add(idx)

    def remove_block(pos, cnt, current, cset):
        idx = current[pos]
        for bit in block_bit_indices[idx]:
            cnt[bit] -= 1
        current.pop(pos)
        cset.discard(idx)

    def do_local_search(blocks, counts, deadline):
        """Perform local search with 1-for-1 swaps and 2-for-1 moves."""
        nonlocal best_len, best_blocks
        current = blocks.copy()
        cnt = counts.copy()
        cset = set(current)
        n_imp = 0
        max_no_improve = 30
        
        while time.time() < deadline:
            # Phase A: prune all redundant blocks (standard: count[bit] >= 2 for all bits in block)
            progress = True
            while progress:
                progress = False
                for pos in range(len(current)):
                    idx = current[pos]
                    redundant = True
                    for bit in block_bit_indices[idx]:
                        if cnt[bit] == 1:
                            redundant = False
                            break
                    if redundant:
                        for bit in block_bit_indices[idx]:
                            cnt[bit] -= 1
                        current.pop(pos)
                        cset.discard(idx)
                        progress = True
                        if len(current) < best_len:
                            best_len = len(current)
                            best_blocks = current.copy()
                        n_imp = 0
                        break

            if len(current) == 0:
                break

            # Phase B: try 2-for-1 move: find two blocks whose union of unique subsets
            # can be covered by a single new block
            found_2for1 = False
            if len(current) >= 2 and time.time() < deadline - 2:
                # Sample pairs of blocks
                n_pairs = min(20, len(current) * (len(current) - 1) // 2)
                if n_pairs > 0:
                    indices = list(range(len(current)))
                    random.shuffle(indices)
                    for pi in range(min(n_pairs, len(indices) - 1)):
                        i1 = indices[pi]
                        i2 = indices[pi + 1]
                        idx1 = current[i1]
                        idx2 = current[i2]
                        
                        # Find subsets uniquely covered by block 1 or block 2
                        unique_to_1 = [bit for bit in block_bit_indices[idx1] if cnt[bit] == 1]
                        unique_to_2 = [bit for bit in block_bit_indices[idx2] if cnt[bit] == 1]
                        
                        if not unique_to_1 and not unique_to_2:
                            # Both are redundant, should have been caught in Phase A
                            continue
                        
                        # We want to find a block C that covers all unique_to_1 ∪ unique_to_2
                        # and does NOT cover any subset that would become uncovered
                        # Actually, C just needs to cover unique_to_1 ∪ unique_to_2.
                        # The other subsets of idx1 and idx2 are covered by other blocks.
                        
                        all_unique = set(unique_to_1) | set(unique_to_2)
                        if not all_unique:
                            continue
                        
                        if len(all_unique) > 20:
                            # Too many unique subsets, unlikely a single block covers them all
                            continue
                        
                        # Find candidates: blocks that cover all subsets in all_unique
                        cands = None
                        for bit in all_unique:
                            cset_for_bit = set(blocks_containing_subset[bit])
                            if cands is None:
                                cands = cset_for_bit
                            else:
                                cands.intersection_update(cset_for_bit)
                                if not cands:
                                    break
                        
                        if cands:
                            # Remove idx1 and idx2, add the best candidate
                            cands.discard(idx1)
                            cands.discard(idx2)
                            if cands:
                                # Pick the candidate that covers the most currently uncovered subsets
                                best_c = None
                                best_extra = -1
                                for c in list(cands)[:10]:
                                    extra = 0
                                    for bit in block_bit_indices[c]:
                                        if cnt[bit] == 0:
                                            extra += 1
                                    if extra > best_extra:
                                        best_extra = extra
                                        best_c = c
                                
                                if best_c is not None:
                                    # Remove i2 first (higher index) to keep indices valid
                                    pos2 = current.index(idx2)
                                    for bit in block_bit_indices[idx2]:
                                        cnt[bit] -= 1
                                    current.pop(pos2)
                                    cset.discard(idx2)
                                    
                                    pos1 = current.index(idx1)
                                    for bit in block_bit_indices[idx1]:
                                        cnt[bit] -= 1
                                    current.pop(pos1)
                                    cset.discard(idx1)
                                    
                                    for bit in block_bit_indices[best_c]:
                                        cnt[bit] += 1
                                    current.append(best_c)
                                    cset.add(best_c)
                                    
                                    found_2for1 = True
                                    n_imp = 0
                                    if len(current) < best_len:
                                        best_len = len(current)
                                        best_blocks = current.copy()
                                    break
            
            if found_2for1:
                continue

            # Phase C: try 1-for-1 swap
            best_swap_candidate = -1
            min_unique = float('inf')
            
            n_sample = min(len(current), 30)
            sampled_indices = random.sample(range(len(current)), n_sample)
            
            for i in sampled_indices:
                idx_i = current[i]
                n_unique = 0
                for bit in block_bit_indices[idx_i]:
                    if cnt[bit] == 1:
                        n_unique += 1
                if 0 < n_unique < min_unique:
                    min_unique = n_unique
                    best_swap_candidate = i
            
            if best_swap_candidate < 0:
                i = random.randint(0, len(current) - 1)
                idx_i = current[i]
                uncovered_bits = [bit for bit in block_bit_indices[idx_i] if cnt[bit] == 1]
                if not uncovered_bits:
                    for bit in block_bit_indices[idx_i]:
                        cnt[bit] -= 1
                    current.pop(i)
                    cset.discard(idx_i)
                    if len(current) < best_len:
                        best_len = len(current)
                        best_blocks = current.copy()
                    n_imp = 0
                    continue
                best_swap_candidate = i
                idx_i = current[i]
                uncovered_bits = [bit for bit in block_bit_indices[idx_i] if cnt[bit] == 1]
            else:
                i = best_swap_candidate
                idx_i = current[i]
                uncovered_bits = [bit for bit in block_bit_indices[idx_i] if cnt[bit] == 1]

            if not uncovered_bits:
                for bit in block_bit_indices[idx_i]:
                    cnt[bit] -= 1
                current.pop(i)
                cset.discard(idx_i)
                if len(current) < best_len:
                    best_len = len(current)
                    best_blocks = current.copy()
                n_imp = 0
                continue

            if len(uncovered_bits) > 5:
                n_imp += 1
                if n_imp > max_no_improve:
                    under_covered = [b for b in range(N_SUBSETS) if cnt[b] == 1]
                    if under_covered:
                        target = random.choice(under_covered)
                        cands_for_target = blocks_containing_subset[target]
                        c = random.choice(cands_for_target)
                        if c not in cset:
                            for bit in block_bit_indices[c]:
                                cnt[bit] += 1
                            current.append(c)
                            cset.add(c)
                    n_imp = 0
                continue

            uncovered_bits_sorted = sorted(uncovered_bits, key=lambda b: len(blocks_containing_subset[b]))
            cands = set(blocks_containing_subset[uncovered_bits_sorted[0]])
            for b in uncovered_bits_sorted[1:]:
                cands.intersection_update(blocks_containing_subset[b])
                if not cands:
                    break

            if cands:
                cands.discard(idx_i)
                if cands:
                    cands_list = list(cands)
                    if len(cands_list) <= 10:
                        best_j = cands_list[0]
                        best_extra = -1
                        for j in cands_list:
                            extra = 0
                            for bit in block_bit_indices[j]:
                                if cnt[bit] == 0:
                                    extra += 1
                            if extra > best_extra:
                                best_extra = extra
                                best_j = j
                        j = best_j
                    else:
                        j = random.choice(cands_list)
                    
                    # Remove i, add j
                    for bit in block_bit_indices[idx_i]:
                        cnt[bit] -= 1
                    current.pop(i)
                    cset.discard(idx_i)
                    
                    for bit in block_bit_indices[j]:
                        cnt[bit] += 1
                    current.append(j)
                    cset.add(j)
                    
                    n_imp = 0
                    continue

            n_imp += 1
            if n_imp > max_no_improve:
                under_covered = [b for b in range(N_SUBSETS) if cnt[b] == 1]
                if under_covered:
                    for _ in range(random.randint(2, 3)):
                        target = random.choice(under_covered)
                        cands_for_target = blocks_containing_subset[target]
                        c = random.choice(cands_for_target)
                        if c not in cset:
                            for bit in block_bit_indices[c]:
                                cnt[bit] += 1
                            current.append(c)
                            cset.add(c)
                else:
                    for _ in range(random.randint(1, 2)):
                        c = random.randint(0, N_BLOCKS - 1)
                        if c not in cset:
                            for bit in block_bit_indices[c]:
                                cnt[bit] += 1
                            current.append(c)
                            cset.add(c)
                n_imp = 0

        return current, cnt

    # Run local search
    current_blocks, counts = do_local_search(current_blocks, counts, deadline)

    # 3. If time remains, try more greedy restarts with different weightings
    if time.time() < deadline - 10:
        for w_vec in [weights_uniform, weights_sqrt_inv]:
            if time.time() > deadline - 5:
                break
            alt = fast_greedy(w_vec)
            if len(alt) < best_len:
                best_len = len(alt)
                best_blocks = alt.copy()
                # Quick prune pass
                counts2 = [0] * N_SUBSETS
                for idx in best_blocks:
                    for bit in block_bit_indices[idx]:
                        counts2[bit] += 1
                changed = True
                while changed and time.time() < deadline - 3:
                    changed = False
                    for pos in range(len(best_blocks)):
                        idx = best_blocks[pos]
                        redundant = True
                        for bit in block_bit_indices[idx]:
                            if counts2[bit] == 1:
                                redundant = False
                                break
                        if redundant:
                            for bit in block_bit_indices[idx]:
                                counts2[bit] -= 1
                            best_blocks.pop(pos)
                            changed = True
                            if len(best_blocks) < best_len:
                                best_len = len(best_blocks)
                            break

    # Final validation: ensure all blocks are distinct and valid
    seen = set()
    final_blocks = []
    for idx in best_blocks:
        if idx not in seen:
            seen.add(idx)
            final_blocks.append(list(all_blocks[idx]))
    
    return final_blocks
