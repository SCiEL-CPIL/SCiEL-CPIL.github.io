import itertools
import random
import time

def construct(v, k, t):
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    
    # Precompute for each k-subset, which t-subsets it covers
    # This is C(19,7) = 50388 k-subsets, each covers C(7,4)=35 t-subsets
    # Total: 50388 * 35 = 1,763,580 operations - feasible
    
    all_k_subs = list(itertools.combinations(range(v), k))
    n_k_subs = len(all_k_subs)
    
    # ksub_to_tsubs[ksub_idx] = list of t-subset indices covered
    ksub_to_tsubs = []
    for ksub in all_k_subs:
        ksub_set = set(ksub)
        covered = []
        for idx, tsub in enumerate(all_t_subs):
            if all(p in ksub_set for p in tsub):
                covered.append(idx)
        ksub_to_tsubs.append(covered)
    
    # tsub_to_ksubs[tsub_idx] = list of k-subset indices that cover it
    tsub_to_ksubs = [set() for _ in range(n_t_subs)]
    for kidx, covered in enumerate(ksub_to_tsubs):
        for tidx in covered:
            tsub_to_ksubs[tidx].add(kidx)
    
    # Greedy initialization
    uncovered = set(range(n_t_subs))
    blocks = []
    block_to_ksub = {}  # block tuple -> ksub index
    
    while uncovered:
        # Pick the uncovered t-subset with fewest covering k-subsets (most constrained)
        target_idx = min(uncovered, key=lambda idx: len(tsub_to_ksubs[idx]))
        # Among k-subsets that cover target_idx, pick the one that covers most uncovered t-subsets
        best_kidx = -1
        best_gain = -1
        for kidx in tsub_to_ksubs[target_idx]:
            gain = sum(1 for tidx in ksub_to_tsubs[kidx] if tidx in uncovered)
            if gain > best_gain:
                best_gain = gain
                best_kidx = kidx
        blocks.append(all_k_subs[best_kidx])
        block_to_ksub[all_k_subs[best_kidx]] = best_kidx
        uncovered -= set(ksub_to_tsubs[best_kidx])
    
    # Local search: try to remove or replace blocks
    rng = random.Random(123)
    current_blocks = blocks[:]
    current_count = len(current_blocks)
    
    # Track which t-subsets are covered by which blocks
    # tsub_cover_count[tidx] = number of blocks covering it
    tsub_cover_count = [0] * n_t_subs
    for block in current_blocks:
        kidx = block_to_ksub[tuple(block)]
        for tidx in ksub_to_tsubs[kidx]:
            tsub_cover_count[tidx] += 1
    
    # Try to remove redundant blocks
    for i in range(len(current_blocks) - 1, -1, -1):
        block = current_blocks[i]
        kidx = block_to_ksub[tuple(block)]
        # Check if all t-subsets covered by this block are still covered by others
        redundant = True
        for tidx in ksub_to_tsubs[kidx]:
            if tsub_cover_count[tidx] <= 1:
                redundant = False
                break
        if redundant:
            for tidx in ksub_to_tsubs[kidx]:
                tsub_cover_count[tidx] -= 1
            current_blocks.pop(i)
    
    # Simulated annealing: try to replace blocks
    def get_coverage(blocks_list):
        """Return set of covered t-subset indices."""
        covered = set()
        for block in blocks_list:
            kidx = block_to_ksub[tuple(block)]
            covered.update(ksub_to_tsubs[kidx])
        return covered
    
    best_blocks = current_blocks[:]
    best_count = len(best_blocks)
    
    # Try random replacements
    for iteration in range(500):
        if len(best_blocks) < 151:
            break
        
        # Pick a random block to try replacing
        i = rng.randint(0, len(best_blocks) - 1)
        old_block = best_blocks[i]
        old_kidx = block_to_ksub[tuple(old_block)]
        
        # Try a random k-subset as replacement
        new_kidx = rng.randint(0, n_k_subs - 1)
        new_block = all_k_subs[new_kidx]
        
        # Check if replacement maintains coverage
        # Remove old block's coverage, add new block's coverage
        # Check if all t-subsets still covered
        temp_blocks = best_blocks[:]
        temp_blocks[i] = new_block
        
        # This is expensive, so let's be smarter
        # Only try if new block covers at least as many currently-unique t-subsets
        
        # Quick check: does new block cover the "critical" t-subsets?
        # A t-subset is critical if it's only covered by old_block
        critical = set()
        for tidx in ksub_to_tsubs[old_kidx]:
            # Count how many other blocks cover tidx
            count = 0
            for j, block in enumerate(best_blocks):
                if j == i:
                    continue
                jidx = block_to_ksub[tuple(block)]
                if tidx in ksub_to_tsubs[jidx]:
                    count += 1
            if count == 0:
                critical.add(tidx)
        
        # New block must cover all critical t-subsets
        new_covered = set(ksub_to_tsubs[new_kidx])
        if not critical.issubset(new_covered):
            continue
        
        # If we can replace and it doesn't increase count, keep it (same count)
        # Or if we can then remove another block, that's a win
        # For now, just keep track of best
        
        # Try to see if after replacement, some other block becomes redundant
        temp_cover_count = [0] * n_t_subs
        for j, block in enumerate(temp_blocks):
            jidx = block_to_ksub[tuple(block)]
            for tidx in ksub_to_tsubs[jidx]:
                temp_cover_count[tidx] += 1
        
        # Check if any block in temp_blocks is now redundant
        for j, block in enumerate(temp_blocks):
            jidx = block_to_ksub[tuple(block)]
            redundant = True
            for tidx in ksub_to_tsubs[jidx]:
                if temp_cover_count[tidx] <= 1:
                    redundant = False
                    break
            if redundant:
                # We found an improvement
                temp_blocks.pop(j)
                if len(temp_blocks) < best_count:
                    best_blocks = temp_blocks[:]
                    best_count = len(best_blocks)
                break
    
    return best_blocks
