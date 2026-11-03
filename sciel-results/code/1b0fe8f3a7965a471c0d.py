import itertools
import random
import time
import numpy as np

_START_TIME = None


def _deadline_ok(budget):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) < budget


def construct(v, k, t):
    global _START_TIME
    _START_TIME = time.time()
    budget = 135.0  # safety margin before 150s kill

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_index = {tr: i for i, tr in enumerate(all_triples)}
    n_triples = len(all_triples)

    # Precompute all C(v,k) blocks and their triple indices
    all_blocks = list(itertools.combinations(points, k))
    n_blocks_total = len(all_blocks)
    
    # For each block, precompute the list of triple indices it covers
    block_triple_lists = []
    for block in all_blocks:
        trs = [triple_index[tr] for tr in itertools.combinations(block, t)]
        block_triple_lists.append(trs)

    # Convert to numpy arrays for fast operations
    # block_triple_matrix: sparse representation would be ideal, but let's use lists
    # Actually, let's precompute a numpy array of shape (n_blocks_total, t) for the triples? No, each block covers C(k,t) triples.
    # Let's use a list of sets for fast membership, but numpy is better for coverage updates.
    
    # Let's represent coverage as a numpy array
    # For fast gain calculation, we need to know for a given block, how many of its triples are uncovered.
    # gain(block) = sum(1 for tr in block_triples if cov[tr] == 0)
    
    # To speed this up, we can precompute for each block the list of triple indices.
    # cov is a numpy array.
    # gain = np.sum(cov[triple_indices] == 0)
    
    # Let's precompute triple_indices as a list of numpy arrays
    block_triple_arrays = [np.array(trs, dtype=np.int32) for trs in block_triple_lists]

    def greedy_cover(seed):
        r = random.Random(seed)
        cov = np.zeros(n_triples, dtype=np.int32)
        uncovered_count = n_triples
        blocks = []
        used_blocks = set()

        while uncovered_count > 0 and _deadline_ok(budget):
            best_block_idx = -1
            best_gain = -1
            
            # Sample a subset of candidate blocks to speed up search?
            # With 495918 blocks, iterating all is ~500k ops. 
            # np.sum on an array of size 35 is very fast.
            # 495918 * (array lookup + sum) might be ~1-2 seconds.
            # Let's try iterating all blocks.
            
            for b_idx in range(n_blocks_total):
                if b_idx in used_blocks:
                    continue
                tr_idx = block_triple_arrays[b_idx]
                # Count how many of these triples are currently uncovered (cov == 0)
                gain = np.sum(cov[tr_idx] == 0)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                elif gain == best_gain and best_block_idx != -1 and r.random() < 0.1:
                    # Tie break randomly
                    best_block_idx = b_idx

            if best_block_idx == -1 or best_gain == 0:
                # No improvement possible with unused blocks? Should not happen if uncovered_count > 0
                # Unless all blocks covering uncovered triples are already used.
                # In that case, we might need to allow re-using blocks? No, blocks must be distinct.
                # If best_gain == 0, it means no unused block covers any uncovered triple.
                # This implies the cover is stuck?
                # Actually, if an uncovered triple exists, ANY block containing it must cover it.
                # If all such blocks are already used, then the triple is covered by a used block?
                # Wait, if a block is used, its triples are covered (cov >= 1).
                # So if a triple is uncovered, NO block containing it has been used.
                # Therefore, there MUST be an unused block containing it, with gain >= 1.
                # So best_gain should be > 0.
                break

            # Add the best block
            blocks.append(all_blocks[best_block_idx])
            used_blocks.add(best_block_idx)
            
            # Update coverage
            tr_idx = block_triple_arrays[best_block_idx]
            # Count how many were 0
            newly_covered = np.sum(cov[tr_idx] == 0)
            uncovered_count -= newly_covered
            cov[tr_idx] += 1

        return blocks, cov, uncovered_count

    def is_valid(blocks):
        cov = np.zeros(n_triples, dtype=np.int32)
        for block in blocks:
            # Find index of this block
            # This is slow. Let's pass indices instead.
            pass
        # We'll track indices in the main loop.

    def local_search(block_indices, cov, uncovered_count, max_time=10.0):
        """
        Local search to remove redundant blocks.
        block_indices: list of indices into all_blocks
        """
        if not _deadline_ok(budget):
            return block_indices, cov, uncovered_count

        ls_start = time.time()
        current_indices = list(block_indices)
        current_cov = cov.copy()
        current_uncovered = uncovered_count
        
        best_indices = list(current_indices)
        best_cov = current_cov.copy()
        best_uncovered = current_uncovered

        # Precompute for each block index in current set, the triple array
        # We need to check if a block is redundant: all its triples have cov >= 2
        
        # Phase 1: Greedy removal
        progress = True
        while progress and _deadline_ok(budget) and (time.time() - ls_start) < max_time:
            progress = False
            for i in range(len(current_indices) - 1, -1, -1):
                b_idx = current_indices[i]
                tr_idx = block_triple_arrays[b_idx]
                # Check if all triples in this block are covered by at least one other block
                # i.e., current_cov[tr] >= 2 for all tr in tr_idx
                if np.all(current_cov[tr_idx] >= 2):
                    # Remove it
                    current_cov[tr_idx] -= 1
                    current_indices.pop(i)
                    progress = True
                    if len(current_indices) < len(best_indices):
                        best_indices = list(current_indices)
                        best_cov = current_cov.copy()
                        best_uncovered = current_uncovered

        return best_indices, best_cov, best_uncovered

    # Main loop
    best_result_indices = None
    best_result_len = float('inf')

    # Run multiple restarts
    for restart in range(1000):
        if not _deadline_ok(budget - 5):
            break
            
        block_indices, cov, unc = greedy_cover(restart)
        
        if unc == 0:
            # Valid cover found
            # Apply local search (removal)
            refined_indices, refined_cov, refined_unc = local_search(block_indices, cov, unc, max_time=2.0)
            
            if len(refined_indices) < best_result_len:
                best_result_len = len(refined_indices)
                best_result_indices = refined_indices
                # If we get close to 76, spend more time polishing
                if best_result_len <= 80:
                    # More aggressive local search
                    refined_indices2, refined_cov2, refined_unc2 = local_search(refined_indices, refined_cov, refined_unc, max_time=10.0)
                    if len(refined_indices2) < best_result_len:
                        best_result_len = len(refined_indices2)
                        best_result_indices = refined_indices2

    if best_result_indices is None:
        # Fallback: return a simple greedy cover (might be invalid if unc > 0, but we assume greedy finds a cover)
        # If greedy failed to cover all, we need to fix it.
        # But greedy always covers all if it runs to completion.
        # If it broke early, it's incomplete.
        # Let's just return the first greedy result's indices.
        block_indices, cov, unc = greedy_cover(0)
        best_result_indices = block_indices

    # Convert indices back to blocks
    result_blocks = [list(all_blocks[i]) for i in best_result_indices]
    
    # Final validation
    cov = np.zeros(n_triples, dtype=np.int32)
    for block in result_blocks:
        # Find index
        # This is O(1) if we use a set of tuples, but we have indices.
        # Let's just recompute coverage from scratch to be safe.
        for tr in itertools.combinations(block, t):
            idx = triple_index[tr]
            cov[idx] += 1
    
    if np.any(cov == 0):
        # Invalid! Add missing blocks
        # Find uncovered triples
        uncovered_triples = [all_triples[i] for i in range(n_triples) if cov[i] == 0]
        for tr in uncovered_triples:
            # Find any block containing this triple
            # This is slow.
            # Better: pick a random block from all_blocks that contains tr.
            # Precompute: for each triple, list of block indices containing it.
            # This is memory heavy: 2024 * (24-3 choose 7-3) = 2024 * 3003 ~ 6M ints. ~24MB. OK.
            pass
            
        # Let's do it properly:
        # Precompute triple_to_blocks
        triple_to_blocks = [[] for _ in range(n_triples)]
        for b_idx, block in enumerate(all_blocks):
            for tr in itertools.combinations(block, t):
                idx = triple_index[tr]
                triple_to_blocks[idx].append(b_idx)
        
        for tr in uncovered_triples:
            idx = triple_index[tr]
            # Pick a block that covers it
            b_idx = triple_to_blocks[idx][0]
            if b_idx not in best_result_indices:
                best_result_indices.append(b_idx)
                result_blocks.append(list(all_blocks[b_idx]))
                # Update cov
                tr_idx = block_triple_arrays[b_idx]
                cov[tr_idx] += 1

    return result_blocks
