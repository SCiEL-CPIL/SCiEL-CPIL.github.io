import time
import itertools
import numpy as np
from collections import defaultdict

_SEED = 20240612

def _rng(seed):
    return np.random.RandomState(seed)

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145
    
    # Precompute all triples
    all_triples = list(itertools.combinations(range(v), t))
    n_all = len(all_triples)
    triple_to_idx = {tr: i for i, tr in enumerate(all_triples)}
    
    # Precompute triples for each potential block?
    # There are C(24,7) = 346104 blocks.
    # Storing 35 indices for each is ~12M ints. 48MB. Acceptable.
    # But generating them takes time.
    # Let's generate them on the fly or precompute a mapping.
    # Actually, for the greedy phase, we need to evaluate many blocks.
    # Let's precompute the list of triples for all blocks.
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Map block tuple to index
    block_to_idx = {b: i for i, b in enumerate(all_blocks)}
    
    # block_triples[i] is a list of triple indices covered by block i
    block_triples = [None] * n_blocks
    for i, b in enumerate(all_blocks):
        block_triples[i] = [triple_to_idx[tr] for tr in itertools.combinations(b, t)]

    count = np.zeros(n_all, dtype=np.int32)
    blocks_in_cover = []
    block_indices_in_cover = [] # indices into all_blocks

    def add_block(b):
        blocks_in_cover.append(b)
        idx = block_to_idx[b]
        block_indices_in_cover.append(idx)
        for tr_idx in block_triples[idx]:
            count[tr_idx] += 1

    def remove_block(b):
        idx = block_to_idx[b]
        # Find position in lists
        # Since we might remove from middle, we should use a set or handle carefully.
        # For speed, we can just mark as removed and clean up later, or use a set of indices.
        # Let's use a set of block indices in cover for O(1) removal.
        pass

    # Let's restructure to use sets for the cover to allow fast removal/addition.
    # cover_set: set of block indices (into all_blocks)
    # count: array of triple counts
    
    cover_set = set()
    
    def add_block_idx(idx):
        cover_set.add(idx)
        for tr_idx in block_triples[idx]:
            count[tr_idx] += 1

    def remove_block_idx(idx):
        cover_set.remove(idx)
        for tr_idx in block_triples[idx]:
            count[tr_idx] -= 1

    def get_blocks():
        return [all_blocks[i] for i in cover_set]

    # Phase 1: Greedy
    uncovered = np.ones(n_all, dtype=bool)
    
    while np.any(uncovered):
        if time.time() > deadline - 20:
            break
            
        best_idx = -1
        best_gain = 0
        
        # To speed up, we can sample blocks if n_blocks is large.
        # But n_blocks is 346k.
        # We can compute gains for all blocks?
        # gains = sum(uncovered[tr_idx] for tr_idx in block_triples[i])
        # This is a dot product if we had a matrix.
        # Let's just iterate. 346k iterations * 35 ops = 12M ops.
        # In Python, this is slow. ~1-2 seconds per iteration.
        # We need ~75 iterations. 75 * 2s = 150s. Too slow.
        
        # Optimization: Use NumPy.
        # Create a sparse matrix or just use the precomputed lists.
        # We can vectorize the gain calculation.
        # gains = np.zeros(n_blocks)
        # For each triple j that is uncovered, add 1 to all blocks covering it.
        # This requires reverse indexing: triple_to_blocks.
        
    # Let's precompute triple_to_blocks
    # triple_to_blocks[j] = list of block indices covering triple j
    triple_to_blocks = [[] for _ in range(n_all)]
    for i in range(n_blocks):
        for tr_idx in block_triples[i]:
            triple_to_blocks[tr_idx].append(i)

    # Now Greedy is faster:
    # Maintain an array `block_gain` of size n_blocks.
    # Initially, block_gain[i] = number of triples covered by block i.
    # block_gain = np.array([len(bt) for bt in block_triples])
    
    # When we add a block, we need to update gains of other blocks?
    # No, greedy picks max gain among *unselected* blocks.
    # But gain depends on *uncovered* triples.
    # Let's maintain `uncovered_count` per block?
    # Actually, simpler:
    # In each iteration, find the block with max overlap with `uncovered` set.
    # We can update a `current_gain` array.
    # Initially, current_gain[i] = len(block_triples[i]) (since all uncovered).
    # When we select block `b`, we mark its triples as covered.
    # For every other block `b'`, if it shares triples with `b`, its gain decreases.
    # This is complex to update incrementally without a matrix.
    
    # Alternative: Just sample a subset of blocks in each greedy step?
    # Or use the precomputed `triple_to_blocks`.
    # If we know which triples are uncovered, we can sum the number of blocks covering them? No.
    
    # Let's go back to the sampling approach in the original code but make it smarter.
    # Or, since v=24 is small, maybe we can just run the greedy for a few steps and then switch to local search.
    # The local search (2-for-1) is very effective.
    
    # Let's initialize with a simple random or greedy cover.
    # I will use a slightly optimized greedy:
    # 1. Find all uncovered triples.
    # 2. Pick a random uncovered triple.
    # 3. Pick a block covering it that maximizes gain.
    
    # Actually, let's just use the original greedy but limit it, and rely on local search.
    
    # Re-implementing the fast greedy with NumPy vectorization for gain calculation
    # We can create a boolean mask of uncovered triples.
    # gain[i] = sum(uncovered_mask[tr_idx] for tr_idx in block_triples[i])
    
    # This is still slow in Python loops.
    
    # Let's use the "2-for-1" local search as the primary engine, starting from a decent initial cover.
    # Initial cover: Random blocks? No, need to be valid.
    # Let's use a simple greedy that is fast enough for ~50 blocks.
    
    # Fast Greedy:
    # Pick random uncovered triple, pick best block covering it.
    
    while np.any(count == 0):
        if time.time() > deadline - 30:
            break
            
        uncovered_indices = np.where(count == 0)[0]
        if len(uncovered_indices) == 0:
            break
            
        # Pick a random uncovered triple
        tr_idx = uncovered_indices[np.random.randint(len(uncovered_indices))]
        
        # Find blocks covering this triple
        candidates = triple_to_blocks[tr_idx]
        
        # Evaluate gain for these candidates
        best_cand = -1
        best_gain = -1
        
        # Sample if too many candidates?
        # Max candidates for a triple: C(21, 4) = 5985.
        # This is small enough to evaluate all.
        
        for b_idx in candidates:
            # Calculate gain: how many *other* uncovered triples does this block cover?
            # We can just count how many triples in block_triples[b_idx] have count == 0
            gain = 0
            for tr_i in block_triples[b_idx]:
                if count[tr_i] == 0:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_cand = b_idx
                
        if best_cand != -1:
            add_block_idx(best_cand)

    # Phase 2: Local Search (2-for-1 and 1-for-0)
    # We represent the cover as a set of block indices.
    
    def try_remove_pair():
        # Try to find a pair to replace with 1 block
        # Iterate over all pairs in cover_set
        cover_list = list(cover_set)
        n = len(cover_list)
        
        for i in range(n):
            for j in range(i+1, n):
                b1_idx = cover_list[i]
                b2_idx = cover_list[j]
                
                # Calculate critical triples
                # We need to know which triples become uncovered if we remove b1 and b2.
                # A triple is critical if count[tr] - (1 if tr in b1 else 0) - (1 if tr in b2 else 0) == 0
                # AND count[tr] > 0 (it was covered).
                
                critical_triples = []
                p_critical = set()
                
                # Get triples in b1
                trs1 = set(block_triples[b1_idx])
                trs2 = set(block_triples[b2_idx])
                all_trs = trs1.union(trs2)
                
                for tr_i in all_trs:
                    c = count[tr_i]
                    c1 = 1 if tr_i in trs1 else 0
                    c2 = 1 if tr_i in trs2 else 0
                    c_rest = c - c1 - c2
                    
                    if c_rest == 0:
                        # This triple will be uncovered
                        # Get the triple points
                        tr_points = all_triples[tr_i]
                        critical_triples.append(tr_i)
                        p_critical.update(tr_points)
                
                if not critical_triples:
                    # Can remove both safely
                    remove_block_idx(b1_idx)
                    remove_block_idx(b2_idx)
                    return True
                
                if len(p_critical) <= k:
                    # Can replace with 1 block
                    # Find a block containing p_critical
                    # We need a block B_new such that p_critical is subset of B_new
                    # And B_new is not in cover_set (to keep distinct? Actually, if it is in cover_set, we just don't add it, but we removed 2 and added 0? No, we must add 1).
                    # If B_new is already in cover_set, we can't add it again.
                    # So we must find a B_new NOT in cover_set.
                    
                    # Generate candidate B_new
                    # p_critical is a set of points.
                    # We need to add k - len(p_critical) points from the remaining v - len(p_critical) points.
                    # There are C(v - len(p_critical), k - len(p_critical)) such blocks.
                    
                    # We just need one that is not in cover_set.
                    # Since cover_set is small (~75) and total blocks is large (~346k), it's very likely the first one we try is not in cover_set.
                    
                    p_list = sorted(p_critical)
                    remaining_points = [p for p in range(v) if p not in p_critical]
                    
                    # Try combinations of remaining points to fill the block
                    # To be fast, just try a few random ones or the first one.
                    # Let's try the lexicographically smallest one first.
                    fill_size = k - len(p_critical)
                    
                    if fill_size == 0:
                        b_new = tuple(p_list)
                        if b_new not in cover_set:
                            remove_block_idx(b1_idx)
                            remove_block_idx(b2_idx)
                            add_block_idx(block_to_idx[b_new])
                            return True
                        else:
                            # Try next?
                            pass
                    else:
                        # Try first combination of remaining_points
                        fill = tuple(remaining_points[:fill_size])
                        b_new = tuple(sorted(p_list + list(fill)))
                        
                        if b_new not in cover_set:
                            remove_block_idx(b1_idx)
                            remove_block_idx(b2_idx)
                            add_block_idx(block_to_idx[b_new])
                            return True
                        else:
                            # Try a few random ones
                            for _ in range(10):
                                fill_idx = np.random.choice(len(remaining_points), fill_size, replace=False)
                                fill = tuple(remaining_points[i] for i in fill_idx)
                                b_new = tuple(sorted(p_list + list(fill)))
                                if b_new not in cover_set:
                                    remove_block_idx(b1_idx)
                                    remove_block_idx(b2_idx)
                                    add_block_idx(block_to_idx[b_new])
                                    return True
                            
        return False

    def try_remove_single():
        # Try to remove a single block
        for b_idx in list(cover_set):
            trs = set(block_triples[b_idx])
            critical = []
            for tr_i in trs:
                if count[tr_i] == 1:
                    critical.append(tr_i)
            
            if not critical:
                remove_block_idx(b_idx)
                return True
            # If critical, we could try to replace with 1 block?
            # That would be a swap.
            # p_critical = union of points in critical
            # If len(p_critical) <= k, we can swap.
            # But swap doesn't reduce count.
            # We only care about reducing count.
            # So single removal is only valid if critical is empty.
        return False

    # Run local search
    while time.time() < deadline - 5:
        improved = False
        if try_remove_pair():
            improved = True
        if not improved and try_remove_single():
            improved = True
            
        if not improved:
            # Try to escape local minimum?
            # Maybe add a random block and then try to remove pairs?
            # This allows "2-for-1" to work if the new block helps cover criticals of two existing blocks.
            if time.time() < deadline - 10:
                # Add a random block
                b_idx = np.random.randint(n_blocks)
                if b_idx not in cover_set:
                    add_block_idx(b_idx)
                    # Now try to remove pairs again
                    # We don't reset 'improved' to False immediately, we just try again in next loop
                    # But we added a block, so count increased.
                    # We hope the next try_remove_pair will remove 2 blocks, netting -1.
                    pass
            else:
                break

    # Final check: ensure full coverage
    # If not fully covered, add blocks until covered
    while time.time() < deadline - 2:
        uncovered_indices = np.where(count == 0)[0]
        if len(uncovered_indices) == 0:
            break
        tr_idx = uncovered_indices[0]
        candidates = triple_to_blocks[tr_idx]
        # Pick first candidate not in cover
        for b_idx in candidates:
            if b_idx not in cover_set:
                add_block_idx(b_idx)
                break

    return get_blocks()
