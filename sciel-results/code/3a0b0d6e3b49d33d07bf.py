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
    budget = 145.0  # safety margin before 150s kill

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_index = {tr: i for i, tr in enumerate(all_triples)}
    n_triples = len(all_triples)

    # Precompute triple indices for fast lookup
    triple_idx = {}
    for tr in all_triples:
        a, b, c = tr
        triple_idx[(a,b,c)] = triple_index[tr]
        triple_idx[(a,c,b)] = triple_index[tr]
        triple_idx[(b,a,c)] = triple_index[tr]
        triple_idx[(b,c,a)] = triple_index[tr]
        triple_idx[(c,a,b)] = triple_index[tr]
        triple_idx[(c,b,a)] = triple_index[tr]

    def get_triple_idx(a, b, c):
        return triple_idx[(a,b,c)]

    def block_triples_indices(block):
        """Return list of triple indices covered by a block."""
        inds = []
        for tr in itertools.combinations(block, t):
            inds.append(triple_index[tr])
        return inds

    rng = random.Random(42)
    
    def compute_block_scores(cov, n_triples, all_triples, triple_idx, points, k, t):
        """
        Compute submodular marginal coverage density scores.
        For each triple, score = marginal gain / structural redundancy.
        Marginal gain: how many uncovered triples this triple's point set can cover.
        Structural redundancy: overlap with already-covered structure.
        """
        if not _deadline_ok(budget):
            return None
        
        # Compute point frequency (how many triples contain each point)
        point_freq = np.zeros(v, dtype=np.int32)
        for tr in all_triples:
            for p in tr:
                point_freq[p] += 1
        
        # For each triple, compute a "difficulty" score based on point frequencies
        # Lower sum of point frequencies = more central/harder to cover
        triple_difficulty = np.zeros(n_triples, dtype=np.float64)
        for i, tr in enumerate(all_triples):
            triple_difficulty[i] = np.sum(point_freq[list(tr)])
        
        # Normalize to 0-1 range
        min_diff = np.min(triple_difficulty)
        max_diff = np.max(triple_difficulty)
        if max_diff > min_diff:
            triple_difficulty = (triple_difficulty - min_diff) / (max_diff - min_diff)
        else:
            triple_difficulty = np.ones(n_triples)
        
        # For uncovered triples, higher difficulty = higher priority
        # For covered triples, lower priority
        priority = triple_difficulty.copy()
        priority[cov > 0] *= 0.1  # Deprioritize already covered triples
        
        return priority

    def greedy_cover(seed):
        r = random.Random(seed)
        cov = np.zeros(n_triples, dtype=np.int32)
        uncovered_count = n_triples
        blocks = []
        
        # Compute initial priority scores
        priority = compute_block_scores(cov, n_triples, all_triples, triple_idx, points, k, t)
        if priority is None:
            priority = np.ones(n_triples)
        
        while uncovered_count > 0 and _deadline_ok(budget):
            # Find an uncovered triple with highest priority
            start_triple = None
            best_priority = -1
            for _ in range(200):
                idx = r.randrange(n_triples)
                if cov[idx] == 0 and priority[idx] > best_priority:
                    best_priority = priority[idx]
                    start_triple = all_triples[idx]
            if start_triple is None:
                # Fallback: scan for any uncovered triple
                for idx in range(n_triples):
                    if cov[idx] == 0:
                        start_triple = all_triples[idx]
                        break
                if start_triple is None:
                    break

            block = list(start_triple)
            block_set = set(block)
            
            # Greedily add points to maximize new coverage, weighted by priority
            while len(block) < k:
                best_x = None
                best_gain = -1
                # Pairs in current block
                block_pairs = list(itertools.combinations(block, t - 1))
                
                for x in points:
                    if x in block_set:
                        continue
                    gain = 0
                    for pair in block_pairs:
                        tr = (pair[0], pair[1], x)
                        idx = triple_idx[tr]
                        if cov[idx] == 0:
                            # Weight by priority
                            gain += priority[idx]
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x is not None:
                        # Tie break: random
                        if r.random() < 0.5:
                            best_x = x
                            
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)

            block.sort()
            blocks.append(block)
            
            # Update coverage
            b_inds = block_triples_indices(block)
            for idx in b_inds:
                if cov[idx] == 0:
                    uncovered_count -= 1
                cov[idx] += 1
            
            # Update priority scores periodically
            if len(blocks) % 10 == 0:
                new_priority = compute_block_scores(cov, n_triples, all_triples, triple_idx, points, k, t)
                if new_priority is not None:
                    priority = new_priority

        return blocks, cov, uncovered_count

    def local_search(blocks, cov, max_time=10.0):
        """
        Local search to minimize number of blocks.
        Moves:
        1. Remove redundant block.
        2. Swap point in a block (if it doesn't uncover any triple).
        """
        if not _deadline_ok(budget):
            return blocks, cov

        ls_start = time.time()
        current_blocks = [list(b) for b in blocks]
        current_cov = cov.copy()
        
        # Precompute triple indices for each current block
        block_inds_list = [block_triples_indices(b) for b in current_blocks]
        
        # Helper to check if a block is removable
        def is_removable(block_idx):
            inds = block_inds_list[block_idx]
            # Check if all covered by others
            # cov[inds] must be >= 2
            return np.all(current_cov[inds] >= 2)

        # Helper to attempt removal
        def try_remove(block_idx):
            nonlocal current_blocks, current_cov, block_inds_list
            if not is_removable(block_idx):
                return False
            inds = block_inds_list[block_idx]
            current_cov[inds] -= 1
            current_blocks.pop(block_idx)
            block_inds_list.pop(block_idx)
            return True

        # Helper to attempt swap
        def try_swap(block_idx, old_p, new_p):
            nonlocal current_blocks, current_cov, block_inds_list
            block = current_blocks[block_idx]
            if new_p in block:
                return False
            
            # Triples removed
            others = [p for p in block if p != old_p]
            rem_inds = []
            add_inds = []
            for tr in itertools.combinations(others, t - 1):
                rem_tr = (tr[0], tr[1], old_p)
                add_tr = (tr[0], tr[1], new_p)
                rem_inds.append(triple_idx[rem_tr])
                add_inds.append(triple_idx[add_tr])
            
            # Check validity: no triple in rem_inds has cov == 1
            if np.any(current_cov[rem_inds] == 1):
                return False
            
            # Apply swap
            current_cov[rem_inds] -= 1
            current_cov[add_inds] += 1
            
            # Update block
            block.remove(old_p)
            block.append(new_p)
            block.sort()
            current_blocks[block_idx] = block
            block_inds_list[block_idx] = block_triples_indices(block)
            
            return True

        # Main loop
        while _deadline_ok(budget) and (time.time() - ls_start) < max_time:
            # Phase 1: Remove redundant blocks
            removed_any = False
            # Iterate backwards to handle index shifts
            for i in range(len(current_blocks) - 1, -1, -1):
                if is_removable(i):
                    try_remove(i)
                    removed_any = True
            # If we removed blocks, we might have made others redundant?
            # Actually, removal only decreases coverage, so it can only make blocks *less* removable.
            # Wait, if I remove block A, coverage of triples in A decreases.
            # If a triple in B was covered by A and B, and A is removed, it's now only covered by B.
            # So B becomes *less* likely to be removable.
            # So one pass of removal is sufficient if we scan all.
            
            # Phase 2: Swaps
            # Try random swaps to find better configurations
            for _ in range(2000):
                if not _deadline_ok(budget) or (time.time() - ls_start) > max_time:
                    break
                if not current_blocks:
                    break
                
                b_idx = rng.randrange(len(current_blocks))
                block = current_blocks[b_idx]
                old_p = rng.choice(block)
                new_p = rng.choice(points)
                
                if new_p == old_p or new_p in block:
                    continue
                
                if try_swap(b_idx, old_p, new_p):
                    # After a swap, check for removals again
                    # This is expensive, so maybe do it less often
                    pass
                    
            # Occasionally do a full removal pass after swaps
            # Since swaps change coverage, blocks might become removable
            for i in range(len(current_blocks) - 1, -1, -1):
                if is_removable(i):
                    try_remove(i)

        return current_blocks, current_cov

    # Main
    best_blocks = None
    best_cov = None
    
    for restart in range(50):
        if not _deadline_ok(budget):
            break
            
        blocks, cov, unc = greedy_cover(restart)
        
        if unc == 0:
            # Local search
            lb_blocks, lb_cov = local_search(blocks, cov, max_time=2.0)
            
            if best_blocks is None or len(lb_blocks) < len(best_blocks):
                best_blocks = lb_blocks
                best_cov = lb_cov
                # If we found a very good one, spend more time on it
                if len(best_blocks) <= 76:
                     lb_blocks2, lb_cov2 = local_search(best_blocks, best_cov, max_time=5.0)
                     if len(lb_blocks2) < len(best_blocks):
                         best_blocks = lb_blocks2
                         best_cov = lb_cov2
        else:
            # If greedy didn't cover, we need to fix it
            # This is rare with good greedy, but handle it
            # Just add blocks for uncovered triples
            while unc > 0 and _deadline_ok(budget):
                # Find uncovered
                target = None
                for idx in range(n_triples):
                    if cov[idx] == 0:
                        target = all_triples[idx]
                        break
                if target is None: break
                
                block = list(target)
                block_set = set(block)
                while len(block) < k:
                    best_x = None
                    best_gain = -1
                    block_pairs = list(itertools.combinations(block, t - 1))
                    for x in points:
                        if x in block_set: continue
                        gain = 0
                        for pair in block_pairs:
                            tr = (pair[0], pair[1], x)
                            idx = triple_idx[tr]
                            if cov[idx] == 0: gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x is None: break
                    block.append(best_x)
                    block_set.add(best_x)
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                b_inds = block_triples_indices(block)
                for idx in b_inds:
                    if cov[idx] == 0: unc -= 1
                    cov[idx] += 1
                    
            if unc == 0:
                lb_blocks, lb_cov = local_search(blocks, cov, max_time=2.0)
                if best_blocks is None or len(lb_blocks) < len(best_blocks):
                    best_blocks = lb_blocks
                    best_cov = lb_cov

    if best_blocks is None:
        # Fallback
        blocks, cov, unc = greedy_cover(0)
        best_blocks = blocks

    return best_blocks
