import itertools
import random
import time
import numpy as np
import sys

_START_TIME = None

def _deadline_ok(budget):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return (time.time() - _START_TIME) < budget

def construct(v, k, t):
    global _START_TIME
    _START_TIME = time.time()
    budget = 140.0  # safety margin before 150s kill

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_index = {tr: i for i, tr in enumerate(all_triples)}
    n_triples = len(all_triples)

    # Precompute triple lookup
    triple_idx = {}
    for tr in all_triples:
        a, b, c = tr
        triple_idx[(a,b,c)] = triple_index[tr]
        triple_idx[(a,c,b)] = triple_index[tr]
        triple_idx[(b,a,c)] = triple_index[tr]
        triple_idx[(b,c,a)] = triple_index[tr]
        triple_idx[(c,a,b)] = triple_index[tr]
        triple_idx[(c,b,a)] = triple_index[tr]

    def block_triples_indices(block):
        """Return list of triple indices covered by a block."""
        inds = []
        for tr in itertools.combinations(block, t):
            inds.append(triple_index[tr])
        return inds

    def greedy_cover(seed):
        r = random.Random(seed)
        cov = np.zeros(n_triples, dtype=np.int32)
        uncovered_count = n_triples
        blocks = []
        
        # Initial priority: random to break symmetry
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
                block_pairs = list(itertools.combinations(block, t - 1))
                
                for x in points:
                    if x in block_set:
                        continue
                    gain = 0
                    for pair in block_pairs:
                        tr = (pair[0], pair[1], x)
                        idx = triple_idx[tr]
                        if cov[idx] == 0:
                            gain += priority[idx]
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                    elif gain == best_gain and best_x is not None:
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
                # Simple heuristic: deprioritize covered triples
                for idx in range(n_triples):
                    if cov[idx] > 0:
                        priority[idx] *= 0.5

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
        
        def is_removable(block_idx):
            inds = block_inds_list[block_idx]
            return np.all(current_cov[inds] >= 2)

        def try_remove(block_idx):
            nonlocal current_blocks, current_cov, block_inds_list
            if not is_removable(block_idx):
                return False
            inds = block_inds_list[block_idx]
            current_cov[inds] -= 1
            current_blocks.pop(block_idx)
            block_inds_list.pop(block_idx)
            return True

        def try_swap(block_idx, old_p, new_p):
            nonlocal current_blocks, current_cov, block_inds_list
            block = current_blocks[block_idx]
            if new_p in block:
                return False
            
            others = [p for p in block if p != old_p]
            rem_inds = []
            add_inds = []
            for tr in itertools.combinations(others, t - 1):
                rem_tr = (tr[0], tr[1], old_p)
                add_tr = (tr[0], tr[1], new_p)
                rem_inds.append(triple_idx[rem_tr])
                add_inds.append(triple_idx[add_tr])
            
            if np.any(current_cov[rem_inds] == 1):
                return False
            
            current_cov[rem_inds] -= 1
            current_cov[add_inds] += 1
            
            block.remove(old_p)
            block.append(new_p)
            block.sort()
            current_blocks[block_idx] = block
            block_inds_list[block_idx] = block_triples_indices(block)
            
            return True

        while _deadline_ok(budget) and (time.time() - ls_start) < max_time:
            # Phase 1: Remove redundant blocks
            for i in range(len(current_blocks) - 1, -1, -1):
                if is_removable(i):
                    try_remove(i)
            
            # Phase 2: Swaps
            for _ in range(2000):
                if not _deadline_ok(budget) or (time.time() - ls_start) > max_time:
                    break
                if not current_blocks:
                    break
                
                b_idx = random.randrange(len(current_blocks))
                block = current_blocks[b_idx]
                old_p = random.choice(block)
                new_p = random.choice(points)
                
                if new_p == old_p or new_p in block:
                    continue
                
                if try_swap(b_idx, old_p, new_p):
                    pass
                    
            for i in range(len(current_blocks) - 1, -1, -1):
                if is_removable(i):
                    try_remove(i)

        return current_blocks, current_cov

    # Main: multiple greedy restarts + local search
    best_blocks = None
    best_cov = None
    
    n_restarts = 80  # Adjust based on time budget
    
    for restart in range(n_restarts):
        if not _deadline_ok(budget):
            break
            
        blocks, cov, unc = greedy_cover(restart)
        
        if unc == 0:
            lb_blocks, lb_cov = local_search(blocks, cov, max_time=1.5)
            
            if best_blocks is None or len(lb_blocks) < len(best_blocks):
                best_blocks = lb_blocks
                best_cov = lb_cov
                if len(best_blocks) <= 76:
                    lb_blocks2, lb_cov2 = local_search(best_blocks, best_cov, max_time=3.0)
                    if len(lb_blocks2) < len(best_blocks):
                        best_blocks = lb_blocks2
                        best_cov = lb_cov2
        else:
            # Fix uncovered triples
            while unc > 0 and _deadline_ok(budget):
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
                lb_blocks, lb_cov = local_search(blocks, cov, max_time=1.5)
                if best_blocks is None or len(lb_blocks) < len(best_blocks):
                    best_blocks = lb_blocks
                    best_cov = lb_cov

    if best_blocks is None:
        blocks, cov, unc = greedy_cover(0)
        best_blocks = blocks

    return best_blocks
