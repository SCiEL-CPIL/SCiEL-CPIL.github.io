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
    # Use a safe budget to ensure we return before the 120s limit
    budget = 110.0 

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    n_triples = len(all_triples)
    
    # Map triple to index
    triple_index = {tr: i for i, tr in enumerate(all_triples)}
    
    # Precompute for each pair (a,b) the list of points c such that (a,b,c) is a triple
    # pair_triples[(a,b)] = [c1, c2, ...]
    pair_triples = {}
    for tr in all_triples:
        a, b, c = tr
        pair_ab = tuple(sorted((a, b)))
        if pair_ab not in pair_triples:
            pair_triples[pair_ab] = []
        pair_triples[pair_ab].append(c)
        
        pair_ac = tuple(sorted((a, c)))
        if pair_ac not in pair_triples:
            pair_triples[pair_ac] = []
        pair_triples[pair_ac].append(b)
        
        pair_bc = tuple(sorted((b, c)))
        if pair_bc not in pair_triples:
            pair_triples[pair_bc] = []
        pair_triples[pair_bc].append(a)

    def get_block_triple_indices(block):
        """Precompute the list of triple indices covered by a block."""
        return [triple_index[tr] for tr in itertools.combinations(block, t)]

    def greedy_cover(seed):
        r = random.Random(seed)
        cov = np.zeros(n_triples, dtype=np.int8)
        uncovered_count = n_triples
        blocks = []
        
        while uncovered_count > 0 and _deadline_ok(budget):
            # Pick a random uncovered triple
            start_triple = None
            for _ in range(200):
                idx = r.randrange(n_triples)
                if cov[idx] == 0:
                    start_triple = all_triples[idx]
                    break
            if start_triple is None:
                # Fallback: scan for first uncovered
                for idx in range(n_triples):
                    if cov[idx] == 0:
                        start_triple = all_triples[idx]
                        break
                if start_triple is None:
                    break

            block = list(start_triple)
            block_set = set(block)
            
            while len(block) < k:
                best_x = None
                best_gain = -1
                
                # Calculate gain for each candidate x
                # Gain = number of uncovered triples {a,b,x} where a,b in block
                # Optimize: Instead of iterating all pairs and all x, 
                # we can precompute for each x the list of pairs it forms with block.
                # But simple loop is likely fast enough for v=24, k=7.
                
                # To speed up, we can just iterate candidates and check uncovered triples
                for x in points:
                    if x in block_set:
                        continue
                    gain = 0
                    # Iterate pairs in current block
                    for i in range(len(block)):
                        for j in range(i+1, len(block)):
                            pair = tuple(sorted((block[i], block[j])))
                            # Check if x is in pair_triples[pair]
                            # And if the triple is uncovered
                            if x in pair_triples.get(pair, []):
                                tr = tuple(sorted(pair + (x,)))
                                idx = triple_index[tr]
                                if cov[idx] == 0:
                                    gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                        
                if best_x is None:
                    break
                block.append(best_x)
                block_set.add(best_x)
                
            block.sort()
            blocks.append(block)
            
            # Update coverage
            block_triples = get_block_triple_indices(block)
            for idx in block_triples:
                if cov[idx] == 0:
                    uncovered_count -= 1
                cov[idx] += 1
                
        return blocks, cov, uncovered_count

    def local_search(blocks, cov, uncovered_count, max_time=20.0):
        ls_start = time.time()
        best_blocks = list(blocks)
        best_len = len(blocks)
        current_blocks = list(blocks)
        current_cov = cov.copy()
        
        # Precompute block triple indices for current blocks
        current_block_triples = [get_block_triple_indices(b) for b in current_blocks]
        
        T = 5.0
        cooling = 0.995
        min_T = 0.1
        
        while (time.time() - ls_start) < max_time and _deadline_ok(budget) and T > min_T:
            if not current_blocks:
                break
                
            move_type = random.random()
            
            if move_type < 0.4:
                # Try to remove a block
                if len(current_blocks) > 1:
                    bi = random.randrange(len(current_blocks))
                    bt = current_block_triples[bi]
                    removable = True
                    for idx in bt:
                        if current_cov[idx] <= 1:
                            removable = False
                            break
                    if removable:
                        for idx in bt:
                            current_cov[idx] -= 1
                        current_blocks.pop(bi)
                        current_block_triples.pop(bi)
                        
                        if len(current_blocks) < best_len:
                            best_blocks = list(current_blocks)
                            best_len = len(current_blocks)
                            
            else:
                # Swap a point in a block
                if current_blocks:
                    bi = random.randrange(len(current_blocks))
                    block = current_blocks[bi]
                    old_p = random.choice(block)
                    new_p = random.choice(points)
                    if new_p == old_p or new_p in block:
                        continue
                        
                    # Calculate delta
                    others = [p for p in block if p != old_p]
                    
                    # Triples removed
                    removed_indices = []
                    for tr in itertools.combinations(block, t):
                        if old_p in tr:
                            removed_indices.append(triple_index[tr])
                            
                    # Triples added
                    added_indices = []
                    for tr in itertools.combinations(others, t-1):
                        tr_full = tuple(sorted(tr + (new_p,)))
                        added_indices.append(triple_index[tr_full])
                        
                    delta_uncov = 0
                    for idx in removed_indices:
                        if current_cov[idx] == 1:
                            delta_uncov += 1
                    for idx in added_indices:
                        if current_cov[idx] == 0:
                            delta_uncov -= 1
                            
                    # Acceptance criterion
                    if delta_uncov < 0:
                        accept = True
                    else:
                        if T > 0:
                            accept = random.random() < np.exp(-delta_uncov / T)
                        else:
                            accept = False
                            
                    if accept:
                        # Apply
                        for idx in removed_indices:
                            current_cov[idx] -= 1
                        for idx in added_indices:
                            current_cov[idx] += 1
                            
                        current_blocks[bi] = sorted(others + [new_p])
                        current_block_triples[bi] = get_block_triple_indices(current_blocks[bi])
                        
                        if len(current_blocks) < best_len:
                            best_blocks = list(current_blocks)
                            best_len = len(current_blocks)
                            
            T *= cooling
            
        return best_blocks

    best_result = None
    best_len = float('inf')
    
    # Run multiple restarts
    # Estimate time per greedy run: ~0.1-0.5s?
    # Estimate time per local search: ~1-5s?
    # Total budget 110s.
    
    for restart in range(100):
        if not _deadline_ok(budget):
            break
            
        # Greedy phase
        blocks, cov, unc = greedy_cover(restart)
        
        # If greedy didn't finish (timeout), skip local search for this restart
        # But we need a valid cover. Greedy always produces a valid cover if it finishes.
        # If it times out, `unc` might be > 0.
        if unc > 0:
            # Try to fix quickly? Or just skip?
            # Skipping is safer to avoid timeout.
            # But we need at least one valid result.
            # The first restart usually finishes.
            continue

        # Local search phase
        # Allocate time based on remaining budget
        elapsed = time.time() - _START_TIME
        remaining = budget - elapsed
        if remaining < 5.0:
            # Not enough time for LS, just use greedy result
            if len(blocks) < best_len:
                best_result = blocks
                best_len = len(blocks)
            continue
            
        ls_time = min(10.0, remaining * 0.5)
        blocks_ls = local_search(blocks, cov, unc, max_time=ls_time)
        
        if len(blocks_ls) < best_len:
            best_result = blocks_ls
            best_len = len(blocks_ls)
            
            # If we find a very good solution, try to polish further with remaining time
            if best_len <= 75:
                elapsed = time.time() - _START_TIME
                remaining = budget - elapsed
                if remaining > 5.0:
                    # Re-run LS on the best found so far? 
                    # We need cov for LS. We lost it. 
                    # Let's just run another LS on the current best if it's very good.
                    # But we don't have cov for best_result.
                    # We can reconstruct cov from best_result.
                    cov_check = np.zeros(n_triples, dtype=np.int8)
                    for b in best_result:
                        for idx in get_block_triple_indices(b):
                            cov_check[idx] += 1
                    unc_check = np.sum(cov_check == 0)
                    if unc_check == 0:
                        blocks_polish = local_search(best_result, cov_check, unc_check, max_time=min(5.0, remaining * 0.5))
                        if len(blocks_polish) < best_len:
                            best_result = blocks_polish
                            best_len = len(blocks_polish)

    if best_result is None:
        # Fallback: run greedy once more, hoping it finishes
        blocks, cov, unc = greedy_cover(999)
        if unc == 0:
            best_result = blocks
        else:
            # Emergency fix: add blocks for uncovered triples
            while unc > 0 and _deadline_ok(budget):
                target = None
                for idx in range(n_triples):
                    if cov[idx] == 0:
                        target = all_triples[idx]
                        break
                if target is None:
                    break
                block = list(target)
                block_set = set(block)
                while len(block) < k:
                    best_x = None
                    best_gain = -1
                    for x in points:
                        if x in block_set:
                            continue
                        gain = 0
                        for i in range(len(block)):
                            for j in range(i+1, len(block)):
                                pair = tuple(sorted((block[i], block[j])))
                                if x in pair_triples.get(pair, []):
                                    tr = tuple(sorted(pair + (x,)))
                                    idx = triple_index[tr]
                                    if cov[idx] == 0:
                                        gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x is None:
                        break
                    block.append(best_x)
                    block_set.add(best_x)
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                for idx in get_block_triple_indices(block):
                    if cov[idx] == 0:
                        unc -= 1
                    cov[idx] += 1
            best_result = blocks

    # Final validation
    cov_check = np.zeros(n_triples, dtype=np.int8)
    for block in best_result:
        for idx in get_block_triple_indices(block):
            cov_check[idx] += 1
            
    if not np.all(cov_check > 0):
        # Emergency fix
        for idx in range(n_triples):
            if cov_check[idx] == 0:
                target = all_triples[idx]
                block = list(target)
                block_set = set(block)
                while len(block) < k:
                    best_x = None
                    best_gain = -1
                    for x in points:
                        if x in block_set:
                            continue
                        gain = 0
                        for i in range(len(block)):
                            for j in range(i+1, len(block)):
                                pair = tuple(sorted((block[i], block[j])))
                                if x in pair_triples.get(pair, []):
                                    tr = tuple(sorted(pair + (x,)))
                                    idx2 = triple_index[tr]
                                    if cov_check[idx2] == 0:
                                        gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                    if best_x is None:
                        break
                    block.append(best_x)
                    block_set.add(best_x)
                block.sort()
                if block not in best_result:
                    best_result.append(block)
                for idx2 in get_block_triple_indices(block):
                    cov_check[idx2] += 1

    return best_result
