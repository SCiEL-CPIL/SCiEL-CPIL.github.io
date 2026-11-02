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
    budget = 135.0

    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    n_triples = len(all_triples)
    
    # Map triple to index
    triple_index = {tr: i for i, tr in enumerate(all_triples)}
    
    # Precompute for each point, the list of triples it is in
    # triples_by_point[p] = list of triple indices containing p
    triples_by_point = [[] for _ in range(v)]
    for tr in all_triples:
        for p in tr:
            triples_by_point[p].append(triple_index[tr])
            
    # Precompute for each pair of points, the triples containing both
    # pairs_by_pair[p1][p2] = list of triple indices containing p1 and p2
    # This is useful for calculating gain when adding a point to a block
    # Gain of adding x to block B = sum over pairs (a,b) in B of (uncovered triples {a,b,x})
    
    # For t=3, gain of adding x to B is:
    # For each pair (a,b) in B, check if {a,b,x} is uncovered.
    # We can precompute for each pair (a,b) the list of third points c such that {a,b,c} is a triple.
    # Actually, for a fixed pair (a,b), the triples are {a,b,c} for c in points\{a,b}.
    # We need to know if cov[{a,b,c}] == 0.
    
    # Let's precompute pair_triples[a][b] = list of c such that (a,b,c) is a triple.
    # Since v=24, pairs are small.
    pair_triples = {}
    for tr in all_triples:
        a, b, c = tr
        # Store in a dict keyed by frozenset or sorted tuple of pair
        # Using sorted tuple for speed
        pair = tuple(sorted((a, b)))
        if pair not in pair_triples:
            pair_triples[pair] = []
        pair_triples[pair].append(c)
        
        pair = tuple(sorted((a, c)))
        if pair not in pair_triples:
            pair_triples[pair] = []
        pair_triples[pair].append(b)
        
        pair = tuple(sorted((b, c)))
        if pair not in pair_triples:
            pair_triples[pair] = []
        pair_triples[pair].append(a)

    def get_triple_index(tr):
        return triple_index[tr]

    def greedy_cover(seed):
        r = random.Random(seed)
        cov = np.zeros(n_triples, dtype=np.int8)
        uncovered_count = n_triples
        blocks = []
        
        while uncovered_count > 0 and _deadline_ok(budget):
            # Pick a random uncovered triple
            start_triple = None
            for _ in range(100):
                idx = r.randrange(n_triples)
                if cov[idx] == 0:
                    start_triple = all_triples[idx]
                    break
            if start_triple is None:
                # Fallback
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
                # We can iterate over pairs in block
                block_pairs = list(itertools.combinations(block, t-1))
                
                # Optimize: Precompute gains for all x?
                # For each x, gain = 0
                # for pair in block_pairs:
                #    c_list = pair_triples.get(pair, [])
                #    if x in c_list:
                #        tr_idx = triple_index[tuple(sorted(pair + (x,)))]
                #        if cov[tr_idx] == 0: gain += 1
                
                # This is still O(v * C(k-1, t-1)).
                # v=24, k=7, t=3. C(6,2)=15. 24*15 = 360 operations per step.
                # 7 steps per block. ~2500 ops per block.
                # 76 blocks * 2500 = 190k ops. Very fast.
                
                for x in points:
                    if x in block_set:
                        continue
                    gain = 0
                    for pair in block_pairs:
                        # Check if x completes a triple with pair
                        # The triple is {pair[0], pair[1], x}
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
            for tr in itertools.combinations(block, t):
                idx = triple_index[tr]
                if cov[idx] == 0:
                    uncovered_count -= 1
                cov[idx] += 1
                
        return blocks, cov, uncovered_count

    def local_search(blocks, cov, uncovered_count, max_time=20.0):
        ls_start = time.time()
        best_blocks = list(blocks)
        best_cov = cov.copy()
        best_uncovered = uncovered_count
        current_blocks = list(blocks)
        current_cov = cov.copy()
        current_uncovered = uncovered_count
        
        # Precompute block triple indices
        def get_block_triples(block):
            return [triple_index[tr] for tr in itertools.combinations(block, t)]
            
        current_block_triples = [get_block_triples(b) for b in current_blocks]
        
        # Simulated Annealing
        T = 10.0
        cooling = 0.99
        
        steps = 0
        while (time.time() - ls_start) < max_time and _deadline_ok(budget):
            steps += 1
            if not current_blocks:
                break
                
            # Choose a move type
            move_type = random.random()
            
            if move_type < 0.3:
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
                        # Remove
                        for idx in bt:
                            current_cov[idx] -= 1
                        current_blocks.pop(bi)
                        current_block_triples.pop(bi)
                        
                        if len(current_blocks) < len(best_blocks):
                            best_blocks = list(current_blocks)
                            best_cov = current_cov.copy()
                            best_uncovered = current_uncovered
                            
            elif move_type < 0.8:
                # Swap a point in a block
                if current_blocks:
                    bi = random.randrange(len(current_blocks))
                    block = current_blocks[bi]
                    old_p = random.choice(block)
                    new_p = random.choice(points)
                    if new_p == old_p or new_p in block:
                        continue
                        
                    # Calculate delta in uncovered count
                    # Triples removed: those with old_p
                    # Triples added: those with new_p
                    
                    # We need to check coverage changes
                    # This is expensive to do exactly for every swap.
                    # Approximation: 
                    # Gain = (# added triples with cov 0) - (# removed triples with cov 1)
                    
                    removed_triples = []
                    added_triples = []
                    
                    others = [p for p in block if p != old_p]
                    
                    for tr in itertools.combinations(block, t):
                        if old_p in tr:
                            removed_triples.append(triple_index[tr])
                            
                    for tr in itertools.combinations(others, t-1):
                        tr_full = tuple(sorted(tr + (new_p,)))
                        added_triples.append(triple_index[tr_full])
                        
                    delta_uncov = 0
                    for idx in removed_triples:
                        if current_cov[idx] == 1:
                            delta_uncov += 1 # Becomes uncovered
                    for idx in added_triples:
                        if current_cov[idx] == 0:
                            delta_uncov -= 1 # Becomes covered
                            
                    # Accept if delta_uncov < 0 (improvement) or with probability exp(-delta/T)
                    if delta_uncov < 0:
                        # Apply
                        for idx in removed_triples:
                            current_cov[idx] -= 1
                        for idx in added_triples:
                            if current_cov[idx] == 0:
                                current_uncovered -= 1
                            current_cov[idx] += 1
                            
                        current_blocks[bi] = sorted(others + [new_p])
                        current_block_triples[bi] = get_block_triples(current_blocks[bi])
                        
                        if len(current_blocks) < len(best_blocks):
                            best_blocks = list(current_blocks)
                            best_cov = current_cov.copy()
                            best_uncovered = current_uncovered
                    else:
                        if random.random() < np.exp(-delta_uncov / T):
                            # Apply
                            for idx in removed_triples:
                                current_cov[idx] -= 1
                            for idx in added_triples:
                                if current_cov[idx] == 0:
                                    current_uncovered -= 1
                                current_cov[idx] += 1
                            
                            current_blocks[bi] = sorted(others + [new_p])
                            current_block_triples[bi] = get_block_triples(current_blocks[bi])
                            
                            if len(current_blocks) < len(best_blocks):
                                best_blocks = list(current_blocks)
                                best_cov = current_cov.copy()
                                best_uncovered = current_uncovered
                            
            else:
                # Add a block (if uncovered > 0)
                if current_uncovered > 0:
                    # Find best block to add
                    # Pick random uncovered triple, build block greedily
                    start_triple = None
                    for _ in range(50):
                        idx = random.randrange(n_triples)
                        if current_cov[idx] == 0:
                            start_triple = all_triples[idx]
                            break
                    if start_triple is None:
                        for idx in range(n_triples):
                            if current_cov[idx] == 0:
                                start_triple = all_triples[idx]
                                break
                    if start_triple is None:
                        continue
                        
                    block = list(start_triple)
                    block_set = set(block)
                    while len(block) < k:
                        best_x = None
                        best_gain = -1
                        block_pairs = list(itertools.combinations(block, t-1))
                        for x in points:
                            if x in block_set:
                                continue
                            gain = 0
                            for pair in block_pairs:
                                tr = tuple(sorted(pair + (x,)))
                                idx = triple_index[tr]
                                if current_cov[idx] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_x = x
                        if best_x is None:
                            break
                        block.append(best_x)
                        block_set.add(best_x)
                        
                    block.sort()
                    # Check if block already exists
                    if block not in current_blocks:
                        current_blocks.append(block)
                        current_block_triples.append(get_block_triples(block))
                        for tr in itertools.combinations(block, t):
                            idx = triple_index[tr]
                            if current_cov[idx] == 0:
                                current_uncovered -= 1
                            current_cov[idx] += 1
                            
            T *= cooling
            
        return best_blocks, best_cov, best_uncovered

    best_result = None
    for restart in range(50):
        if not _deadline_ok(budget):
            break
        blocks, cov, unc = greedy_cover(restart)
        if unc == 0:
            blocks, cov, unc = local_search(blocks, cov, unc, max_time=15.0)
            if best_result is None or len(blocks) < len(best_result):
                best_result = blocks
                # If we found a very good solution, do more search
                if len(blocks) <= 72:
                    blocks, cov, unc = local_search(blocks, cov, unc, max_time=10.0)
                    if len(blocks) < len(best_result):
                        best_result = blocks
        else:
            # Fix incomplete cover
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
                    block_pairs = list(itertools.combinations(block, t-1))
                    for x in points:
                        if x in block_set:
                            continue
                        gain = 0
                        for pair in block_pairs:
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
                for tr in itertools.combinations(block, t):
                    idx = triple_index[tr]
                    if cov[idx] == 0:
                        unc -= 1
                    cov[idx] += 1
                    
            if unc == 0:
                blocks, cov, unc = local_search(blocks, cov, unc, max_time=10.0)
                if best_result is None or len(blocks) < len(best_result):
                    best_result = blocks

    if best_result is None:
        blocks, cov, unc = greedy_cover(0)
        best_result = blocks
        
    # Final validation
    cov_check = np.zeros(n_triples, dtype=np.int8)
    for block in best_result:
        for tr in itertools.combinations(block, t):
            cov_check[triple_index[tr]] += 1
            
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
                    block_pairs = list(itertools.combinations(block, t-1))
                    for x in points:
                        if x in block_set:
                            continue
                        gain = 0
                        for pair in block_pairs:
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
                for tr in itertools.combinations(block, t):
                    cov_check[triple_index[tr]] += 1

    return best_result
