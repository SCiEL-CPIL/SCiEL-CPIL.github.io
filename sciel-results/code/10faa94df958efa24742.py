import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 35s safety margin before 150s kill

    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    num_t = len(all_tsubs)
    tsub_idx = {ts: i for i, ts in enumerate(all_tsubs)}

    # Precompute for each possible k-subset (block) a bitmask of which t-subs it covers
    # There are C(v,k) blocks; for v=24,k=7 that's 346104. We can't store all masks as full arrays
    # but we can precompute the list of t-sub indices for each block lazily or use a compact form.
    
    # Alternative: precompute for each point, the list of t-sub indices containing it
    tsubs_containing_point = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for p in ts:
            tsubs_containing_point[p].append(i)

    def block_tsub_indices(block):
        """Return list of t-sub indices covered by this block."""
        block_set = set(block)
        idxs = []
        for ts in itertools.combinations(block, t):
            idxs.append(tsub_idx[ts])
        return idxs

    def build_coverage(blocks):
        """Build coverage count array for given list of blocks."""
        cov = np.zeros(num_t, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cov[tsub_idx[ts]] += 1
        return cov

    def is_valid(cov):
        return (cov >= 1).all()

    def greedy_cover(seed=0):
        """Greedy set cover: repeatedly pick block covering most uncovered t-subs."""
        rng = random.Random(seed)
        uncovered = np.ones(num_t, dtype=bool)
        blocks = []
        # Precompute for each block its t-sub list for faster access
        # But C(24,7)=346104 blocks, each with C(7,3)=35 t-subs. 
        # Greedy over all blocks each iteration is too slow.
        # Instead: greedy construction by growing blocks one point at a time.
        
        # We'll use a simpler approach: start with empty, add blocks that maximize new coverage
        # Use a candidate pool of random blocks + greedy-grown blocks
        
        # Actually, let's do a proper greedy: at each step, find the block (from a pool) 
        # that covers the most currently-uncovered t-subs.
        
        # Build a pool of candidate blocks:
        # 1. All blocks from a few greedy constructions
        # 2. Random blocks
        pool = set()
        
        # Greedy-grown blocks
        for s in range(20):
            rng2 = random.Random(seed * 1000 + s)
            unc = np.ones(num_t, dtype=bool)
            gblocks = []
            while unc.any():
                # Pick random uncovered t-sub to anchor
                unc_idx = np.where(unc)[0]
                if len(unc_idx) == 0:
                    break
                anchor_i = unc_idx[rng2.randint(0, len(unc_idx) - 1)]
                anchor = all_tsubs[anchor_i]
                block = list(anchor)
                block_set = set(block)
                
                while len(block) < k:
                    best_gain = 0
                    best_cands = []
                    for x in range(v):
                        if x in block_set:
                            continue
                        # Count uncovered t-subs that contain x and other t-1 points in block
                        gain = 0
                        for ti in tsubs_containing_point[x]:
                            if not unc[ti]:
                                continue
                            ts = all_tsubs[ti]
                            # Check all other points in block
                            ok = True
                            for p in ts:
                                if p != x and p not in block_set:
                                    ok = False
                                    break
                            if ok:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_cands = [x]
                        elif gain == best_gain and gain > 0:
                            best_cands.append(x)
                    if best_cands:
                        x = best_cands[rng2.randint(0, len(best_cands) - 1)]
                    else:
                        avail = [x for x in range(v) if x not in block_set]
                        x = avail[rng2.randint(0, len(avail) - 1)]
                    block.append(x)
                    block_set.add(x)
                
                block.sort()
                gblocks.append(block)
                for ts in itertools.combinations(block, t):
                    unc[tsub_idx[ts]] = False
            
            for b in gblocks:
                pool.add(tuple(b))
        
        # Add random blocks
        for _ in range(500):
            b = tuple(sorted(rng.sample(range(v), k)))
            pool.add(b)
        
        pool_list = list(pool)
        
        # Now greedy set cover over pool
        uncovered = np.ones(num_t, dtype=bool)
        selected = []
        while uncovered.any() and time.time() < deadline - 30:
            best_gain = 0
            best_block = None
            # Sample a subset of pool for speed
            sample_size = min(len(pool_list), 2000)
            sample = rng.sample(pool_list, sample_size) if len(pool_list) > sample_size else pool_list
            for b in sample:
                gain = 0
                for ts in itertools.combinations(b, t):
                    if uncovered[tsub_idx[ts]]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b
                elif gain == best_gain and gain > 0 and best_block is None:
                    best_block = b
            
            if best_block is None or best_gain == 0:
                # Try all pool
                for b in pool_list:
                    gain = 0
                    for ts in itertools.combinations(b, t):
                        if uncovered[tsub_idx[ts]]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b
            
            if best_block is None or best_gain == 0:
                break
            
            selected.append(list(best_block))
            for ts in itertools.combinations(best_block, t):
                uncovered[tsub_idx[ts]] = False
        
        return selected

    def remove_redundant(blocks):
        """Remove blocks whose all t-subs are covered by other blocks."""
        if not blocks:
            return blocks
        cov = build_coverage(blocks)
        changed = True
        while changed:
            changed = False
            n = len(blocks)
            order = list(range(n))
            random.shuffle(order)
            for i in order:
                if i >= len(blocks):
                    continue
                b = blocks[i]
                redundant = True
                for ts in itertools.combinations(b, t):
                    if cov[tsub_idx[ts]] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts in itertools.combinations(b, t):
                        cov[tsub_idx[ts]] -= 1
                    blocks.pop(i)
                    changed = True
        return blocks

    def simulated_annealing(blocks, time_limit=60.0):
        """SA local search: swap elements in blocks, try to reduce block count."""
        rng = random.Random(42)
        blocks = [list(b) for b in blocks]
        cov = build_coverage(blocks)
        best_blocks = [list(b) for b in blocks]
        best_len = len(blocks)
        
        T = 10.0
        T_min = 0.01
        cooling = 0.999
        t_end = start_time + time_limit
        
        while time.time() < t_end:
            T *= cooling
            if T < T_min:
                T = T_min
            
            # Random move: pick a block, swap one or two elements
            i = rng.randint(0, len(blocks) - 1)
            old_block = list(blocks[i])
            new_block = list(old_block)
            
            n_swap = rng.choice([1, 1, 2])
            for _ in range(n_swap):
                pos = rng.randint(0, k - 1)
                old_val = new_block[pos]
                new_val = rng.randint(0, v - 1)
                if new_val not in new_block:
                    new_block[pos] = new_val
            
            new_block.sort()
            if set(new_block) == set(old_block):
                continue
            
            # Compute delta in coverage
            old_tsubs = set(itertools.combinations(old_block, t))
            new_tsubs = set(itertools.combinations(new_block, t))
            
            # T-subs only in old block
            lost = old_tsubs - new_tsubs
            # T-subs only in new block
            gained = new_tsubs - old_tsubs
            
            # Check if any lost t-sub becomes uncovered
            critical = False
            for ts in lost:
                if cov[tsub_idx[ts]] == 1:
                    critical = True
                    break
            
            if critical:
                # Accept with probability exp(-1/T) if it doesn't break validity
                # Actually, we should reject or accept based on energy
                # Energy = number of uncovered t-subs + 0.1 * number of blocks
                # Let's use a simpler criterion: reject if critical
                # Or accept with small probability
                if rng.random() > 0.1:
                    continue
            else:
                # Safe move, always accept
                pass
            
            # Apply move
            blocks[i] = new_block
            for ts in lost:
                cov[tsub_idx[ts]] -= 1
            for ts in gained:
                cov[tsub_idx[ts]] += 1
            
            # Check for redundant blocks
            if len(blocks) > best_len:
                # Try to remove redundant blocks
                removed = False
                for j in range(len(blocks) - 1, -1, -1):
                    b = blocks[j]
                    can_remove = True
                    for ts in itertools.combinations(b, t):
                        if cov[tsub_idx[ts]] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for ts in itertools.combinations(b, t):
                            cov[tsub_idx[ts]] -= 1
                        blocks.pop(j)
                        removed = True
                        if len(blocks) < best_len:
                            best_len = len(blocks)
                            best_blocks = [list(x) for x in blocks]
                        break
            
            # Track best
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = [list(x) for x in blocks]
        
        return best_blocks

    def try_remove_blocks(blocks, time_limit=10.0):
        """Try to remove blocks by finding alternative coverage."""
        cov = build_coverage(blocks)
        t_end = start_time + time_limit
        changed = True
        while changed and time.time() < t_end:
            changed = False
            n = len(blocks)
            order = list(range(n))
            random.shuffle(order)
            for i in order:
                if i >= len(blocks):
                    continue
                b = blocks[i]
                # Check if all t-subs of b are covered by other blocks
                can_remove = True
                for ts in itertools.combinations(b, t):
                    if cov[tsub_idx[ts]] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for ts in itertools.combinations(b, t):
                        cov[tsub_idx[ts]] -= 1
                    blocks.pop(i)
                    changed = True
        return blocks

    # Main
    best_blocks = None
    
    # Phase 1: Greedy construction
    t_phase1_end = start_time + 15.0
    for seed in range(10):
        if time.time() > t_phase1_end:
            break
        blocks = greedy_cover(seed)
        blocks = remove_redundant(blocks)
        if best_blocks is None or len(blocks) < len(best_blocks):
            best_blocks = blocks
    
    if best_blocks is None:
        # Fallback: simple greedy
        blocks = greedy_cover(0)
        best_blocks = blocks
    
    if time.time() > deadline - 40:
        return best_blocks
    
    # Phase 2: SA refinement
    t_phase2_end = deadline - 20.0
    blocks = simulated_annealing(best_blocks, time_limit=max(0, t_phase2_end - start_time))
    if len(blocks) < len(best_blocks):
        best_blocks = blocks
    
    # Phase 3: Final removal
    if time.time() < deadline - 10:
        best_blocks = try_remove_blocks(best_blocks, time_limit=8.0)
    
    return best_blocks
