import itertools
import random
import time
import sys

def construct(v, k, t):
    # Precompute all t-subsets
    t_subs = list(itertools.combinations(range(v), t))
    sub_to_id = {sub: i for i, sub in enumerate(t_subs)}
    num_subs = len(t_subs)
    
    def get_sub_ids(block):
        """Get IDs of all t-subsets contained in block."""
        return [sub_to_id[sub] for sub in itertools.combinations(block, t)]

    # 1. Generate a diverse pool of candidate blocks
    pool = []
    pool_set = set()
    
    # Fast randomized greedy to generate blocks
    for seed in range(20):
        random.seed(seed)
        uncovered = set(range(num_subs))
        
        while uncovered:
            # Pick a random uncovered subset to start a block
            start_sub_id = random.choice(list(uncovered))
            start_sub = t_subs[start_sub_id]
            block = list(start_sub)
            
            # Greedily extend to size k
            while len(block) < k:
                # Find the element that maximizes new coverage
                best_gain = -1
                best_elem = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    # New subsets are those containing x and t-1 elements from block
                    for rest in itertools.combinations(block, t - 1):
                        s = tuple(sorted((x,) + rest))
                        sid = sub_to_id.get(s)
                        if sid is not None and sid in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_elem = x
                
                if best_elem == -1:
                    candidates = [x for x in range(v) if x not in block]
                    if not candidates:
                        break
                    best_elem = random.choice(candidates)
                
                block.append(best_elem)
            
            block.sort()
            blk_tuple = tuple(block)
            
            if blk_tuple not in pool_set:
                pool_set.add(blk_tuple)
                pool.append(block)
            
            for sid in get_sub_ids(block):
                uncovered.discard(sid)
        
        if len(pool) > 6000:
            break

    # Ensure we have enough pool
    random.seed(999)
    while len(pool) < 5000:
        blk = tuple(sorted(random.sample(range(v), k)))
        if blk not in pool_set:
            pool_set.add(blk)
            pool.append(list(blk))

    num_pool = len(pool)
    
    # Precompute subset IDs for each block in the pool
    pool_sub_ids = [get_sub_ids(blk) for blk in pool]
    
    # Build inverted index: for each t-subset, which pool blocks cover it?
    sub_to_pool_blocks = [[] for _ in range(num_subs)]
    for i in range(num_pool):
        for sid in pool_sub_ids[i]:
            sub_to_pool_blocks[sid].append(i)

    # 2. Local Search (MinLS / WalkSAT style)
    best_size = float('inf')
    best_selected = None
    start_time = time.time()
    timeout = 115.0
    
    # State variables
    selected = set()
    cover_count = [0] * num_subs
    uncovered_count = num_subs
    
    def add_block(idx):
        nonlocal uncovered_count
        selected.add(idx)
        for sid in pool_sub_ids[idx]:
            cover_count[sid] += 1
            if cover_count[sid] == 1:
                uncovered_count -= 1

    def remove_block(idx):
        nonlocal uncovered_count
        selected.discard(idx)
        for sid in pool_sub_ids[idx]:
            cover_count[sid] -= 1
            if cover_count[sid] == 0:
                uncovered_count += 1

    def is_valid():
        return uncovered_count == 0

    def try_remove_random():
        """Try to remove a random block. Returns True if removed."""
        if not selected:
            return False
        idx = random.choice(list(selected))
        safe = True
        for sid in pool_sub_ids[idx]:
            if cover_count[sid] == 1:
                safe = False
                break
        
        if safe:
            remove_block(idx)
            return True
        return False

    def repair():
        """Add blocks until valid."""
        while uncovered_count > 0:
            # Pick a random uncovered subset
            target_sid = -1
            for _ in range(50):
                cand_sid = random.randint(0, num_subs - 1)
                if cover_count[cand_sid] == 0:
                    target_sid = cand_sid
                    break
            
            if target_sid == -1:
                for sid in range(num_subs):
                    if cover_count[sid] == 0:
                        target_sid = sid
                        break
            
            if target_sid == -1:
                break

            candidates = sub_to_pool_blocks[target_sid]
            if not candidates:
                break
            
            idx = random.choice(candidates)
            if idx not in selected:
                add_block(idx)

    # Initialize with a greedy cover from the pool
    repair()
    
    # Optimization loop
    stall_counter = 0
    
    while time.time() - start_time < timeout:
        if is_valid():
            # Try to minimize
            if try_remove_random():
                stall_counter = 0
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_selected = list(selected)
            else:
                stall_counter += 1
        
        else:
            # Repair
            repair()
            stall_counter = 0

        # If stalled, perform a Kick
        if stall_counter > 50:
            kick_idx = -1
            for _ in range(20):
                cand = random.randint(0, num_pool - 1)
                if cand not in selected:
                    kick_idx = cand
                    break
            
            if kick_idx != -1:
                add_block(kick_idx)
            
            # Now try to minimize again
            for _ in range(100):
                if not try_remove_random():
                    break
            
            stall_counter = 0

    if best_selected is None:
        if is_valid():
            best_selected = list(selected)
        else:
            repair()
            best_selected = list(selected)

    return [pool[i] for i in best_selected]
