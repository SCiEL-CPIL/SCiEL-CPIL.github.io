import itertools
import random
import time
import sys
from collections import defaultdict

def construct(v, k, t):
    # Precompute all t-subsets and their IDs
    t_subs = list(itertools.combinations(range(v), t))
    sub_to_id = {sub: i for i, sub in enumerate(t_subs)}
    num_subs = len(t_subs)
    
    def get_sub_ids(block):
        """Get IDs of all t-subsets contained in block."""
        return [sub_to_id[sub] for sub in itertools.combinations(block, t)]

    def greedy_cover(v, k, t, sub_to_id, t_subs, num_subs, seed):
        random.seed(seed)
        uncovered = set(range(num_subs))
        blocks = []
        
        while uncovered:
            # Pick a random uncovered subset to start
            start_sub_id = random.choice(list(uncovered))
            start_sub = t_subs[start_sub_id]
            block = list(start_sub)
            
            while len(block) < k:
                gains = []
                for x in range(v):
                    if x in block:
                        continue
                    # Calculate how many new uncovered subsets this point would help cover
                    remaining_points = [p for p in range(v) if p not in block and p != x]
                    need = t - len(block) - 1
                    count = 0
                    if need >= 0:
                        for rest in itertools.combinations(remaining_points, need):
                            s = tuple(sorted(tuple(block) + (x,) + rest))
                            sid = sub_to_id.get(s)
                            if sid is not None and sid in uncovered:
                                count += 1
                    gains.append((count, x))
                
                if not gains or max(g[0] for g in gains) == 0:
                    candidates = [x for x in range(v) if x not in block]
                    if candidates:
                        block.append(random.choice(candidates))
                    else:
                        break
                else:
                    # Sort by gain, then randomly among top to add diversity
                    gains.sort(reverse=True)
                    top_k = min(5, len(gains))
                    candidates = [g[1] for g in gains[:top_k]]
                    block.append(random.choice(candidates))
            
            block.sort()
            blocks.append(block)
            for sid in get_sub_ids(block):
                uncovered.discard(sid)
        
        return blocks

    def greedy_pool_block(v, k, t, sub_to_id, t_subs, num_subs, seed):
        """Generate a single good block using greedy heuristic."""
        random.seed(seed)
        # Start with a random t-subset
        start_sub = random.sample(range(v), t)
        block = list(start_sub)
        
        while len(block) < k:
            gains = []
            for x in range(v):
                if x in block:
                    continue
                # Count how many t-subsets containing x and current block elements are uncovered
                count = 0
                need = t - len(block)
                if need > 0:
                    remaining_points = [p for p in range(v) if p not in block and p != x]
                    for rest in itertools.combinations(remaining_points, need):
                        s = tuple(sorted(tuple(block) + (x,) + rest))
                        sid = sub_to_id.get(s)
                        if sid is not None:
                            count += 1
                gains.append((count, x))
            
            if not gains or max(g[0] for g in gains) == 0:
                candidates = [x for x in range(v) if x not in block]
                if candidates:
                    block.append(random.choice(candidates))
                else:
                    break
            else:
                gains.sort(reverse=True)
                top_k = min(3, len(gains))
                candidates = [g[1] for g in gains[:top_k]]
                block.append(random.choice(candidates))
        
        block.sort()
        return block

    # 1. Generate a diverse pool of candidate blocks
    pool = []
    pool_set = set()
    
    # Add blocks from several greedy runs to ensure good coverage
    for s in range(30):
        b = greedy_cover(v, k, t, sub_to_id, t_subs, num_subs, seed=1000 + s)
        for blk in b:
            if tuple(blk) not in pool_set:
                pool_set.add(tuple(blk))
                pool.append(list(blk))
        
        if len(pool) > 10000:
            break
    
    # Also add individually generated blocks
    for s in range(50):
        blk = tuple(greedy_pool_block(v, k, t, sub_to_id, t_subs, num_subs, seed=5000 + s))
        if blk not in pool_set:
            pool_set.add(blk)
            pool.append(list(blk))
    
    # Fill with random blocks to increase diversity for local search
    random.seed(42)
    target_pool_size = 15000
    while len(pool) < target_pool_size:
        blk = tuple(sorted(random.sample(range(v), k)))
        if blk not in pool_set:
            pool_set.add(blk)
            pool.append(list(blk))
    
    num_blocks = len(pool)
    
    # Precompute subset IDs for each block in the pool
    pool_sub_ids = [get_sub_ids(blk) for blk in pool]
    
    # Build inverted index: for each t-subset, which blocks cover it?
    # This is crucial for the WalkSAT "pick variable to satisfy clause" step
    sub_to_blocks = [[] for _ in range(num_subs)]
    for i in range(num_blocks):
        for sid in pool_sub_ids[i]:
            sub_to_blocks[sid].append(i)

    # 2. Local Search (WalkSAT-style)
    best_size = float('inf')
    best_selected = None
    start_time = time.time()
    timeout = 115.0
    
    # Run multiple iterations with different random seeds/initial states
    num_iterations = 10
    
    for iter_seed in range(num_iterations):
        if time.time() - start_time > timeout:
            break
            
        random.seed(2000 + iter_seed * 100)
        
        # Quick Greedy Initialization from pool
        uncovered = set(range(num_subs))
        selected_set = set()
        cover_count = [0] * num_subs
        
        # Greedy selection: pick block covering most uncovered subsets
        while uncovered:
            # Find block with maximum gain
            best_gain = -1
            best_cand = None
            
            # Sample a few random blocks to find good candidates (faster than scanning all)
            sample_size = min(200, num_blocks)
            sampled_indices = random.sample(range(num_blocks), sample_size)
            
            for cand_idx in sampled_indices:
                gain = 0
                for sid in pool_sub_ids[cand_idx]:
                    if sid in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cand = cand_idx
            
            if best_cand is not None and best_gain > 0:
                selected_set.add(best_cand)
                for sid in pool_sub_ids[best_cand]:
                    cover_count[sid] += 1
                    if cover_count[sid] == 1:
                        uncovered.discard(sid)
            
            # Remove redundant blocks
            for i in list(selected_set):
                redundant = True
                for sid in pool_sub_ids[i]:
                    if cover_count[sid] == 1:
                        redundant = False
                        break
                if redundant:
                    for sid in pool_sub_ids[i]:
                        cover_count[sid] -= 1
                        if cover_count[sid] == 0:
                            uncovered.add(sid)
                    selected_set.discard(i)
        
        current_size = len(selected_set)
        
        # WalkSAT Loop with simulated annealing
        temperature = 10.0
        stall_count = 0
        max_stall = 50
        
        while time.time() - start_time < timeout:
            # If we found a better solution, record it
            if current_size < best_size:
                best_size = current_size
                best_selected = list(selected_set)
                stall_count = 0
            else:
                stall_count += 1
            
            if stall_count > max_stall:
                break
            
            # WalkSAT: add a block, then delete redundant ones
            # Choose block to add: with some probability, pick one that covers many "critical" subsets
            added = False
            for _ in range(30):
                j = random.randint(0, num_blocks - 1)
                if j not in selected_set:
                    # Add block j
                    selected_set.add(j)
                    for sid in pool_sub_ids[j]:
                        cover_count[sid] += 1
                        if cover_count[sid] == 1:
                            uncovered.discard(sid)
                    added = True
                    break
            
            if not added:
                break
            
            # Aggressively delete redundant blocks
            deleted_any = True
            while deleted_any:
                deleted_any = False
                for i in list(selected_set):
                    redundant = True
                    for sid in pool_sub_ids[i]:
                        if cover_count[sid] == 1:
                            redundant = False
                            break
                    
                    if redundant:
                        for sid in pool_sub_ids[i]:
                            cover_count[sid] -= 1
                            if cover_count[sid] == 0:
                                uncovered.add(sid)
                        selected_set.discard(i)
                        deleted_any = True
                        current_size -= 1
            
            # Simulated annealing: if we got worse, revert with some probability
            if current_size > best_size and random.random() < 0.5:
                # Revert to best known state
                # This is expensive; instead, just continue
                pass

    # Convert best_selected indices back to blocks
    if best_selected is None:
        # Fallback: return a simple greedy cover
        return greedy_cover(v, k, t, sub_to_id, t_subs, num_subs, seed=9999)
        
    best_blocks = [pool[i] for i in best_selected]
    return best_blocks
