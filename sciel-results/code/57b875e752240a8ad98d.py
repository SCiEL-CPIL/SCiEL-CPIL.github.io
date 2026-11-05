import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin

    # Precompute
    all_triplets = list(itertools.combinations(range(v), t))
    n_triplets = len(all_triplets)
    triplet_to_id = {triplet: i for i, triplet in enumerate(all_triplets)}
    
    def get_ids(block):
        return [triplet_to_id[triplet] for triplet in itertools.combinations(block, t)]

    def greedy_init(seed):
        rng = random.Random(seed)
        uncovered = set(range(n_triplets))
        blocks = []
        while uncovered:
            # Pick random uncovered triplet
            tid = rng.choice(list(uncovered))
            start_triplet = all_triplets[tid]
            
            # Build block around start_triplet
            block = list(start_triplet)
            block_set = set(block)
            
            # Greedily add points
            while len(block) < k:
                best_p = -1
                best_gain = -1
                # Candidates
                for p in range(v):
                    if p in block_set: continue
                    # Gain: how many uncovered triplets does p cover with current block?
                    gain = 0
                    for pair in itertools.combinations(block, t-1):
                        triplet = tuple(sorted(pair + (p,)))
                        tid2 = triplet_to_id[triplet]
                        if tid2 in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_p = p
                if best_p == -1:
                    # Fallback
                    avail = [p for p in range(v) if p not in block_set]
                    best_p = rng.choice(avail)
                block.append(best_p)
                block_set.add(best_p)
            
            block.sort()
            blocks.append(block)
            
            # Update uncovered
            for tid2 in get_ids(block):
                uncovered.discard(tid2)
        
        return blocks

    # Phase 1: Init
    best_blocks = None
    best_count = float('inf')
    
    for seed in range(20):
        if time.time() > deadline - 60: break
        blocks = greedy_init(seed)
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = blocks

    if best_blocks is None:
        best_blocks = greedy_init(0)
        best_count = len(best_blocks)

    # Phase 2: Local Search
    current_blocks = list(best_blocks)
    count = np.zeros(n_triplets, dtype=np.int32)
    
    def update_counts(blocks, delta):
        for b in blocks:
            for tid in get_ids(b):
                count[tid] += delta

    update_counts(current_blocks, 1)

    def is_redundant(b):
        for tid in get_ids(b):
            if count[tid] == 1:
                return False
        return True

    # Helper: Get critical points of a block
    def get_critical_points(b):
        pts = set()
        for tid in get_ids(b):
            if count[tid] == 1:
                pts.update(all_triplets[tid])
        return pts

    progress = True
    while progress and time.time() < deadline - 2:
        progress = False
        
        # 1. Remove redundant
        new_blocks = []
        for b in current_blocks:
            if is_redundant(b):
                update_counts([b], -1)
                progress = True
            else:
                new_blocks.append(b)
        current_blocks = new_blocks
        
        # 2. Swap to enable removal
        if not progress and len(current_blocks) > 0:
            # Try a few random swaps
            for _ in range(100): # Limit per outer loop
                if time.time() > deadline - 2: break
                
                idx = random.randint(0, len(current_blocks) - 1)
                b_old = current_blocks[idx]
                
                crit_pts = get_critical_points(b_old)
                
                if not crit_pts:
                    # Should have been removed, but maybe race condition?
                    continue
                    
                if len(crit_pts) > k:
                    continue # Cannot cover all critical points in one block
                
                # Construct candidate
                cand = list(crit_pts)
                while len(cand) < k:
                    p = random.randint(0, v - 1)
                    if p not in cand:
                        cand.append(p)
                cand.sort()
                
                # Check validity of swap
                # Condition: For all t in b_old, if t not in cand, count[t] must be >= 2
                valid = True
                cand_set = set(cand)
                for tid in get_ids(b_old):
                    if tid not in get_ids(cand): # This check is slow. Better to check points
                        # We need to check if the triplet is in cand
                        # Triplet is in cand if all 3 points are in cand_set
                        if not all(p in cand_set for p in all_triplets[tid]):
                            if count[tid] < 2:
                                valid = False
                                break
                if valid:
                    # Execute swap
                    update_counts([b_old], -1)
                    update_counts([cand], 1)
                    current_blocks[idx] = cand
                    progress = True # Force another removal pass

    return current_blocks
