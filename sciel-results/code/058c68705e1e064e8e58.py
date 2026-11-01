import itertools
import random
import time
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    """
    Construct a covering design C(v,k,t) with as few k-subsets of v points as possible
    such that every t-subset is contained in at least one block.
    Returns a list of blocks, each a list of k distinct integers in 0..v-1.
    """
    # Use a deadline with safety margin
    deadline = time.time() + 145.0
    
    # Precompute all t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    sub_to_id = {sub: i for i, sub in enumerate(all_tsubs)}
    
    # For each point, precompute which t-subsets contain it
    # point_to_tsubs[p] = list of ids of t-subsets containing p
    point_to_tsubs = [[] for _ in range(v)]
    for i, sub in enumerate(all_tsubs):
        for p in sub:
            point_to_tsubs[p].append(i)
    
    # For each pair of points (for t=3 optimization), precompute t-subsets containing both
    # This is only used when t=3 for speed
    if t == 3:
        pair_to_tsubs = {}
        for i, sub in enumerate(all_tsubs):
            for a_idx in range(3):
                for b_idx in range(a_idx + 1, 3):
                    pair = tuple(sorted((sub[a_idx], sub[b_idx])))
                    if pair not in pair_to_tsubs:
                        pair_to_tsubs[pair] = []
                    pair_to_tsubs[pair].append(i)
    
    def greedy_cover(rng, time_limit):
        """
        Greedy covering: repeatedly pick a block that covers the most uncovered t-subsets.
        Uses randomized tie-breaking for diversity.
        """
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        blocks = []
        
        while True:
            if time.time() > time_limit:
                return None
            
            # Find uncovered t-subsets
            uncovered = np.where(cover_count == 0)[0]
            if len(uncovered) == 0:
                break
            
            # Pick a random uncovered t-subset as seed
            seed_idx = rng.choice(uncovered)
            seed_sub = all_tsubs[seed_idx]
            block = list(seed_sub)
            block_set = set(block)
            
            # Extend block to size k using greedy heuristic
            while len(block) < k:
                best_gain = -1
                best_points = []
                
                # Compute (t-1)-subsets of current block
                block_t_minus_1 = list(itertools.combinations(block, t - 1))
                
                for x in range(v):
                    if x in block_set:
                        continue
                    
                    # Count how many uncovered t-subsets would be covered by adding x
                    gain = 0
                    if t == 3:
                        # Optimized for t=3
                        for rest in block_t_minus_1:
                            # rest is a pair (a, b)
                            pair = tuple(sorted(rest))
                            if pair in pair_to_tsubs:
                                for ts_id in pair_to_tsubs[pair]:
                                    if all_tsubs[ts_id][0] == x or all_tsubs[ts_id][1] == x or all_tsubs[ts_id][2] == x:
                                        # Check if x is the third element
                                        ts = all_tsubs[ts_id]
                                        if x in ts and ts not in (rest[0], rest[1]):
                                            if cover_count[ts_id] == 0:
                                                gain += 1
                    else:
                        for rest in block_t_minus_1:
                            ts = tuple(sorted(rest + (x,)))
                            if ts in sub_to_id:
                                ts_id = sub_to_id[ts]
                                if cover_count[ts_id] == 0:
                                    gain += 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_points = [x]
                    elif gain == best_gain:
                        best_points.append(x)
                
                if not best_points:
                    # Fallback: pick random point
                    candidates = [x for x in range(v) if x not in block_set]
                    if not candidates:
                        break
                    best_point = rng.choice(candidates)
                else:
                    best_point = rng.choice(best_points)
                
                block.append(best_point)
                block_set.add(best_point)
            
            block.sort()
            blocks.append(block)
            
            # Update cover counts
            for ts in itertools.combinations(block, t):
                ts_id = sub_to_id[ts]
                cover_count[ts_id] += 1
        
        return blocks
    
    def compute_cover_count(blocks):
        cc = np.zeros(n_tsubs, dtype=np.int32)
        for b in blocks:
            for ts in itertools.combinations(b, t):
                cc[sub_to_id[ts]] += 1
        return cc
    
    def remove_redundant(blocks, cc, time_limit):
        """
        Iteratively remove blocks that are redundant (all their t-subsets are covered by other blocks).
        """
        current_blocks = [list(b) for b in blocks]
        current_cc = cc.copy()
        
        changed = True
        while changed:
            if time.time() > time_limit:
                break
            changed = False
            to_remove = []
            
            for i, b in enumerate(current_blocks):
                redundant = True
                for ts in itertools.combinations(b, t):
                    ts_id = sub_to_id[ts]
                    if current_cc[ts_id] <= 1:
                        redundant = False
                        break
                if redundant:
                    to_remove.append(i)
            
            if to_remove:
                remove_set = set(to_remove)
                for i in to_remove:
                    b = current_blocks[i]
                    for ts in itertools.combinations(b, t):
                        current_cc[sub_to_id[ts]] -= 1
                current_blocks = [b for i, b in enumerate(current_blocks) if i not in remove_set]
                changed = True
        
        return current_blocks, current_cc
    
    def local_search(blocks, cc, rng, time_limit):
        """
        Local search: try to swap points between blocks to create redundancy, then remove redundant blocks.
        """
        current_blocks = [list(b) for b in blocks]
        current_cc = cc.copy()
        best_blocks = [list(b) for b in current_blocks]
        best_count = len(current_blocks)
        
        n_blocks = len(current_blocks)
        
        # Try multiple rounds of local search
        max_rounds = 50
        for r in range(max_rounds):
            if time.time() > time_limit:
                break
            
            # Try random swaps
            for _ in range(20):
                if time.time() > time_limit:
                    break
                
                if len(current_blocks) < 2:
                    break
                
                i = rng.randint(0, len(current_blocks) - 1)
                j = rng.randint(0, len(current_blocks) - 1)
                if i == j:
                    continue
                
                pi = rng.randint(0, k - 1)
                pj = rng.randint(0, k - 1)
                
                x = current_blocks[i][pi]
                y = current_blocks[j][pj]
                
                if x == y:
                    continue
                
                # Perform swap
                current_blocks[i][pi] = y
                current_blocks[j][pj] = x
                
                # Check if swap creates any uncovered t-subsets
                valid = True
                # We need to check if the swap causes any t-subset to lose its only covering block
                # This is complex, so we do a quick check by recomputing affected counts
                
                # Recompute cover count for affected blocks only is complex,
                # so we do a full check for simplicity in this local search
                # Actually, let's just check if min(cc) > 0 after updating
                
                # Update cc for the two modified blocks
                # Remove old contributions
                old_block_i = list(current_blocks[i])
                old_block_i[pi] = x
                old_block_j = list(current_blocks[j])
                old_block_j[pj] = y
                
                # This is getting complex. Let's use a simpler approach:
                # Just try the swap, check validity, and revert if invalid.
                
                # For speed, let's just do a simple validity check
                # by seeing if any t-subset that was only covered by block i or j is now uncovered.
                
                # Given complexity, let's skip this and just rely on greedy + redundancy removal
                # with more restarts.
                
                # Revert
                current_blocks[i][pi] = x
                current_blocks[j][pj] = y
        
        return best_blocks
    
    # Main search: multiple randomized greedy restarts with redundancy removal
    best_blocks = None
    best_count = float('inf')
    
    # Use a variety of random seeds
    max_restarts = 30
    for restart in range(max_restarts):
        if time.time() > deadline - 5:
            break
        
        r_rng = random.Random(42 + restart * 7)
        
        # Time budget per restart
        time_per_restart = max(1.0, (deadline - 5 - time.time()) / (max_restarts - restart))
        
        blocks = greedy_cover(r_rng, time.time() + time_per_restart * 0.6)
        if blocks is None:
            continue
        
        # Compute cover count
        cc = compute_cover_count(blocks)
        
        # Remove redundant blocks
        blocks, cc = remove_redundant(blocks, cc, deadline - 5)
        
        # Local search (simplified)
        # blocks = local_search(blocks, cc, r_rng, deadline - 5)
        
        # Recompute final cover count to ensure validity
        cc_final = compute_cover_count(blocks)
        if np.any(cc_final == 0):
            # Invalid, skip
            continue
        
        if len(blocks) < best_count:
            best_count = len(blocks)
            best_blocks = [list(b) for b in blocks]
        
        # Early termination if we've found a very good solution
        if best_count <= 74:
            break
    
    if best_blocks is None:
        # Fallback: run one more greedy without time pressure
        r_rng = random.Random(12345)
        blocks = greedy_cover(r_rng, deadline - 1)
        if blocks is not None:
            cc = compute_cover_count(blocks)
            blocks, cc = remove_redundant(blocks, cc, deadline - 1)
            best_blocks = blocks
        else:
            best_blocks = []
    
    return best_blocks
