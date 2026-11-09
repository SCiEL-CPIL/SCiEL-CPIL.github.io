"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random


def construct(v, k, t):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    rng = random.Random(42)
    
    # Precompute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    n_tsubs = len(all_tsubs)
    
    # For each point, precompute which t-subset indices it participates in
    point_tsub_indices = [[] for _ in range(v)]
    for i, ts in enumerate(all_tsubs):
        for p in ts:
            point_tsub_indices[p].append(i)
    
    # Greedy covering with fast gain calculation
    uncovered = bytearray([1] * n_tsubs)
    blocks = []
    
    while any(uncovered):
        # Find first uncovered t-subset
        first_uncovered = None
        for i in range(n_tsubs):
            if uncovered[i]:
                first_uncovered = i
                break
        if first_uncovered is None:
            break
        
        block = list(all_tsubs[first_uncovered])
        
        while len(block) < k:
            # Calculate gain for each candidate point
            best_gain = -1
            best_point = -1
            block_set = set(block)
            
            # Precompute which t-subsets in block are uncovered
            block_uncovered_mask = [False] * n_tsubs
            for combo in itertools.combinations(block, t - 1):
                for p in range(v):
                    if p not in block_set:
                        ts = tuple(sorted(combo + (p,)))
                        idx = tsub_to_idx.get(ts)
                        if idx is not None and uncovered[idx]:
                            block_uncovered_mask[idx] = True
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(ts)
                    if idx is not None and uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            
            if best_point >= 0:
                block.append(best_point)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for combo in itertools.combinations(block, t):
            idx = tsub_to_idx.get(combo)
            if idx is not None:
                uncovered[idx] = 0
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Local search: try to remove blocks
    def can_remove_block(blocks, block_idx, uncovered_check):
        """Check if removing block_idx leaves all t-subsets covered."""
        block = blocks[block_idx]
        # Find t-subsets only covered by this block
        for combo in itertools.combinations(block, t):
            idx = tsub_to_idx.get(combo)
            if idx is None:
                continue
            # Check if any other block covers this t-subset
            covered = False
            for j, other_block in enumerate(blocks):
                if j == block_idx:
                    continue
                if all(p in other_block for p in combo):
                    covered = True
                    break
            if not covered:
                return False
        return True
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline:
                break
            if can_remove_block(blocks, i, uncovered):
                blocks.pop(i)
                improved = True
    
    return blocks
# EVOLVE-BLOCK-END
