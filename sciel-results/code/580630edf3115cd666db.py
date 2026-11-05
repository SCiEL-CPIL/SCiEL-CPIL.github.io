import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    num_triples = len(triples)
    
    blocks = list(itertools.combinations(points, k))
    num_blocks = len(blocks)
    
    # block_triples[b] is array of triple indices
    block_triples = np.empty((num_blocks, len(list(itertools.combinations(range(k), t)))), dtype=np.int32)
    # Wait, C(k,t) is fixed.
    c_k_t = len(list(itertools.combinations(range(k), t)))
    block_triples = np.empty((num_blocks, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(blocks):
        idxs = [triple_to_idx[sub] for sub in itertools.combinations(block, t)]
        block_triples[i] = idxs
        
    # triple_to_blocks
    triple_to_blocks = [[] for _ in range(num_triples)]
    for b_idx in range(num_blocks):
        for t_idx in block_triples[b_idx]:
            triple_to_blocks[t_idx].append(b_idx)

    def greedy_init(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        uncovered = np.ones(num_triples, dtype=bool)
        selected = []
        
        while uncovered.any():
            if time.time() > deadline:
                break
            # Vectorized gain calculation
            gains = uncovered[block_triples].sum(axis=1)
            max_gain = np.max(gains)
            if max_gain == 0: break
            # Pick random from top candidates to add diversity
            # Find all with max_gain
            candidates = np.where(gains == max_gain)[0]
            # If too many, pick random subset?
            # For now, just pick one randomly from max
            b_idx = candidates[np.random.randint(len(candidates))]
            selected.append(b_idx)
            uncovered[block_triples[b_idx]] = False
            
        return selected

    def prune(indices):
        if not indices:
            return [], np.zeros(num_triples, dtype=np.int32)
            
        cover_count = np.zeros(num_triples, dtype=np.int32)
        for b_idx in indices:
            cover_count[block_triples[b_idx]] += 1
            
        active = [True] * len(indices)
        changed = True
        while changed:
            changed = False
            for i, b_idx in enumerate(indices):
                if not active[i]:
                    continue
                # Check if redundant
                # If min cover count for this block's triples is > 1, it's redundant
                # Note: cover_count includes this block. So if min > 1, it means at least one OTHER block covers it.
                # Actually, if min(cover_count[trs]) > 1, then for every t in trs, there is at least one other block.
                # So yes, redundant.
                trs = block_triples[b_idx]
                if np.min(cover_count[trs]) > 1:
                    active[i] = False
                    cover_count[trs] -= 1
                    changed = True
                    
        new_indices = [indices[i] for i in range(len(indices)) if active[i]]
        return new_indices, cover_count

    # Initial Solution
    best_indices = greedy_init()
    best_indices, best_cover = prune(best_indices)
    best_size = len(best_indices)
    
    current_indices = best_indices.copy()
    current_cover = best_cover.copy()
    
    stuck_count = 0
    
    while time.time() < deadline:
        improved = False
        
        # 1. Prune (in case of changes from previous step)
        prev_size = len(current_indices)
        current_indices, current_cover = prune(current_indices)
        if len(current_indices) < prev_size:
            improved = True
            if len(current_indices) < best_size:
                best_size = len(current_indices)
                best_indices = current_indices.copy()
                best_cover = current_cover.copy()
            stuck_count = 0
            continue

        # 2. Local Search: Swap
        if not current_indices:
            break
            
        n_blocks = len(current_indices)
        # Pick a random position to swap
        pos = np.random.randint(n_blocks)
        b_idx = current_indices[pos]
        
        # Find exclusive triples
        exclusive = []
        trs_b = block_triples[b_idx]
        for t_idx in trs_b:
            if current_cover[t_idx] == 1:
                exclusive.append(t_idx)
                
        if not exclusive:
            # Should have been pruned, but if not, remove it
            current_indices.pop(pos)
            current_indices, current_cover = prune(current_indices)
            if len(current_indices) < best_size:
                best_size = len(current_indices)
                best_indices = current_indices.copy()
                best_cover = current_cover.copy()
            stuck_count = 0
            continue
            
        # Find candidates
        cand_set = set(triple_to_blocks[exclusive[0]])
        for t_idx in exclusive[1:]:
            cand_set.intersection_update(triple_to_blocks[t_idx])
        cand_set.discard(b_idx)
        cands = list(cand_set)
        
        if not cands:
            stuck_count += 1
            if stuck_count > 20:
                # Perturb
                # Remove random block
                rem_pos = np.random.randint(len(current_indices))
                rem_idx = current_indices[rem_pos]
                current_indices.pop(rem_pos)
                current_cover[block_triples[rem_idx]] -= 1
                
                # Greedy fill
                uncovered = current_cover == 0
                # Use a fast greedy loop
                while uncovered.any():
                    gains = uncovered[block_triples].sum(axis=1)
                    max_gain = np.max(gains)
                    if max_gain == 0: break
                    candidates = np.where(gains == max_gain)[0]
                    nb = candidates[np.random.randint(len(candidates))]
                    current_indices.append(nb)
                    current_cover[block_triples[nb]] += 1
                    uncovered[block_triples[nb]] = False
                    
                current_indices, current_cover = prune(current_indices)
                if len(current_indices) < best_size:
                    best_size = len(current_indices)
                    best_indices = current_indices.copy()
                    best_cover = current_cover.copy()
                stuck_count = 0
            continue
            
        # Score candidates
        # We want to maximize overlap with existing cover (excluding b_idx)
        # Score(C) = sum(current_cover[t] for t in block_triples[C])
        # This is a proxy for redundancy potential.
        
        # Vectorized scoring
        # cands is a list of indices.
        # block_triples[cands] gives shape (len(cands), 35)
        # current_cover[block_triples[cands]] gives shape (len(cands), 35)
        # sum(axis=1) gives scores.
        
        # To avoid huge memory if cands is large (max 5985), 5985*35 is ~200k ints. Fine.
        cand_arr = np.array(cands, dtype=np.int32)
        scores = current_cover[block_triples[cand_arr]].sum(axis=1)
        
        best_cand_score = np.max(scores)
        best_cands = cand_arr[scores == best_cand_score]
        best_cand = best_cands[np.random.randint(len(best_cands))]
        
        # Perform Swap
        current_indices[pos] = best_cand
        # Update cover count?
        # Easier to just re-prune which recalculates cover count.
        # But prune takes a list and returns new list and count.
        # It's O(N * 35). N is ~100. Very fast.
        
        prev_size = len(current_indices)
        current_indices, current_cover = prune(current_indices)
        
        if len(current_indices) < prev_size:
            improved = True
            if len(current_indices) < best_size:
                best_size = len(current_indices)
                best_indices = current_indices.copy()
                best_cover = current_cover.copy()
            stuck_count = 0
        else:
            # Revert
            current_indices[pos] = b_idx
            current_indices, current_cover = prune(current_indices)
            stuck_count += 1
            
        if stuck_count > 50:
            # Reset to best and perturb
            current_indices = best_indices.copy()
            current_cover = best_cover.copy()
            stuck_count = 0
            # Force a perturbation next time?
            # Or just continue.
            
    return [list(blocks[b]) for b in best_indices]
