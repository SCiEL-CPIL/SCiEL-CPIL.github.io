import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 118.0
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute all blocks and their triples
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    # block_triples[b] is a list of triple indices
    block_triples = []
    for b in all_blocks:
        trs = [triple_to_idx[tr] for tr in itertools.combinations(b, t)]
        block_triples.append(trs)
        
    # Convert to numpy array for vectorization? 
    # It's a list of lists. We can pad them.
    # All blocks have exactly C(k,t) triples.
    triples_per_block = len(block_triples[0])
    # block_triples_np shape: (num_blocks, triples_per_block)
    block_triples_np = np.array(block_triples, dtype=np.int32)
    
    # Inverse index: for each triple, which blocks contain it?
    # triple_to_blocks[t] is a list of block indices
    triple_to_blocks = [[] for _ in range(num_triples)]
    for b_idx in range(num_blocks):
        for t_idx in block_triples_np[b_idx]:
            triple_to_blocks[t_idx].append(b_idx)
            
    def greedy_construct_idx(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        uncovered_count = np.ones(num_triples, dtype=np.int32)
        gain = np.full(num_blocks, triples_per_block, dtype=np.int32)
        
        selected_indices = []
        
        while uncovered_count.sum() > 0:
            max_gain = np.max(gain)
            candidates = np.where(gain == max_gain)[0]
            b_star = candidates[np.random.randint(len(candidates))]
            
            selected_indices.append(b_star)
            
            for t_idx in block_triples_np[b_star]:
                if uncovered_count[t_idx] > 0:
                    uncovered_count[t_idx] = 0
                    gain[np.array(triple_to_blocks[t_idx])] -= 1
                    
        return selected_indices

    def prune_indices(indices):
        if not indices:
            return [], np.zeros(num_triples, dtype=np.int32)
            
        cover_count = np.zeros(num_triples, dtype=np.int32)
        
        # Count coverage
        for b_idx in indices:
            cover_count[block_triples_np[b_idx]] += 1
            
        new_indices = []
        new_cover_count = np.zeros(num_triples, dtype=np.int32)
        
        for b_idx in indices:
            # Check if any triple in this block is exclusively covered by it
            # i.e., cover_count[t] == 1
            trs = block_triples_np[b_idx]
            is_redundant = True
            for t_idx in trs:
                if cover_count[t_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                # Remove
                cover_count[trs] -= 1
            else:
                new_indices.append(b_idx)
                new_cover_count[trs] += 1
                
        return new_indices, new_cover_count

    # Initial solution
    best_indices = greedy_construct_idx(42)
    best_indices, best_cover = prune_indices(best_indices)
    best_size = len(best_indices)
    
    stuck_count = 0
    current_indices = best_indices.copy()
    current_cover = best_cover.copy()
    
    while time.time() < deadline:
        improved = False
        
        # 1. Prune
        prev_size = len(current_indices)
        current_indices, current_cover = prune_indices(current_indices)
        if len(current_indices) < prev_size:
            improved = True
            if len(current_indices) < best_size:
                best_size = len(current_indices)
                best_indices = current_indices.copy()
                best_cover = current_cover.copy()
            stuck_count = 0
            continue
            
        # 2. Local Search: Swap
        n_blocks = len(current_indices)
        if n_blocks == 0:
            break
            
        # Pick a random block to replace
        idx_pos = np.random.randint(n_blocks)
        b_idx = current_indices[idx_pos]
        
        # Find exclusive triples of b_idx
        # Exclusive if current_cover[t] == 1
        trs = block_triples_np[b_idx]
        exclusive_triples = []
        for t_idx in trs:
            if current_cover[t_idx] == 1:
                exclusive_triples.append(t_idx)
                
        if not exclusive_triples:
            # Should have been pruned. Force remove.
            current_indices.pop(idx_pos)
            current_indices, current_cover = prune_indices(current_indices)
            if len(current_indices) < best_size:
                best_size = len(current_indices)
                best_indices = current_indices.copy()
                best_cover = current_cover.copy()
            stuck_count = 0
            continue
            
        # Find candidate blocks that cover all exclusive_triples
        # Intersection of triple_to_blocks for each exclusive triple
        candidate_set = None
        for t_idx in exclusive_triples:
            blocks_with_t = set(triple_to_blocks[t_idx])
            if candidate_set is None:
                candidate_set = blocks_with_t
            else:
                candidate_set.intersection_update(blocks_with_t)
                
        if not candidate_set:
            stuck_count += 1
            if stuck_count > 50:
                # Restart greedy
                new_indices = greedy_construct_idx(random.randint(0, 10000))
                new_indices, new_cover = prune_indices(new_indices)
                if len(new_indices) < best_size:
                    best_size = len(new_indices)
                    best_indices = new_indices.copy()
                    best_cover = new_cover.copy()
                current_indices = best_indices.copy()
                current_cover = best_cover.copy()
                stuck_count = 0
            continue
            
        # Exclude current block
        candidate_set.discard(b_idx)
        
        if not candidate_set:
            stuck_count += 1
            continue
            
        # Score candidates
        # We want to maximize the number of "redundant" triples created.
        # A triple t in candidate block c is "helpful" if current_cover[t] > 1 
        # (meaning it's covered by others, so adding c makes it more redundant)
        # BUT we must be careful: current_cover includes b_idx.
        # If t is in b_idx, current_cover[t] >= 1.
        # If we replace b_idx with c, the new count for t is current_cover[t] - (1 if t in b_idx else 0) + (1 if t in c else 0).
        # We want to maximize the potential for future pruning.
        # A good proxy: sum of current_cover[t] for t in c.
        # Higher sum means more overlap with existing coverage.
        
        best_candidate = None
        best_score = -1
        
        # Limit candidates to avoid timeout
        # candidate_set size can be large.
        # Let's sample if too large?
        # Or just iterate if small.
        # Max size of candidate_set:
        # If exclusive_triples is 1, size is 5985.
        # If exclusive_triples is 3, size is smaller.
        
        for c_idx in candidate_set:
            # Calculate score
            score = 0
            c_trs = block_triples_np[c_idx]
            for t_idx in c_trs:
                score += current_cover[t_idx]
                
            # Tie break randomly
            if score > best_score:
                best_score = score
                best_candidate = c_idx
            elif score == best_score and random.random() < 0.1:
                best_candidate = c_idx
                
        if best_candidate is not None:
            # Perform swap
            current_indices[idx_pos] = best_candidate
            # Prune
            prev_size = len(current_indices)
            current_indices, current_cover = prune_indices(current_indices)
            
            if len(current_indices) < prev_size:
                improved = True
                if len(current_indices) < best_size:
                    best_size = len(current_indices)
                    best_indices = current_indices.copy()
                    best_cover = current_cover.copy()
                stuck_count = 0
            else:
                # No improvement, revert swap
                current_indices[idx_pos] = b_idx
                current_indices, current_cover = prune_indices(current_indices)
                stuck_count += 1
                if stuck_count > 100:
                    # Perturb: random restart or random swap
                    # Let's try a random valid swap without score check
                    # Or just restart
                    if random.random() < 0.2:
                        new_indices = greedy_construct_idx(random.randint(0, 10000))
                        new_indices, new_cover = prune_indices(new_indices)
                        if len(new_indices) < best_size:
                            best_size = len(new_indices)
                            best_indices = new_indices.copy()
                            best_cover = new_cover.copy()
                        current_indices = best_indices.copy()
                        current_cover = best_cover.copy()
                    stuck_count = 0

    # Convert best_indices back to list of blocks
    return [list(all_blocks[b]) for b in best_indices]
