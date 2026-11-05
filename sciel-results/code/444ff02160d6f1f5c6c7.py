import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0
    
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
        
    triples_per_block = len(block_triples[0])
    block_triples_np = np.array(block_triples, dtype=np.int32)
    
    # Inverse index: for each triple, which blocks contain it?
    triple_to_blocks = [[] for _ in range(num_triples)]
    for b_idx in range(num_blocks):
        for t_idx in block_triples_np[b_idx]:
            triple_to_blocks[t_idx].append(b_idx)
    
    # Convert triple_to_blocks to numpy arrays for faster intersection
    triple_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in triple_to_blocks]
    
    def greedy_construct_idx(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        uncovered = set(range(num_triples))
        selected_indices = []
        
        while uncovered:
            # Find block covering most uncovered triples
            best_block = -1
            best_count = -1
            
            # Sample a subset of blocks to speed up
            if num_blocks > 1000:
                sample_size = min(num_blocks, 200)
                candidates = random.sample(range(num_blocks), sample_size)
            else:
                candidates = range(num_blocks)
            
            for b_idx in candidates:
                count = 0
                for t_idx in block_triples_np[b_idx]:
                    if t_idx in uncovered:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_block = b_idx
            
            if best_block == -1:
                break
                
            selected_indices.append(best_block)
            for t_idx in block_triples_np[best_block]:
                uncovered.discard(t_idx)
                
        return selected_indices

    def prune_indices(indices):
        if not indices:
            return [], np.zeros(num_triples, dtype=np.int32)
            
        cover_count = np.zeros(num_triples, dtype=np.int32)
        
        for b_idx in indices:
            cover_count[block_triples_np[b_idx]] += 1
            
        new_indices = []
        new_cover_count = np.zeros(num_triples, dtype=np.int32)
        
        for b_idx in indices:
            trs = block_triples_np[b_idx]
            # Check if any triple in this block is exclusively covered by it
            is_redundant = True
            for t_idx in trs:
                if cover_count[t_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                cover_count[trs] -= 1
            else:
                new_indices.append(b_idx)
                new_cover_count[trs] += 1
                
        return new_indices, new_cover_count

    def try_swap_remove(indices, cover, b_pos):
        """Try to remove block at b_pos and see if coverage is still valid."""
        b_idx = indices[b_pos]
        trs = block_triples_np[b_idx]
        
        # Check if any triple would become uncovered
        for t_idx in trs:
            if cover[t_idx] == 1:
                return None  # Can't remove
        
        # Safe to remove
        new_indices = indices[:b_pos] + indices[b_pos+1:]
        new_cover = cover.copy()
        new_cover[trs] -= 1
        return new_indices, new_cover

    def try_swap(indices, cover, b_pos, seed):
        """Try to replace block at b_pos with a better one."""
        random.seed(seed)
        np.random.seed(seed)
        
        b_idx = indices[b_pos]
        trs = block_triples_np[b_idx]
        
        # Find exclusive triples
        exclusive_triples = []
        for t_idx in trs:
            if cover[t_idx] == 1:
                exclusive_triples.append(t_idx)
                
        if not exclusive_triples:
            # Block is redundant, just remove it
            return try_swap_remove(indices, cover, b_pos)
        
        # Find candidate blocks that cover all exclusive triples
        candidate_set = None
        for t_idx in exclusive_triples:
            blocks_with_t = triple_to_blocks_np[t_idx]
            if candidate_set is None:
                candidate_set = set(blocks_with_t.tolist())
            else:
                candidate_set.intersection_update(blocks_with_t.tolist())
                
        if not candidate_set:
            return None
            
        candidate_set.discard(b_idx)
        
        if not candidate_set:
            return None
        
        # Score candidates: prefer blocks that overlap more with existing coverage
        # This creates more redundancy for future pruning
        best_candidate = None
        best_score = -1
        
        # Limit candidates for speed
        if len(candidate_set) > 50:
            sampled = random.sample(list(candidate_set), 50)
        else:
            sampled = list(candidate_set)
        
        for c_idx in sampled:
            c_trs = block_triples_np[c_idx]
            # Score: sum of cover counts for triples in c
            # Higher means more overlap = more redundancy potential
            score = np.sum(cover[c_trs])
            
            # Also add a small random component for diversity
            score += random.random() * 0.1
            
            if score > best_score:
                best_score = score
                best_candidate = c_idx
        
        if best_candidate is None:
            return None
            
        # Perform the swap
        new_indices = indices.copy()
        new_indices[b_pos] = best_candidate
        new_cover = cover.copy()
        new_cover[trs] -= 1
        new_cover[block_triples_np[best_candidate]] += 1
        
        return new_indices, new_cover

    # Initial solution: try multiple greedy runs with different seeds
    best_indices = None
    best_cover = None
    best_size = float('inf')
    
    for seed in [42, 123, 456, 789, 1000]:
        if time.time() > deadline - 30:
            break
        indices = greedy_construct_idx(seed)
        indices, cover = prune_indices(indices)
        if len(indices) < best_size:
            best_size = len(indices)
            best_indices = indices
            best_cover = cover
    
    if best_indices is None:
        best_indices = greedy_construct_idx(42)
        best_indices, best_cover = prune_indices(best_indices)
        best_size = len(best_indices)
    
    current_indices = best_indices.copy()
    current_cover = best_cover.copy()
    
    stuck_count = 0
    iteration = 0
    
    while time.time() < deadline:
        iteration += 1
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
        
        n_blocks = len(current_indices)
        if n_blocks == 0:
            break
        
        # 2. Try multiple random swaps
        attempts = 0
        max_attempts = 20
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            idx_pos = np.random.randint(n_blocks)
            
            # First try simple removal
            result = try_swap_remove(current_indices, current_cover, idx_pos)
            if result is not None:
                current_indices, current_cover = result
                if len(current_indices) < best_size:
                    best_size = len(current_indices)
                    best_indices = current_indices.copy()
                    best_cover = current_cover.copy()
                improved = True
                stuck_count = 0
                break
            
            # Then try swap
            seed = hash((iteration, idx_pos, attempts)) % 100000
            result = try_swap(current_indices, current_cover, idx_pos, seed)
            if result is not None:
                new_indices, new_cover = result
                # Prune after swap
                new_indices, new_cover = prune_indices(new_indices)
                
                if len(new_indices) < len(current_indices):
                    current_indices = new_indices
                    current_cover = new_cover
                    if len(current_indices) < best_size:
                        best_size = len(current_indices)
                        best_indices = current_indices.copy()
                        best_cover = current_cover.copy()
                    improved = True
                    stuck_count = 0
                    break
                elif len(new_indices) == len(current_indices):
                    # Same size but different structure - accept with some probability
                    if random.random() < 0.3:
                        current_indices = new_indices
                        current_cover = new_cover
                    stuck_count += 1
                else:
                    stuck_count += 1
            
            if stuck_count > 50:
                # Restart from best
                current_indices = best_indices.copy()
                current_cover = best_cover.copy()
                stuck_count = 0
                break
        
        if not improved:
            stuck_count += 1
            if stuck_count > 100:
                # Full restart with new seed
                seed = random.randint(0, 100000)
                new_indices = greedy_construct_idx(seed)
                new_indices, new_cover = prune_indices(new_indices)
                if len(new_indices) < best_size:
                    best_size = len(new_indices)
                    best_indices = new_indices.copy()
                    best_cover = new_cover.copy()
                current_indices = best_indices.copy()
                current_cover = best_cover.copy()
                stuck_count = 0

    # Final prune
    best_indices, best_cover = prune_indices(best_indices)
    
    return [list(all_blocks[b]) for b in best_indices]
