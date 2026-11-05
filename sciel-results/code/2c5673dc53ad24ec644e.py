import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = 112.0
    
    points = list(range(v))
    all_triples = list(itertools.combinations(points, t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    all_blocks = list(itertools.combinations(points, k))
    num_blocks = len(all_blocks)
    
    block_triples = []
    for b in all_blocks:
        trs = [triple_to_idx[tr] for tr in itertools.combinations(b, t)]
        block_triples.append(trs)
        
    triples_per_block = len(block_triples[0])
    block_triples_np = np.array(block_triples, dtype=np.int32)
    
    triple_to_blocks = [[] for _ in range(num_triples)]
    for b_idx in range(num_blocks):
        for t_idx in block_triples_np[b_idx]:
            triple_to_blocks[t_idx].append(b_idx)
    
    triple_to_blocks_np = [np.array(triple_to_blocks[t_idx], dtype=np.int32) for t_idx in range(num_triples)]
    
    def greedy_construct_idx(seed, weight_strategy='uniform'):
        random.seed(seed)
        np.random.seed(seed)
        
        uncovered_count = np.ones(num_triples, dtype=np.int32)
        gain = np.full(num_blocks, triples_per_block, dtype=np.int32)
        
        selected_indices = []
        
        while uncovered_count.sum() > 0:
            if weight_strategy == 'random_tie':
                max_gain = np.max(gain)
                candidates = np.where(gain == max_gain)[0]
                b_star = candidates[np.random.randint(len(candidates))]
            else:
                b_star = np.argmax(gain)
            
            selected_indices.append(b_star)
            
            for t_idx in block_triples_np[b_star]:
                if uncovered_count[t_idx] > 0:
                    uncovered_count[t_idx] = 0
                    gain[triple_to_blocks_np[t_idx]] -= 1
                    
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
            has_exclusive = np.any(cover_count[trs] == 1)
            
            if not has_exclusive:
                cover_count[trs] -= 1
            else:
                new_indices.append(b_idx)
                new_cover_count[trs] += 1
                
        return new_indices, new_cover_count

    def local_search(indices, cover, max_iters=2000):
        current_indices = indices.copy()
        current_cover = cover.copy()
        best_size = len(current_indices)
        best_indices = current_indices.copy()
        best_cover = current_cover.copy()
        
        stuck = 0
        n = len(current_indices)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Prune
            current_indices, current_cover = prune_indices(current_indices)
            if len(current_indices) < best_size:
                best_size = len(current_indices)
                best_indices = current_indices.copy()
                best_cover = current_cover.copy()
                stuck = 0
                n = len(current_indices)
                if n == 0:
                    break
                continue
            
            # Try swap
            if n == 0:
                break
                
            idx_pos = np.random.randint(n)
            b_idx = current_indices[idx_pos]
            trs = block_triples_np[b_idx]
            exclusive_mask = (current_cover[trs] == 1)
            
            if not np.any(exclusive_mask):
                current_indices.pop(idx_pos)
                current_indices, current_cover = prune_indices(current_indices)
                if len(current_indices) < best_size:
                    best_size = len(current_indices)
                    best_indices = current_indices.copy()
                    best_cover = current_cover.copy()
                stuck = 0
                n = len(current_indices)
                continue
                
            exclusive_triples = trs[exclusive_mask].tolist()
            
            candidate_set = None
            for t_idx in exclusive_triples:
                blocks_with_t = set(triple_to_blocks[t_idx])
                if candidate_set is None:
                    candidate_set = blocks_with_t
                else:
                    candidate_set.intersection_update(blocks_with_t)
                    
            if candidate_set is None or not candidate_set:
                stuck += 1
                if stuck > 30:
                    break
                continue
                
            candidate_set.discard(b_idx)
            
            if not candidate_set:
                stuck += 1
                continue
                
            cand_arr = np.array(list(candidate_set), dtype=np.int32)
            scores = np.zeros(len(cand_arr), dtype=np.int32)
            for i, c_idx in enumerate(cand_arr):
                scores[i] = np.sum(current_cover[block_triples_np[c_idx]])
                
            max_score_idx = np.argmax(scores)
            best_score = scores[max_score_idx]
            tied = np.where(scores == best_score)[0]
            if len(tied) > 1:
                best_candidate = cand_arr[tied[random.randint(len(tied))]]
            else:
                best_candidate = cand_arr[max_score_idx]
                
            if best_candidate != b_idx:
                old_block = current_indices[idx_pos]
                current_indices[idx_pos] = best_candidate
                prev_size = len(current_indices)
                current_indices, current_cover = prune_indices(current_indices)
                
                if len(current_indices) < prev_size:
                    stuck = 0
                    n = len(current_indices)
                    if len(current_indices) < best_size:
                        best_size = len(current_indices)
                        best_indices = current_indices.copy()
                        best_cover = current_cover.copy()
                else:
                    current_indices[idx_pos] = old_block
                    current_indices, current_cover = prune_indices(current_indices)
                    stuck += 1
                    if stuck > 30:
                        break
        
        return best_indices, best_cover

    # Initial greedy with random tie-breaking
    best_indices = greedy_construct_idx(42, 'random_tie')
    best_indices, best_cover = prune_indices(best_indices)
    best_size = len(best_indices)
    
    # Local search phase
    while time.time() < deadline - 5:
        current_indices = best_indices.copy()
        current_cover = best_cover.copy()
        
        # Run local search
        ls_indices, ls_cover = local_search(current_indices, current_cover, max_iters=500)
        
        if len(ls_indices) < best_size:
            best_size = len(ls_indices)
            best_indices = ls_indices.copy()
            best_cover = ls_cover.copy()
            
    # Try a few more restarts if time permits
    while time.time() < deadline - 10:
        seed = random.randint(0, 100000)
        new_indices = greedy_construct_idx(seed, 'random_tie')
        new_indices, new_cover = prune_indices(new_indices)
        
        if len(new_indices) < best_size:
            best_size = len(new_indices)
            best_indices = new_indices.copy()
            best_cover = new_cover.copy()
            
            # Local search on the new best
            while time.time() < deadline - 5:
                ls_indices, ls_cover = local_search(best_indices.copy(), best_cover.copy(), max_iters=300)
                if len(ls_indices) < best_size:
                    best_size = len(ls_indices)
                    best_indices = ls_indices.copy()
                    best_cover = ls_cover.copy()
                else:
                    break
    
    return [list(all_blocks[b]) for b in best_indices]
