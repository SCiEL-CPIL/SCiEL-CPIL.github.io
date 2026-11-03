import itertools
import random
import time

_PRECOMP = {}

def _precompute(v, k, t):
    if (v, k, t) in _PRECOMP:
        return _PRECOMP[(v, k, t)]
    
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {fs: i for i, fs in enumerate(all_tsubs)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_to_tsubs = []
    triple_to_blocks = [[] for _ in range(n_tsubs)]
    
    for b_idx, block in enumerate(all_blocks):
        covered_tsubs = []
        for c in itertools.combinations(block, t):
            idx = tsub_to_idx[frozenset(c)]
            covered_tsubs.append(idx)
            triple_to_blocks[idx].append(b_idx)
        block_to_tsubs.append(covered_tsubs)
        
    _PRECOMP[(v, k, t)] = {
        'all_blocks': all_blocks,
        'block_to_tsubs': block_to_tsubs,
        'triple_to_blocks': triple_to_blocks,
        'n_tsubs': n_tsubs
    }
    return _PRECOMP[(v, k, t)]

def construct(v, k, t):
    data = _precompute(v, k, t)
    all_blocks = data['all_blocks']
    block_to_tsubs = data['block_to_tsubs']
    triple_to_blocks = data['triple_to_blocks']
    n_tsubs = data['n_tsubs']
    
    rng = random.Random(12345)
    start_time = time.time()
    time_limit = 115.0
    
    def greedy_cover():
        uncovered = set(range(n_tsubs))
        blocks = []
        block_set = set()
        
        while uncovered:
            seed_tsub = rng.choice(list(uncovered))
            candidates = triple_to_blocks[seed_tsub]
            
            # Score candidates
            scores = []
            for b_idx in candidates:
                score = 0
                for t_idx in block_to_tsubs[b_idx]:
                    if t_idx in uncovered:
                        score += 1
                scores.append((score, b_idx))
            
            scores.sort(reverse=True)
            top_k = min(10, len(scores))
            if top_k > 0:
                chosen_score, chosen_block = rng.choice(scores[:top_k])
            else:
                continue
            
            if chosen_block not in block_set:
                blocks.append(chosen_block)
                block_set.add(chosen_block)
                for t_idx in block_to_tsubs[chosen_block]:
                    uncovered.discard(t_idx)
                
        return blocks

    def local_search(blocks):
        counts = [0] * n_tsubs
        for b_idx in blocks:
            for t_idx in block_to_tsubs[b_idx]:
                counts[t_idx] += 1
        
        block_set = set(blocks)
        blocks_list = list(blocks)
        
        while time.time() - start_time < time_limit:
            progress = False
            
            # Removal
            rng.shuffle(blocks_list)
            for b_idx in blocks_list:
                if b_idx not in block_set:
                    continue
                is_critical = False
                for t_idx in block_to_tsubs[b_idx]:
                    if counts[t_idx] == 1:
                        is_critical = True
                        break
                if not is_critical:
                    for t_idx in block_to_tsubs[b_idx]:
                        counts[t_idx] -= 1
                    blocks_list.remove(b_idx)
                    block_set.remove(b_idx)
                    progress = True
                    break
            
            if progress:
                continue
            
            # Swap
            rng.shuffle(blocks_list)
            for b_idx in blocks_list:
                if b_idx not in block_set:
                    continue
                criticals = [t_idx for t_idx in block_to_tsubs[b_idx] if counts[t_idx] == 1]
                if not criticals:
                    continue
                
                # Intersect
                criticals_sorted = sorted(criticals, key=lambda x: len(triple_to_blocks[x]))
                candidate_set = set(triple_to_blocks[criticals_sorted[0]])
                for t_idx in criticals_sorted[1:]:
                    candidate_set.intersection_update(triple_to_blocks[t_idx])
                    if not candidate_set:
                        break
                
                if not candidate_set:
                    continue
                
                candidate_set.discard(b_idx)
                candidate_set.difference_update(block_set)
                
                if not candidate_set:
                    continue
                
                new_b_idx = rng.choice(list(candidate_set))
                
                # Swap
                for t_idx in block_to_tsubs[b_idx]:
                    counts[t_idx] -= 1
                for t_idx in block_to_tsubs[new_b_idx]:
                    counts[t_idx] += 1
                
                blocks_list.remove(b_idx)
                blocks_list.append(new_b_idx)
                block_set.remove(b_idx)
                block_set.add(new_b_idx)
                
                progress = True
                break
            
            if progress:
                continue
            
            # Stuck
            break
            
        return blocks_list

    best_blocks = None
    best_count = float('inf')
    
    while time.time() - start_time < time_limit:
        init_blocks = greedy_cover()
        optimized_blocks = local_search(init_blocks)
        
        if len(optimized_blocks) < best_count:
            best_count = len(optimized_blocks)
            best_blocks = optimized_blocks
            
        if time.time() - start_time > time_limit - 5:
            break
        
    return [list(all_blocks[b_idx]) for b_idx in best_blocks]
