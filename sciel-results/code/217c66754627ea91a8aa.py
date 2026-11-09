import itertools
import time
import random

_CACHE = {}

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    key = (v, k, t)
    if key not in _CACHE:
        points = list(range(v))
        all_blocks_list = list(itertools.combinations(points, k))
        n_blocks = len(all_blocks_list)
        
        all_t_subsets_list = list(itertools.combinations(points, t))
        n_subsets = len(all_t_subsets_list)
        
        subset_to_idx = {s: i for i, s in enumerate(all_t_subsets_list)}
        
        block_to_subsets = []
        subset_to_blocks = [[] for _ in range(n_subsets)]
        
        for b_idx, block in enumerate(all_blocks_list):
            subs = []
            for sub in itertools.combinations(block, t):
                s_idx = subset_to_idx[sub]
                subs.append(s_idx)
                subset_to_blocks[s_idx].append(b_idx)
            block_to_subsets.append(subs)
        
        _CACHE[key] = {
            'all_blocks_list': all_blocks_list,
            'n_blocks': n_blocks,
            'n_subsets': n_subsets,
            'block_to_subsets': block_to_subsets,
            'subset_to_blocks': subset_to_blocks
        }
    
    data = _CACHE[key]
    all_blocks_list = data['all_blocks_list']
    n_subsets = data['n_subsets']
    block_to_subsets = data['block_to_subsets']
    subset_to_blocks = data['subset_to_blocks']
    
    best_blocks = None
    best_count = float('inf')
    
    while time.time() < deadline:
        blocks, count = _solve_once(v, k, t, data)
        
        if count < best_count:
            best_count = count
            best_blocks = blocks
            if best_count <= 335:
                break

    if best_blocks is None:
        # Fallback
        blocks, _ = _solve_once(v, k, t, data)
        return blocks

    return best_blocks

def _solve_once(v, k, t, data):
    n_subsets = data['n_subsets']
    all_blocks_list = data['all_blocks_list']
    block_to_subsets = data['block_to_subsets']
    subset_to_blocks = data['subset_to_blocks']
    
    # 1. Greedy
    uncovered = set(range(n_subsets))
    counts = [0] * n_subsets
    current_blocks = []
    selected_set = set()
    
    while uncovered:
        # Pick random uncovered target
        target_idx = random.choice(list(uncovered))
        candidates = subset_to_blocks[target_idx]
        
        best_gain = -1
        best_cands = []
        
        for b_idx in candidates:
            gain = 0
            for s_idx in block_to_subsets[b_idx]:
                if s_idx in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_cands = [b_idx]
            elif gain == best_gain:
                best_cands.append(b_idx)
        
        chosen = random.choice(best_cands)
        current_blocks.append(chosen)
        selected_set.add(chosen)
        
        for s_idx in block_to_subsets[chosen]:
            counts[s_idx] += 1
            if counts[s_idx] == 1:
                uncovered.discard(s_idx)
                
    # 2. Reduction
    _reduce_blocks(current_blocks, counts, block_to_subsets, selected_set)
    
    # 3. Swaps
    # Try to improve by swapping
    # Limit swaps to avoid infinite loops or too much time
    swap_limit = 100
    for _ in range(swap_limit):
        if not current_blocks:
            break
        b_out_idx = random.randint(0, len(current_blocks) - 1)
        b_out = current_blocks[b_out_idx]
        
        unique_subsets = [s for s in block_to_subsets[b_out] if counts[s] == 1]
        
        if not unique_subsets:
            # Redundant
            for s in block_to_subsets[b_out]:
                counts[s] -= 1
            current_blocks.pop(b_out_idx)
            selected_set.discard(b_out)
            continue
            
        # Find b_in covering all unique_subsets
        target_s = unique_subsets[0]
        candidates = subset_to_blocks[target_s]
        
        valid_in = []
        for b_in in candidates:
            if b_in in selected_set:
                continue
            # Check coverage of all unique_subsets
            ok = True
            for s in unique_subsets:
                # Check if s is in block_to_subsets[b_in]
                # O(35) search
                if s not in block_to_subsets[b_in]:
                    ok = False
                    break
            if ok:
                valid_in.append(b_in)
                
        if valid_in:
            b_in = random.choice(valid_in)
            # Swap
            for s in block_to_subsets[b_out]:
                counts[s] -= 1
            current_blocks.pop(b_out_idx)
            selected_set.discard(b_out)
            
            for s in block_to_subsets[b_in]:
                counts[s] += 1
            current_blocks.append(b_in)
            selected_set.add(b_in)
            
            # Reduce after swap
            _reduce_blocks(current_blocks, counts, block_to_subsets, selected_set)

    return [list(all_blocks_list[b]) for b in current_blocks], len(current_blocks)

def _reduce_blocks(blocks, counts, block_to_subsets, selected_set):
    changed = True
    while changed:
        changed = False
        random.shuffle(blocks)
        for i in range(len(blocks)):
            b = blocks[i]
            # Check redundancy
            is_red = True
            for s in block_to_subsets[b]:
                if counts[s] == 1:
                    is_red = False
                    break
            if is_red:
                for s in block_to_subsets[b]:
                    counts[s] -= 1
                blocks.pop(i)
                selected_set.discard(b)
                changed = True
                break
