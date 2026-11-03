import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140
    
    def elapsed():
        return time.time() - start_time

    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    num_t = len(t_subsets)
    t_index = {s: i for i, s in enumerate(t_subsets)}
    
    k_blocks = list(itertools.combinations(points, k))
    num_k = len(k_blocks)
    
    block_t_indices = []
    for block in k_blocks:
        covered = set()
        for sub in itertools.combinations(block, t):
            covered.add(t_index[sub])
        block_t_indices.append(list(covered))
        
    t_to_blocks = [set() for _ in range(num_t)]
    for b_idx, subs in enumerate(block_t_indices):
        for s_idx in subs:
            t_to_blocks[s_idx].add(b_idx)

    if elapsed() > deadline - 60:
        return [list(k_blocks[0])] 

    # Greedy Initialization
    t_weights = np.zeros(num_t, dtype=np.float64)
    for i in range(num_t):
        count = len(t_to_blocks[i])
        if count > 0:
            t_weights[i] = 1.0 / count
            
    gain = np.zeros(num_k, dtype=np.float64)
    for i in range(num_k):
        if block_t_indices[i]:
            gain[i] = np.sum(t_weights[block_t_indices[i]])
            
    uncovered = np.ones(num_t, dtype=np.bool_)
    uncovered_count = num_t
    selected = []
    selected_mask = np.zeros(num_k, dtype=np.bool_)
    
    while uncovered_count > 0 and elapsed() < deadline - 80:
        current_gains = gain.copy()
        current_gains[selected_mask] = -1.0
        best_idx = np.argmax(current_gains)
        
        if gain[best_idx] <= 0:
            break
            
        selected.append(best_idx)
        selected_mask[best_idx] = True
        
        for idx in block_t_indices[best_idx]:
            if uncovered[idx]:
                uncovered[idx] = False
                uncovered_count -= 1
                w = t_weights[idx]
                for b in t_to_blocks[idx]:
                    if not selected_mask[b]:
                        gain[b] -= w
                        
    cover_count = np.zeros(num_t, dtype=np.int32)
    for b_idx in selected:
        for s_idx in block_t_indices[b_idx]:
            cover_count[s_idx] += 1
            
    best_size = len(selected)
    
    def try_drop_blocks():
        nonlocal selected, cover_count
        changed = True
        while changed and elapsed() < deadline - 5:
            changed = False
            new_selected = []
            for b_idx in selected:
                needed = False
                for s_idx in block_t_indices[b_idx]:
                    if cover_count[s_idx] == 1:
                        needed = True
                        break
                if needed:
                    new_selected.append(b_idx)
                else:
                    changed = True
                    for s_idx in block_t_indices[b_idx]:
                        cover_count[s_idx] -= 1
            if changed:
                selected = new_selected
                return True
        return False

    def try_swap():
        nonlocal selected, cover_count
        improved = False
        for i in range(len(selected)):
            if elapsed() > deadline - 5:
                break
            b_out = selected[i]
            unique_subs = [s_idx for s_idx in block_t_indices[b_out] if cover_count[s_idx] == 1]
            if not unique_subs:
                continue
            candidates = None
            for s_idx in unique_subs:
                if candidates is None:
                    candidates = set(t_to_blocks[s_idx])
                else:
                    candidates &= t_to_blocks[s_idx]
            if not candidates:
                continue
            candidates.discard(b_out)
            if not candidates:
                continue
            
            for b_in in list(candidates)[:5]:
                for s_idx in block_t_indices[b_out]:
                    cover_count[s_idx] -= 1
                for s_idx in block_t_indices[b_in]:
                    cover_count[s_idx] += 1
                selected[i] = b_in
                if try_drop_blocks():
                    improved = True
                    break
                else:
                    for s_idx in block_t_indices[b_in]:
                        cover_count[s_idx] -= 1
                    for s_idx in block_t_indices[b_out]:
                        cover_count[s_idx] += 1
                    selected[i] = b_out
            if improved:
                break
        return improved

    def local_search():
        nonlocal best_size
        improved = True
        while improved and elapsed() < deadline - 5:
            improved = False
            if try_drop_blocks():
                improved = True
                continue
            if try_swap():
                improved = True
                continue
        if len(selected) < best_size:
            best_size = len(selected)

    local_search()

    # Iterated Local Search
    while elapsed() < deadline - 10:
        if not selected:
            break
        idx_out = random.randint(0, len(selected)-1)
        b_out = selected[idx_out]
        
        for s_idx in block_t_indices[b_out]:
            cover_count[s_idx] -= 1
        selected.pop(idx_out)
        selected_mask[b_out] = False
        
        # Repair: Randomized Greedy
        # Find uncovered
        uncovered_now = [i for i in range(num_t) if cover_count[i] == 0]
        
        while uncovered_now and elapsed() < deadline - 5:
            # Pick random uncovered subset
            target_sub = random.choice(uncovered_now)
            # Pick random block covering it
            candidate_blocks = list(t_to_blocks[target_sub])
            if not candidate_blocks:
                break
            b_in = random.choice(candidate_blocks)
            
            # Add b_in
            selected.append(b_in)
            selected_mask[b_in] = True
            for s_idx in block_t_indices[b_in]:
                if cover_count[s_idx] == 0:
                    uncovered_now.remove(s_idx)
                cover_count[s_idx] += 1
                
        local_search()

    result = [list(k_blocks[i]) for i in selected]
    return result
