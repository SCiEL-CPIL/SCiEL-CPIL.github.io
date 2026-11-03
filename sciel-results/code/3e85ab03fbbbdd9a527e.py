import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115
    
    def time_left():
        return deadline - time.time()
    
    # Precompute
    all_t_subs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_t_subs)
    tsub_index = {s: i for i, s in enumerate(all_t_subs)}
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, get its t-subset indices as a set for fast lookup
    block_tsubs_set = []
    block_tsubs_list = []
    for b in all_blocks:
        subs = list(itertools.combinations(b, t))
        indices = [tsub_index[s] for s in subs]
        block_tsubs_set.append(set(indices))
        block_tsubs_list.append(indices)
    
    # Reverse mapping: t-subset -> list of blocks containing it
    tsub_to_blocks = [[] for _ in range(n_tsubs)]
    for i, subs in enumerate(block_tsubs_list):
        for s in subs:
            tsub_to_blocks[s].append(i)
    
    best_blocks = None
    best_count = float('inf')
    
    while time_left() > 5:
        # Greedy with randomization
        uncovered = set(range(n_tsubs))
        selected = []
        
        while uncovered and time_left() > 2:
            # Pick random uncovered t-subset
            seed_tsub = random.choice(list(uncovered))
            candidates = tsub_to_blocks[seed_tsub]
            
            # Find best candidate by gain
            best_gain = -1
            best_cands = []
            for idx in candidates:
                # Gain = number of uncovered t-subsets in this block
                gain = len(block_tsubs_set[idx] & uncovered)
                if gain > best_gain:
                    best_gain = gain
                    best_cands = [idx]
                elif gain == best_gain:
                    best_cands.append(idx)
            
            if not best_cands:
                break
            
            chosen = random.choice(best_cands)
            selected.append(chosen)
            uncovered -= block_tsubs_set[chosen]
        
        if not selected:
            continue
        
        # Reduction: remove redundant blocks
        changed = True
        while changed and time_left() > 2:
            changed = False
            new_selected = []
            for i, b_idx in enumerate(selected):
                # Check if all t-subsets of this block are covered by others
                redundant = True
                for s in block_tsubs_list[b_idx]:
                    covered = False
                    for j, other_idx in enumerate(selected):
                        if j == i:
                            continue
                        if s in block_tsubs_set[other_idx]:
                            covered = True
                            break
                    if not covered:
                        redundant = False
                        break
                if redundant:
                    changed = True
                else:
                    new_selected.append(b_idx)
            if changed:
                selected = new_selected
        
        if len(selected) < best_count:
            best_count = len(selected)
            best_blocks = [list(all_blocks[idx]) for idx in selected]
            if best_count <= 265:
                break
    
    if best_blocks is None:
        return []
    
    return best_blocks
