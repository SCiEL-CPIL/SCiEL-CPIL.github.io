import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    k_subs = list(itertools.combinations(range(v), k))
    block_to_mask = {}
    for b in k_subs:
        mask = 0
        for ts in itertools.combinations(b, t):
            idx = t_sub_to_idx[ts]
            mask |= (1 << idx)
        block_to_mask[b] = mask
        
    all_blocks = list(k_subs)
    t_sub_to_blocks = {ts: [] for ts in t_subs}
    for b in all_blocks:
        for ts in itertools.combinations(b, t):
            t_sub_to_blocks[ts].append(b)
            
    uncovered_set = set(t_subs)
    blocks = []
    
    while uncovered_set:
        if time.time() > deadline:
            break
        ts = next(iter(uncovered_set))
        candidates = t_sub_to_blocks[ts]
        if not candidates:
            break
        b = random.choice(candidates)
        blocks.append(b)
        
        for ts_covered in itertools.combinations(b, t):
            if ts_covered in uncovered_set:
                uncovered_set.remove(ts_covered)
                
    blocks_set = set(blocks)
    
    def is_redundant(block, others):
        union_others = 0
        for b in others:
            union_others |= block_to_mask[b]
        return (block_to_mask[block] & ~union_others) == 0

    def cleanup(blocks_list):
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks_list)):
                b = blocks_list[i]
                others = blocks_list[:i] + blocks_list[i+1:]
                if is_redundant(b, others):
                    blocks_list.pop(i)
                    changed = True
                    break
        return blocks_list

    blocks = cleanup(blocks)
    
    def get_unique_masks(blocks_list):
        unique = {}
        for b in blocks_list:
            others = [x for x in blocks_list if x != b]
            union_others = 0
            for b2 in others:
                union_others |= block_to_mask[b2]
            unique[b] = block_to_mask[b] & ~union_others
        return unique

    while time.time() < deadline - 5:
        old_len = len(blocks)
        blocks = cleanup(blocks)
        if len(blocks) < old_len:
            continue
            
        unique_masks = get_unique_masks(blocks)
        
        improved = False
        blocks_list_copy = blocks[:]
        random.shuffle(blocks_list_copy)
        
        for b_old in blocks_list_copy:
            if time.time() > deadline - 5:
                break
            u_old = unique_masks[b_old]
            
            if u_old == 0:
                blocks.remove(b_old)
                improved = True
                break
            
            idx = (u_old & -u_old).bit_length() - 1
            ts_target = t_subs[idx]
            
            candidates = t_sub_to_blocks[ts_target]
            if len(candidates) > 0:
                num_tries = 0
                max_tries = min(50, len(candidates))
                while num_tries < max_tries:
                    b_new = random.choice(candidates)
                    num_tries += 1
                    if b_new == b_old or b_new in blocks:
                        continue
                    if (block_to_mask[b_new] & u_old) == u_old:
                        blocks.remove(b_old)
                        blocks.append(b_new)
                        improved = True
                        break
                        
        if not improved:
            break
            
    return [list(b) for b in blocks]
