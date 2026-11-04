import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140
    
    all_tsubs = list(itertools.combinations(range(v), t))
    n_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    def block_tsub_indices(block):
        return [tsub_to_idx[ts] for ts in itertools.combinations(block, t)]
    
    def greedy_block(cover_count, seed_tsub=None):
        if seed_tsub is None:
            seed_tsub = None
            for i in range(n_tsubs):
                if cover_count[i] == 0:
                    seed_tsub = all_tsubs[i]
                    break
            if seed_tsub is None:
                return None
        block = list(seed_tsub)
        block_set = set(block)
        
        while len(block) < k:
            best_point = None
            best_gain = -1
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for rest in itertools.combinations(block, t-1):
                    new_ts = tuple(sorted(rest + (x,)))
                    idx = tsub_to_idx.get(new_ts)
                    if idx is not None and cover_count[idx] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = x
            if best_point is None:
                break
            block.append(best_point)
            block_set.add(best_point)
        
        block.sort()
        return block
    
    def greedy_construct():
        blocks = []
        cover_count = [0] * n_tsubs
        while True:
            if time.time() > deadline:
                break
            block = greedy_block(cover_count)
            if block is None:
                break
            blocks.append(block)
            for idx in block_tsub_indices(block):
                cover_count[idx] += 1
        return blocks, cover_count
    
    def remove_redundant(blocks, cover_count):
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                redundant = True
                for ts in itertools.combinations(blocks[i], t):
                    if cover_count[tsub_to_idx[ts]] == 1:
                        redundant = False
                        break
                if redundant:
                    for idx in block_tsub_indices(blocks[i]):
                        cover_count[idx] -= 1
                    blocks.pop(i)
                    changed = True
                    break
        return blocks, cover_count
    
    def repair(blocks, cover_count):
        while True:
            if time.time() > deadline:
                break
            uncovered = [i for i in range(n_tsubs) if cover_count[i] == 0]
            if not uncovered:
                break
            seed_tsub = all_tsubs[uncovered[0]]
            block = greedy_block(cover_count, seed_tsub)
            if block is None:
                break
            blocks.append(block)
            for idx in block_tsub_indices(block):
                cover_count[idx] += 1
        return blocks, cover_count
    
    blocks, cover_count = greedy_construct()
    blocks, cover_count = remove_redundant(blocks, cover_count)
    best_len = len(blocks)
    
    no_improve = 0
    while time.time() < deadline and no_improve < 200:
        if not blocks:
            break
        idx = random.randint(0, len(blocks) - 1)
        removed_block = blocks.pop(idx)
        for t_idx in block_tsub_indices(removed_block):
            cover_count[t_idx] -= 1
        
        blocks, cover_count = repair(blocks, cover_count)
        blocks, cover_count = remove_redundant(blocks, cover_count)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            no_improve = 0
        else:
            no_improve += 1
    
    return blocks
