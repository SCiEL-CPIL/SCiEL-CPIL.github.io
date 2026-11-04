import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0

    # Precomputation
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_idx = {tsub: i for i, tsub in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    all_ksubs = list(itertools.combinations(range(v), k))
    num_ksubs = len(all_ksubs)
    
    covers = [set() for _ in range(num_ksubs)]
    for b_idx, block in enumerate(all_ksubs):
        for tsub in itertools.combinations(block, t):
            covers[b_idx].add(tsub_to_idx[tsub])
            
    blocks_containing = [[] for _ in range(num_tsubs)]
    for b_idx in range(num_ksubs):
        for tsub_idx in covers[b_idx]:
            blocks_containing[tsub_idx].append(b_idx)

    def greedy_construction(rng, time_limit):
        uncovered = set(range(num_tsubs))
        blocks = []
        
        while uncovered:
            if time.time() > time_limit:
                break
            
            seed_tsub = rng.choice(list(uncovered))
            candidates = blocks_containing[seed_tsub]
            
            best_block = -1
            best_gain = 0
            
            for b_idx in candidates:
                if b_idx in blocks: # Avoid duplicates
                    continue
                gain = 0
                for t_idx in covers[b_idx]:
                    if t_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
                    
            if best_block == -1:
                break
                
            blocks.append(best_block)
            for t_idx in covers[best_block]:
                uncovered.discard(t_idx)
                
        return blocks

    def is_valid(blocks):
        counts = [0] * num_tsubs
        for b_idx in blocks:
            for t_idx in covers[b_idx]:
                counts[t_idx] += 1
        return all(c > 0 for c in counts)

    def local_search(blocks, rng, time_limit):
        block_set = set(blocks)
        blocks = list(blocks)
        
        counts = [0] * num_tsubs
        for b_idx in blocks:
            for t_idx in covers[b_idx]:
                counts[t_idx] += 1
                
        def is_redundant(b_idx):
            for t_idx in covers[b_idx]:
                if counts[t_idx] <= 1:
                    return False
            return True

        def remove_block(b_idx):
            for t_idx in covers[b_idx]:
                counts[t_idx] -= 1
            blocks.remove(b_idx)
            block_set.discard(b_idx)

        def add_block(b_idx):
            for t_idx in covers[b_idx]:
                counts[t_idx] += 1
            blocks.append(b_idx)
            block_set.add(b_idx)

        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Deletion pass
            rng.shuffle(blocks)
            for b_idx in blocks[:]:
                if is_redundant(b_idx):
                    remove_block(b_idx)
                    improved = True
            
            # 2. Swap/Repair
            if blocks:
                b_idx = rng.choice(blocks)
                for t_idx in covers[b_idx]:
                    counts[t_idx] -= 1
                blocks.remove(b_idx)
                block_set.discard(b_idx)
                
                uncovered = [t_idx for t_idx in covers[b_idx] if counts[t_idx] == 0]
                
                if not uncovered:
                    improved = True
                else:
                    seed_tsub = rng.choice(uncovered)
                    candidates = blocks_containing[seed_tsub]
                    
                    replacement_found = False
                    for b_new in candidates:
                        if b_new in block_set:
                            continue
                        valid = True
                        for t_idx in uncovered:
                            if t_idx not in covers[b_new]:
                                valid = False
                                break
                        if valid:
                            add_block(b_new)
                            replacement_found = True
                            improved = True
                            break
                    
                    if not replacement_found:
                        add_block(b_idx)

            # 3. 1-2 Exchange
            if len(blocks) >= 2:
                b1, b2 = rng.sample(blocks, 2)
                for t_idx in covers[b1]:
                    counts[t_idx] -= 1
                for t_idx in covers[b2]:
                    counts[t_idx] -= 1
                blocks.remove(b1)
                blocks.remove(b2)
                block_set.discard(b1)
                block_set.discard(b2)
                
                uncovered = [t_idx for t_idx in range(num_tsubs) if counts[t_idx] == 0]
                
                if not uncovered:
                    improved = True
                else:
                    seed_tsub = rng.choice(uncovered)
                    candidates = blocks_containing[seed_tsub]
                    
                    replacement_found = False
                    for b_new in candidates:
                        if b_new in block_set:
                            continue
                        valid = True
                        for t_idx in uncovered:
                            if t_idx not in covers[b_new]:
                                valid = False
                                break
                        if valid:
                            add_block(b_new)
                            replacement_found = True
                            improved = True
                            break
                            
                    if not replacement_found:
                        add_block(b1)
                        add_block(b2)

        return blocks

    rng = random.Random(42)
    blocks = greedy_construction(rng, deadline - 10.0)
    
    if not is_valid(blocks):
        uncovered = set(range(num_tsubs))
        for b_idx in blocks:
            for t_idx in covers[b_idx]:
                uncovered.discard(t_idx)
        
        while uncovered and time.time() < deadline - 5.0:
            seed_tsub = rng.choice(list(uncovered))
            candidates = blocks_containing[seed_tsub]
            best_block = -1
            best_gain = 0
            for b_idx in candidates:
                if b_idx in blocks:
                    continue
                gain = 0
                for t_idx in covers[b_idx]:
                    if t_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = b_idx
            if best_block != -1:
                blocks.append(best_block)
                for t_idx in covers[best_block]:
                    uncovered.discard(t_idx)
            else:
                break

    blocks = local_search(blocks, rng, deadline - 1.0)
    
    result = [list(all_ksubs[b_idx]) for b_idx in blocks]
    return result
