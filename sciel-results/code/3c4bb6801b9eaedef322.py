import itertools
import random
import time
import heapq

def _precompute(v, k, t):
    sub_to_idx = {}
    idx_to_sub = []
    for sub in itertools.combinations(range(v), t):
        sub_to_idx[sub] = len(idx_to_sub)
        idx_to_sub.append(sub)
        
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    block_to_subs = []
    sub_to_blocks = [[] for _ in range(len(idx_to_sub))]
    
    for b_idx, b in enumerate(all_blocks):
        subs = []
        for s in itertools.combinations(b, t):
            s_idx = sub_to_idx[s]
            subs.append(s_idx)
            sub_to_blocks[s_idx].append(b_idx)
        block_to_subs.append(subs)
        
    return sub_to_idx, idx_to_sub, all_blocks, block_to_subs, sub_to_blocks

def _greedy_init(all_blocks, block_to_subs, sub_to_blocks, n_subsets):
    n_blocks = len(all_blocks)
    # score[b] = number of uncovered t-subsets in block b
    scores = [len(block_to_subs[b]) for b in range(n_blocks)]
    
    # We need to know which t-subsets are uncovered.
    # We can just use scores. If score[b] > 0, it covers some uncovered.
    # But we need to know if ALL are covered.
    # Total uncovered count.
    total_uncovered = n_subsets
    
    blocks = []
    
    # Max heap of (-score, block_idx)
    heap = [(-scores[b], b) for b in range(n_blocks)]
    heapq.heapify(heap)
    
    # We need to handle lazy deletion or updates.
    # Since scores only decrease, we can just pop and check if it's still valid.
    # But scores change for many blocks.
    # Simpler: Just scan for max score? 75k scan is ~1ms.
    # 300 steps * 1ms = 0.3s. This is fine.
    # Let's stick to scanning for simplicity and robustness against heap staleness.
    
    is_covered = [False] * n_subsets
    
    while total_uncovered > 0:
        best_b = -1
        max_gain = 0
        for b in range(n_blocks):
            if scores[b] > max_gain:
                max_gain = scores[b]
                best_b = b
                
        if best_b == -1 or max_gain == 0:
            break # Should not happen
            
        blocks.append(best_b)
        
        # Update scores and is_covered
        for s in block_to_subs[best_b]:
            if not is_covered[s]:
                is_covered[s] = True
                total_uncovered -= 1
                # Decrement score of all blocks that cover s
                for b_prime in sub_to_blocks[s]:
                    scores[b_prime] -= 1
                    
    return blocks

def _polish(blocks, block_to_subs, n_subsets):
    cover_count = [0] * n_subsets
    for b in blocks:
        for s in block_to_subs[b]:
            cover_count[s] += 1
            
    changed = True
    blocks = list(blocks)
    
    while changed:
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            b = blocks[i]
            redundant = True
            for s in block_to_subs[b]:
                if cover_count[s] == 1:
                    redundant = False
                    break
            if redundant:
                for s in block_to_subs[b]:
                    cover_count[s] -= 1
                blocks.pop(i)
                changed = True
                break
                
    return blocks

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140
    
    sub_to_idx, idx_to_sub, all_blocks, block_to_subs, sub_to_blocks = _precompute(v, k, t)
    n_subsets = len(idx_to_sub)
    
    # Initial Greedy
    blocks = _greedy_init(all_blocks, block_to_subs, sub_to_blocks, n_subsets)
    blocks = _polish(blocks, block_to_subs, n_subsets)
    
    best_blocks = blocks
    best_len = len(blocks)
    
    rng = random.Random(42)
    
    while time.time() < deadline - 5:
        current = list(best_blocks)
        
        if not current:
            break
            
        # Perturb: Swap one block
        idx_to_remove = rng.randint(0, len(current) - 1)
        b_old = current[idx_to_remove]
        
        b_new = rng.randint(0, len(all_blocks) - 1)
        # Avoid same block
        attempts = 0
        while b_new == b_old and attempts < 10:
            b_new = rng.randint(0, len(all_blocks) - 1)
            attempts += 1
            
        current[idx_to_remove] = b_new
        
        # Repair
        # Calculate coverage
        cover_count = [0] * n_subsets
        for b in current:
            for s in block_to_subs[b]:
                cover_count[s] += 1
                
        # Find uncovered
        uncovered = [s for s in range(n_subsets) if cover_count[s] == 0]
        
        if uncovered:
            # Greedy repair
            # We want to cover 'uncovered'.
            # We can use the same score logic, but only for uncovered sets.
            # Or just simple greedy: pick block covering most uncovered.
            
            # Optimization: Precompute score for repair?
            # Just scan.
            
            while uncovered and time.time() < deadline - 5:
                best_b = -1
                max_gain = 0
                for b in range(len(all_blocks)):
                    gain = 0
                    for s in block_to_subs[b]:
                        if cover_count[s] == 0:
                            gain += 1
                    if gain > max_gain:
                        max_gain = gain
                        best_b = b
                        
                if best_b == -1 or max_gain == 0:
                    break
                    
                current.append(best_b)
                for s in block_to_subs[best_b]:
                    cover_count[s] += 1
                    if cover_count[s] == 1:
                         # It was 0, now 1.
                         # We need to remove it from 'uncovered'
                         # 'uncovered' is a list. Removing is O(N).
                         # Let's just recompute uncovered after each add?
                         # Or check condition: if not any(cover_count[s]==0 for s in uncovered)
                         pass
                
                # Check if done
                # Efficient check:
                still_uncovered = False
                for s in uncovered:
                    if cover_count[s] == 0:
                        still_uncovered = True
                        break
                if not still_uncovered:
                    break
            
        # Polish
        current = _polish(current, block_to_subs, n_subsets)
        
        if len(current) < best_len:
            best_len = len(current)
            best_blocks = current
            
    return [list(all_blocks[b]) for b in best_blocks]
