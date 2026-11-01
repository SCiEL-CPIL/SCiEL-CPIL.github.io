import itertools
import random
import time
import math

def construct(v, k, t):
    points = range(v)
    all_blocks = list(itertools.combinations(points, k))
    N = len(all_blocks)
    
    all_targets = list(itertools.combinations(points, t))
    M = len(all_targets)
    
    target_index = {target: i for i, target in enumerate(all_targets)}
    
    # Precompute subsets for each block
    block_subsets = []
    for block in all_blocks:
        subs = []
        for sub in itertools.combinations(block, t):
            subs.append(target_index[sub])
        block_subsets.append(subs)
        
    random.seed(42)
    
    # State variables
    x = [0] * N
    cov = [0] * M
    selected_list = [] # List of indices of selected blocks
    
    def add_block(i):
        if x[i]:
            return 0, 0
        x[i] = 1
        selected_list.append(i)
        delta_blocks = 1
        delta_uncovered = 0
        for s in block_subsets[i]:
            if cov[s] == 0:
                delta_uncovered -= 1
            cov[s] += 1
        return delta_blocks, delta_uncovered

    def remove_block(i):
        if not x[i]:
            return 0, 0
        x[i] = 0
        # Remove from selected_list
        try:
            idx = selected_list.index(i)
            selected_list[idx] = selected_list[-1]
            selected_list.pop()
        except ValueError:
            pass
        delta_blocks = -1
        delta_uncovered = 0
        for s in block_subsets[i]:
            cov[s] -= 1
            if cov[s] == 0:
                delta_uncovered += 1
        return delta_blocks, delta_uncovered

    # 1. Initialization: Fast Greedy
    # Reuse the efficient greedy from the seed to get a valid start
    target_to_blocks = [[] for _ in range(M)]
    for i, subs in enumerate(block_subsets):
        for s in subs:
            target_to_blocks[s].append(i)
            
    uncovered = list(range(M))
    while uncovered:
        if not uncovered: break
        t_idx = random.choice(uncovered)
        candidates = target_to_blocks[t_idx]
        
        best_block = -1
        max_new = -1
        
        for b in candidates:
            if x[b]:
                continue
            new_cov_count = 0
            for s in block_subsets[b]:
                if cov[s] == 0:
                    new_cov_count += 1
            if new_cov_count > max_new:
                max_new = new_cov_count
                best_block = b
                
        if best_block != -1:
            add_block(best_block)
        uncovered = [j for j in uncovered if cov[j] == 0]
        
    # Remove redundant
    changed = True
    while changed:
        changed = False
        for i in selected_list[:]:
            redundant = True
            for s in block_subsets[i]:
                if cov[s] <= 1:
                    redundant = False
                    break
            if redundant:
                remove_block(i)
                changed = True
                
    current_num_blocks = len(selected_list)
    current_uncovered = sum(1 for c in cov if c == 0)
    assert current_uncovered == 0, "Greedy failed to cover"
    
    best_num_blocks = current_num_blocks
    best_x = x[:]
    best_selected = selected_list[:]
    
    # 2. Simulated Annealing
    start_time = time.time()
    time_limit = 115.0
    
    T = 5.0
    alpha = 0.99998
    LAMBDA = 10000.0
    
    num_blocks = current_num_blocks
    num_uncovered = 0
    
    while time.time() - start_time < time_limit:
        # Move probability: 50/50 Add/Remove
        if random.random() < 0.5:
            # Add
            i = random.randint(0, N - 1)
            if x[i]:
                continue
            # Tentative Add
            x[i] = 1
            selected_list.append(i)
            temp_db = 1
            temp_du = 0
            for s in block_subsets[i]:
                if cov[s] == 0:
                    temp_du -= 1
                cov[s] += 1
            
            delta_E = temp_db + LAMBDA * temp_du
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                num_blocks += temp_db
                num_uncovered += temp_du
                if num_uncovered == 0 and num_blocks < best_num_blocks:
                    best_num_blocks = num_blocks
                    best_x = x[:]
                    best_selected = selected_list[:]
            else:
                # Reject: Revert
                x[i] = 0
                selected_list.pop()
                for s in block_subsets[i]:
                    cov[s] -= 1
        else:
            # Remove
            if not selected_list:
                continue
            i = random.choice(selected_list)
            
            # Tentative Remove
            x[i] = 0
            idx = selected_list.index(i)
            selected_list[idx] = selected_list[-1]
            selected_list.pop()
            
            temp_db = -1
            temp_du = 0
            for s in block_subsets[i]:
                cov[s] -= 1
                if cov[s] == 0:
                    temp_du += 1
            
            delta_E = temp_db + LAMBDA * temp_du
            
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                # Accept
                num_blocks += temp_db
                num_uncovered += temp_du
                if num_uncovered == 0 and num_blocks < best_num_blocks:
                    best_num_blocks = num_blocks
                    best_x = x[:]
                    best_selected = selected_list[:]
            else:
                # Reject: Revert
                x[i] = 1
                selected_list.append(i)
                for s in block_subsets[i]:
                    cov[s] += 1
                    
        T *= alpha
        if T < 0.01:
            T = 0.01

    # Final output
    blocks = []
    for i in best_selected:
        blocks.append(list(all_blocks[i]))
        
    unique_blocks = []
    seen = set()
    for b in blocks:
        b.sort()
        t_b = tuple(b)
        if t_b not in seen:
            seen.add(t_b)
            unique_blocks.append(b)
            
    return unique_blocks
