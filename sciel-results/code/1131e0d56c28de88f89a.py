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
        
    # Precompute target_to_blocks
    target_to_blocks = [[] for _ in range(M)]
    for i, subs in enumerate(block_subsets):
        for j in subs:
            target_to_blocks[j].append(i)
            
    random.seed(42)
    
    # State variables
    cov = [0] * M
    x = [0] * N  # 1 if selected
    selected = set()
    
    # Helper: Add block i
    def add_block(i):
        if i in selected:
            return
        x[i] = 1
        selected.add(i)
        for s in block_subsets[i]:
            cov[s] += 1
            
    # Helper: Remove block i
    def remove_block(i):
        if i not in selected:
            return
        x[i] = 0
        selected.remove(i)
        for s in block_subsets[i]:
            cov[s] -= 1
            
    # Helper: Get number of uncovered targets
    def get_uncovered_count():
        # This is O(M). M is ~11k for C(19,8,5). 
        # To speed up, we can maintain a counter, but for simplicity in SA steps, 
        # we might calculate delta or just check.
        # Let's maintain a global uncovered_count variable for O(1) cost evaluation.
        pass

    # We will maintain `uncovered_count` explicitly.
    uncovered_count = M
    
    def add_block_optimized(i):
        nonlocal uncovered_count
        if i in selected:
            return
        x[i] = 1
        selected.add(i)
        for s in block_subsets[i]:
            if cov[s] == 0:
                uncovered_count -= 1
            cov[s] += 1
            
    def remove_block_optimized(i):
        nonlocal uncovered_count
        if i not in selected:
            return
        x[i] = 0
        selected.remove(i)
        for s in block_subsets[i]:
            cov[s] -= 1
            if cov[s] == 0:
                uncovered_count += 1

    # 1. Initialization: Fast Greedy to get a valid starting solution
    # This ensures we start from a valid state (uncovered_count = 0)
    
    # Helper: Greedy Step for initialization
    def greedy_step_init(uncovered_list):
        if not uncovered_list:
            return False
        t_idx = random.choice(uncovered_list)
        candidates = target_to_blocks[t_idx]
        
        best_block = -1
        max_new = -1
        
        for b in candidates:
            if b in selected:
                continue
            new_cov_count = 0
            for s in block_subsets[b]:
                if cov[s] == 0:
                    new_cov_count += 1
            if new_cov_count > max_new:
                max_new = new_cov_count
                best_block = b
                
        if best_block != -1:
            add_block_optimized(best_block)
            return True
        return False

    uncovered = list(range(M))
    while uncovered:
        if not greedy_step_init(uncovered):
            break
        uncovered = [i for i in uncovered if cov[i] == 0]
        
    # Remove redundant blocks to start SA with a better solution
    def remove_redundant():
        nonlocal uncovered_count
        removed = True
        while removed:
            removed = False
            for i in list(selected):
                redundant = True
                for s in block_subsets[i]:
                    if cov[s] <= 1:
                        redundant = False
                        break
                if redundant:
                    remove_block_optimized(i)
                    removed = True

    remove_redundant()
    
    current_best_size = len(selected)
    best_selected = set(selected)
    
    start_time = time.time()
    time_limit = 115.0
    
    # 2. Simulated Annealing
    # Cost function: size + penalty * uncovered
    # We want to minimize size. If we remove a block and uncover sets, 
    # we accept if (new_size + P*new_uncov) < (old_size + P*old_uncov) + noise
    
    penalty = 1000 # Large penalty to discourage invalid states, but allow exploration
    
    T = 50.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    while time.time() - start_time < time_limit and T > T_min:
        T *= cooling_rate
        
        # Current cost
        cur_cost = len(selected) + penalty * uncovered_count
        
        # Random Move: 70% Remove, 30% Add
        if selected and random.random() < 0.7:
            # Remove a random block
            victim = random.choice(list(selected))
            # Calculate delta cost
            # new_size = len(selected) - 1
            # new_uncov: count how many subsets in victim become uncovered
            new_uncov_delta = 0
            for s in block_subsets[victim]:
                if cov[s] == 1:
                    new_uncov_delta += 1
            
            delta_cost = -1 + penalty * new_uncov_delta
            new_cost = cur_cost + delta_cost
            
            # Accept if better or with probability exp(-delta/T)
            if new_cost < cur_cost or random.random() < math.exp(-delta_cost / T):
                remove_block_optimized(victim)
                cur_cost = new_cost
                
                if uncovered_count == 0 and len(selected) < current_best_size:
                    current_best_size = len(selected)
                    best_selected = set(selected)
                    
        else:
            # Add a random block
            # Pick a random block not in selected
            # To be efficient, we can pick a random index and check
            b = random.randint(0, N - 1)
            if b in selected:
                continue
            
            # Calculate delta cost
            # new_size = len(selected) + 1
            # new_uncov_delta: count how many subsets in b are currently uncovered
            new_uncov_delta = 0
            for s in block_subsets[b]:
                if cov[s] == 0:
                    new_uncov_delta += 1
            
            delta_cost = 1 - penalty * new_uncov_delta
            new_cost = cur_cost + delta_cost
            
            if new_cost < cur_cost or random.random() < math.exp(-delta_cost / T):
                add_block_optimized(b)
                cur_cost = new_cost
                
        # Occasionally repair if we are far from valid? 
        # Or just let SA handle it.
        # If uncovered_count is high, the penalty makes cost high, 
        # so Add moves become attractive.
        
    # Final check: if best_selected is empty or invalid, fallback to last valid state
    # best_selected is only updated when uncovered_count == 0, so it is valid.
    
    if not best_selected:
         # Fallback to current state if it's valid, else re-greedy
         if uncovered_count == 0:
             best_selected = set(selected)
         else:
             # Re-run greedy on current state to fix
             uncovered = [i for i in range(M) if cov[i] == 0]
             while uncovered:
                 if not greedy_step_init(uncovered):
                     break
                 uncovered = [i for i in uncovered if cov[i] == 0]
             best_selected = set(selected)

    blocks = []
    for i in best_selected:
        blocks.append(list(all_blocks[i]))
        
    # Ensure distinct and sorted
    unique_blocks = []
    seen = set()
    for b in blocks:
        b.sort()
        t_b = tuple(b)
        if t_b not in seen:
            seen.add(t_b)
            unique_blocks.append(b)
            
    return unique_blocks
