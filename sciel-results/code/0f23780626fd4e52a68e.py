import itertools
import random
import numpy as np
import time
import math

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using Simulated Annealing 
    with Greedy Initialization.
    """
    start_time = time.time()
    time_limit = 115.0

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Precompute block -> t-subset indices mapping
    block_t_indices = []
    for block in all_k_subsets:
        indices = []
        for ts in itertools.combinations(block, t):
            indices.append(t_to_idx[ts])
        block_t_indices.append(indices)
        
    # Precompute t-subset -> block indices mapping (for greedy init)
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for t_idx in indices:
            t_to_blocks[t_idx].append(b_idx)

    best_blocks = None
    best_cost = float('inf') # Cost = num_blocks + lambda * uncovered
    
    # Greedy Initialization
    def greedy_init():
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        uncovered = set(range(num_t_subsets))
        current_blocks = []
        
        while uncovered:
            # Pick random uncovered t-subset to break symmetry
            t_idx = random.choice(tuple(uncovered))
            candidates = t_to_blocks[t_idx]
            
            best_b_idx = -1
            best_gain = -1
            
            for b_idx in candidates:
                gain = 0
                for t2_idx in block_t_indices[b_idx]:
                    if t2_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
            
            current_blocks.append(best_b_idx)
            for t2_idx in block_t_indices[best_b_idx]:
                coverage[t2_idx] += 1
                if coverage[t2_idx] == 1:
                    uncovered.discard(t2_idx)
                    
        # Remove redundancies
        current_blocks, coverage = remove_redundant(current_blocks, coverage)
        return current_blocks, coverage

    def remove_redundant(blocks, coverage):
        changed = True
        while changed:
            changed = False
            for i in range(len(blocks)):
                b_idx = blocks[i]
                is_redundant = True
                for t_idx in block_t_indices[b_idx]:
                    if coverage[t_idx] <= 1:
                        is_redundant = False
                        break
                if is_redundant:
                    blocks.pop(i)
                    for t_idx in block_t_indices[b_idx]:
                        coverage[t_idx] -= 1
                    changed = True
                    break
        return blocks, coverage

    # Simulated Annealing
    for seed in range(20):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(seed)
        current_blocks, coverage = greedy_init()
        current_size = len(current_blocks)
        current_uncovered = np.sum(coverage == 0)
        
        # Cost function: size + penalty * uncovered
        # We want to minimize size. Uncovered should be 0.
        # If we allow uncovered, we can explore larger/smaller sets.
        # Let's define cost = size + 100 * uncovered.
        # We accept moves that reduce cost, or increase cost with prob exp(-delta/T).
        
        T = 10.0
        T_min = 0.1
        cooling = 0.99
        
        for step in range(5000):
            if time.time() - start_time > time_limit:
                break
            
            # Pick a move
            move_type = random.random()
            
            if move_type < 0.7: # Swap
                if not current_blocks:
                    continue
                i = random.randint(0, len(current_blocks) - 1)
                old_b_idx = current_blocks[i]
                new_b_idx = random.randint(0, num_k_subsets - 1)
                if old_b_idx == new_b_idx:
                    continue
                
                # Calculate delta cost
                # Delta size = 0
                # Delta uncovered = (new_uncovered - old_uncovered)
                
                # Apply swap temporarily to check uncovered
                # We need to know which t-subsets become uncovered or stay covered.
                # t in old \ new: coverage drops by 1. If it was 1, it becomes 0 (uncovered +1).
                # t in new \ old: coverage rises by 1. If it was 0, it becomes 1 (uncovered -1).
                
                old_t_set = set(block_t_indices[old_b_idx])
                new_t_set = set(block_t_indices[new_b_idx])
                
                delta_uncovered = 0
                # Check t in old \ new
                for t_idx in old_t_set:
                    if t_idx not in new_t_set:
                        if coverage[t_idx] == 1:
                            delta_uncovered += 1
                # Check t in new \ old
                for t_idx in new_t_set:
                    if t_idx not in old_t_set:
                        if coverage[t_idx] == 0:
                            delta_uncovered -= 1
                
                delta_cost = delta_uncovered * 100.0
                
                # Accept?
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    # Perform swap
                    for t_idx in old_t_set:
                        coverage[t_idx] -= 1
                    for t_idx in new_t_set:
                        coverage[t_idx] += 1
                    current_blocks[i] = new_b_idx
                    current_uncovered += delta_uncovered
                    
                    # Try to remove redundant blocks
                    old_size = len(current_blocks)
                    current_blocks, coverage = remove_redundant(current_blocks, coverage)
                    new_size = len(current_blocks)
                    
                    # Recalculate uncovered after removal
                    current_uncovered = np.sum(coverage == 0)
                    
                    # Update best
                    if current_uncovered == 0 and len(current_blocks) < best_cost:
                        best_cost = len(current_blocks)
                        best_blocks = current_blocks[:]
                        
            elif move_type < 0.85: # Remove
                if not current_blocks:
                    continue
                i = random.randint(0, len(current_blocks) - 1)
                b_idx = current_blocks[i]
                
                delta_uncovered = 0
                for t_idx in block_t_indices[b_idx]:
                    if coverage[t_idx] == 1:
                        delta_uncovered += 1
                        
                delta_cost = -1 + delta_uncovered * 100.0 # Size -1, Uncovered +delta
                
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    # Perform removal
                    for t_idx in block_t_indices[b_idx]:
                        coverage[t_idx] -= 1
                    current_blocks.pop(i)
                    current_uncovered += delta_uncovered
                    
                    # Update best
                    if current_uncovered == 0 and len(current_blocks) < best_cost:
                        best_cost = len(current_blocks)
                        best_blocks = current_blocks[:]
                        
            else: # Add (less likely, to escape local minima)
                new_b_idx = random.randint(0, num_k_subsets - 1)
                if new_b_idx in current_blocks:
                    continue
                
                delta_uncovered = 0
                for t_idx in block_t_indices[new_b_idx]:
                    if coverage[t_idx] == 0:
                        delta_uncovered -= 1
                        
                delta_cost = 1 + delta_uncovered * 100.0
                
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    # Perform add
                    for t_idx in block_t_indices[new_b_idx]:
                        coverage[t_idx] += 1
                    current_blocks.append(new_b_idx)
                    current_uncovered += delta_uncovered
                    
                    # Try to remove redundant blocks
                    old_size = len(current_blocks)
                    current_blocks, coverage = remove_redundant(current_blocks, coverage)
                    new_size = len(current_blocks)
                    current_uncovered = np.sum(coverage == 0)
                    
                    # Update best
                    if current_uncovered == 0 and len(current_blocks) < best_cost:
                        best_cost = len(current_blocks)
                        best_blocks = current_blocks[:]
                        
            T *= cooling
            if T < T_min:
                T = T_min

    if best_blocks is None:
        # Fallback to greedy
        current_blocks, coverage = greedy_init()
        best_blocks = current_blocks

    final_blocks = [list(all_k_subsets[idx]) for idx in best_blocks]
    return final_blocks
