"""Covering design C(24,7,3) via greedy initialization and Simulated Annealing local search on block selection."""
import itertools
import time
import random
import math

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    # Precompute all t-subsets and k-subsets
    all_tsets = list(itertools.combinations(range(v), t))
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    num_tsets = len(all_tsets)
    
    all_ksets = list(itertools.combinations(range(v), k))
    num_ksets = len(all_ksets)
    
    # For each k-set, store the indices of the t-sets it covers
    kset_tsets = []
    for kset in all_ksets:
        tsets_in_kset = [tset_to_idx[frozenset(c)] for c in itertools.combinations(kset, t)]
        kset_tsets.append(tsets_in_kset)
        
    # Precompute tset_to_ksets_idx for greedy repair
    tset_to_ksets_idx = [[] for _ in range(num_tsets)]
    for k_idx, t_idxs in enumerate(kset_tsets):
        for t_idx in t_idxs:
            tset_to_ksets_idx[t_idx].append(k_idx)

    # Phase 1: Greedy Construction
    uncovered = set(range(num_tsets))
    selected_blocks = set()
    cov_counts = [0] * num_tsets
    
    def add_block(idx):
        if idx in selected_blocks:
            return
        selected_blocks.add(idx)
        for t_idx in kset_tsets[idx]:
            cov_counts[t_idx] += 1
            if cov_counts[t_idx] == 1:
                uncovered.discard(t_idx)

    def remove_block(idx):
        if idx not in selected_blocks:
            return
        selected_blocks.remove(idx)
        for t_idx in kset_tsets[idx]:
            cov_counts[t_idx] -= 1
            if cov_counts[t_idx] == 0:
                uncovered.add(t_idx)

    while uncovered and time.time() < deadline - 60:
        seed_t_idx = random.choice(list(uncovered))
        candidates = tset_to_ksets_idx[seed_t_idx]
        
        if not candidates:
            break
            
        best_k_idx = -1
        max_gain = -1
        
        # Sample candidates to speed up
        if len(candidates) > 50:
            sample = random.sample(candidates, 50)
        else:
            sample = candidates
            
        for k_idx in sample:
            gain = 0
            for t_idx in kset_tsets[k_idx]:
                if t_idx in uncovered:
                    gain += 1
            if gain > max_gain:
                max_gain = gain
                best_k_idx = k_idx
        
        if best_k_idx != -1:
            add_block(best_k_idx)

    # Ensure validity
    if uncovered:
        for t_idx in list(uncovered):
            candidates = tset_to_ksets_idx[t_idx]
            if candidates:
                k_idx = random.choice(candidates)
                add_block(k_idx)

    # Phase 2: Simulated Annealing
    current_blocks = set(selected_blocks)
    current_uncovered = len(uncovered)
    penalty = 1000.0
    current_cost = len(current_blocks) + penalty * current_uncovered
    
    best_blocks = set(current_blocks)
    best_uncovered = current_uncovered
    best_cost = current_cost
    
    T_start = 5.0
    T_end = 0.1
    start_time = time.time()
    
    all_block_indices = list(range(num_ksets))
    
    while time.time() < deadline - 5:
        elapsed = time.time() - start_time
        # Exponential cooling
        T = T_start * math.exp(-elapsed / 20.0)
        if T < T_end:
            T = T_end
            
        # Random Move
        if random.random() < 0.5:
            # Remove
            if current_blocks:
                k_idx = random.choice(list(current_blocks))
                delta_unc = 0
                for t_idx in kset_tsets[k_idx]:
                    if cov_counts[t_idx] == 1:
                        delta_unc += 1
                delta_cost = -1.0 + penalty * delta_unc
                
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    current_blocks.remove(k_idx)
                    for t_idx in kset_tsets[k_idx]:
                        cov_counts[t_idx] -= 1
                        if cov_counts[t_idx] == 0:
                            current_uncovered += 1
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = set(current_blocks)
                        best_uncovered = current_uncovered
        else:
            # Add
            # Bias towards blocks that cover uncovered triples
            if current_uncovered > 0 and random.random() < 0.8:
                # Pick a random uncovered triple
                # We need a list of uncovered. 
                # Maintaining a list is hard. 
                # Just pick a random block and check.
                pass
                
            k_idx = random.choice(all_block_indices)
            if k_idx not in current_blocks:
                delta_unc = 0
                for t_idx in kset_tsets[k_idx]:
                    if cov_counts[t_idx] == 0:
                        delta_unc += 1
                delta_cost = 1.0 - penalty * delta_unc
                
                if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
                    current_blocks.add(k_idx)
                    for t_idx in kset_tsets[k_idx]:
                        cov_counts[t_idx] += 1
                        if cov_counts[t_idx] == 1:
                            current_uncovered -= 1
                    current_cost += delta_cost
                    
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_blocks = set(current_blocks)
                        best_uncovered = current_uncovered

    # Convert best_blocks to list of lists
    result_blocks = [list(all_ksets[i]) for i in best_blocks]
    
    # Final validity check and repair
    final_uncovered = set(range(num_tsets))
    for b in result_blocks:
        for c in itertools.combinations(b, t):
            final_uncovered.discard(frozenset(c))
            
    if final_uncovered:
        for t_idx in final_uncovered:
            candidates = tset_to_ksets_idx[t_idx]
            if candidates:
                # Add a block
                chosen = random.choice(candidates)
                result_blocks.append(list(all_ksets[chosen]))
                
    return result_blocks
