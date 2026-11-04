import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin

    # Precompute all t-subsets
    all_tsubsets = list(itertools.combinations(range(v), t))
    n_tsubsets = len(all_tsubsets)
    
    # Map t-subset tuple to index
    ts_to_idx = {ts: i for i, ts in enumerate(all_tsubsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, find the indices of the t-subsets it covers
    block_ts_indices = []
    for blk in all_blocks:
        indices = []
        for ts in itertools.combinations(blk, t):
            indices.append(ts_to_idx[ts])
        block_ts_indices.append(indices)
        
    # For each t-subset, find the blocks that cover it
    ts_to_blocks = [[] for _ in range(n_tsubsets)]
    for b_idx, ts_list in enumerate(block_ts_indices):
        for ts_idx in ts_list:
            ts_to_blocks[ts_idx].append(b_idx)
            
    # Precompute sets for fast membership in local search
    block_ts_sets = [set(indices) for indices in block_ts_indices]

    def run_greedy_local_search(seed_val):
        random.seed(seed_val)
        
        # Greedy Phase (Pivot based)
        counts = [0] * n_tsubsets
        selected = []
        remaining = set(range(n_tsubsets))
        
        while remaining:
            if time.time() > deadline:
                return None
            
            # Pick a random pivot from remaining
            pivot_ts = random.choice(list(remaining))
            
            best_block = -1
            best_gain = -1
            best_score = -1
            
            # Only consider blocks covering the pivot
            candidates = ts_to_blocks[pivot_ts]
            
            for b_idx in candidates:
                gain = 0
                score = 0.0
                for ts_idx in block_ts_indices[b_idx]:
                    if ts_idx in remaining:
                        gain += 1
                        score += 1.0 / (counts[ts_idx] + 1)
                
                if gain > best_gain:
                    best_gain = gain
                    best_score = score
                    best_block = b_idx
                elif gain == best_gain:
                    # Tie-breaking with noise
                    if score + random.random() * 0.1 > best_score:
                        best_score = score
                        best_block = b_idx
            
            if best_block == -1:
                break
                
            selected.append(best_block)
            
            for ts_idx in block_ts_indices[best_block]:
                counts[ts_idx] += 1
                if counts[ts_idx] == 1:
                    remaining.discard(ts_idx)
        
        # Local Search Phase
        selected_set = set(selected)
        
        def is_redundant(b_idx):
            for ts_idx in block_ts_indices[b_idx]:
                if counts[ts_idx] == 1:
                    return False
            return True

        # Reduction
        changed = True
        while changed:
            changed = False
            for i in range(len(selected) - 1, -1, -1):
                b_idx = selected[i]
                if is_redundant(b_idx):
                    for ts_idx in block_ts_indices[b_idx]:
                        counts[ts_idx] -= 1
                    selected.pop(i)
                    selected_set.remove(b_idx)
                    changed = True
                    if time.time() > deadline:
                        return selected
        
        # Swap
        max_swaps = 100
        swaps_made = 0
        
        while swaps_made < max_swaps and time.time() < deadline:
            progress = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for i in indices:
                if time.time() > deadline:
                    return selected
                    
                b_out = selected[i]
                
                critical_ts = []
                for ts_idx in block_ts_indices[b_out]:
                    if counts[ts_idx] == 1:
                        critical_ts.append(ts_idx)
                
                if not critical_ts:
                    for ts_idx in block_ts_indices[b_out]:
                        counts[ts_idx] -= 1
                    selected.pop(i)
                    selected_set.remove(b_out)
                    swaps_made += 1
                    progress = True
                    break
                
                candidates = ts_to_blocks[critical_ts[0]]
                best_in = -1
                best_score = -1
                
                for b_in in candidates:
                    if b_in in selected_set:
                        continue
                    
                    # Check if b_in covers all critical_ts
                    covers_all = True
                    for c in critical_ts:
                        if c not in block_ts_sets[b_in]:
                            covers_all = False
                            break
                    
                    if not covers_all:
                        continue
                    
                    score = 0.0
                    for ts_idx in block_ts_indices[b_in]:
                        score += 1.0 / (counts[ts_idx] + 1)
                        
                    if score > best_score:
                        best_score = score
                        best_in = b_in
                
                if best_in != -1:
                    for ts_idx in block_ts_indices[b_out]:
                        counts[ts_idx] -= 1
                    for ts_idx in block_ts_indices[best_in]:
                        counts[ts_idx] += 1
                    
                    selected[i] = best_in
                    selected_set.remove(b_out)
                    selected_set.add(best_in)
                    
                    swaps_made += 1
                    progress = True
                    break
            
            if not progress:
                break

        return selected

    best_solution = None
    best_len = float('inf')
    
    num_restarts = 100 # Can do more now that greedy is faster
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        
        sol = run_greedy_local_search(i * 12345 + 42)
        
        if sol is not None:
            if len(sol) < best_len:
                best_len = len(sol)
                best_solution = sol[:]

    final_blocks = []
    if best_solution:
        for b_idx in best_solution:
            final_blocks.append(list(all_blocks[b_idx]))
    else:
        # Fallback
        sol = run_greedy_local_search(0)
        if sol:
             for b_idx in sol:
                final_blocks.append(list(all_blocks[b_idx]))
        else:
             raise Exception("Greedy failed")

    return final_blocks
