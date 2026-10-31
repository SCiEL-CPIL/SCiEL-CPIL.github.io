import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    best_solution = None
    best_len = float('inf')
    
    # Precomputation
    # 1. All t-subsets
    all_tsubs = list(itertools.combinations(range(v), t))
    tsub_to_id = {ts: i for i, ts in enumerate(all_tsubs)}
    num_tsubs = len(all_tsubs)
    
    # 2. All k-subsets
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # 3. Map blocks to t-subset IDs
    block_to_tsub_ids = []
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    
    for i, b in enumerate(all_blocks):
        tsub_ids = []
        for ts in itertools.combinations(b, t):
            tid = tsub_to_id[ts]
            tsub_ids.append(tid)
            tsub_to_blocks[tid].append(i)
        block_to_tsub_ids.append(tsub_ids)
        
    initial_gain = len(block_to_tsub_ids[0]) if block_to_tsub_ids else 0
    
    # Run multiple instances
    while time.time() < deadline:
        gains = [initial_gain] * num_blocks
        coverage_count = [0] * num_tsubs
        selected_indices = []
        
        while True:
            if time.time() > deadline:
                break
                
            if num_blocks == 0:
                break

            max_gain = max(gains)
            if max_gain <= 0:
                break
                
            count = 0
            best_idx = -1
            for i in range(num_blocks):
                if gains[i] == max_gain:
                    count += 1
                    if random.random() < 1.0/count:
                        best_idx = i
            
            selected_indices.append(best_idx)
            
            for tid in block_to_tsub_ids[best_idx]:
                coverage_count[tid] += 1
                if coverage_count[tid] == 1:
                    for b_idx in tsub_to_blocks[tid]:
                        gains[b_idx] -= 1
                        
        # Convert to blocks
        # Redundancy removal
        sol_tsub_counts = [0] * num_tsubs
        for i in selected_indices:
            for tid in block_to_tsub_ids[i]:
                sol_tsub_counts[tid] += 1
                
        final_indices = []
        for i in selected_indices:
            is_essential = False
            for tid in block_to_tsub_ids[i]:
                if sol_tsub_counts[tid] == 1:
                    is_essential = True
                    break
            if is_essential:
                final_indices.append(i)
                
        final_blocks = [list(all_blocks[i]) for i in final_indices]
        
        if len(final_blocks) < best_len:
            best_len = len(final_blocks)
            best_solution = final_blocks
            
    if best_solution is None:
        # Fallback: return greedy result without redundancy removal if time ran out during first run?
        # Or just return empty?
        # Let's return the last known best, or empty.
        return []
        
    return best_solution
