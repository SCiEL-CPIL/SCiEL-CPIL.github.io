"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, processing_times):
    """Compute makespan for a given job order."""
    n_machines = processing_times.shape[1]
    if n_machines == 0 or len(order) == 0:
        return 0.0
    completion = np.zeros(n_machines)
    for job in order:
        p = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            prev = max(prev, completion[m]) + p[m]
            completion[m] = prev
    return float(completion[-1])


def _neh_initialization(processing_times):
    """
    NEH heuristic: insert jobs one by one in order of decreasing total processing time,
    choosing the position that minimizes the makespan at each step.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Order jobs by decreasing total processing time
    totals = processing_times.sum(axis=1)
    job_order = list(np.argsort(-totals))
    
    # Build sequence greedily
    seq = [job_order[0]]
    current_makespan = _makespan(seq, processing_times)
    
    for job in job_order[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
        current_makespan = best_ms
    
    return seq


def _improve_with_local_search(order, processing_times, time_limit=110.0):
    """
    2-opt and Or-opt local search to improve the initial solution.
    """
    start_time = time.time()
    n_jobs = len(order)
    if n_jobs <= 1:
        return order
    
    best_order = list(order)
    best_ms = _makespan(best_order, processing_times)
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() - start_time < time_limit and no_improve_count < max_no_improve:
        improved = False
        
        # 2-opt: reverse a segment
        for i in range(n_jobs):
            for j in range(i + 2, n_jobs):
                new_order = best_order[:i] + best_order[i:j+1][::-1] + best_order[j+1:]
                ms = _makespan(new_order, processing_times)
                if ms < best_ms - 1e-9:
                    best_order = new_order
                    best_ms = ms
                    improved = True
                    no_improve_count = 0
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Or-opt: move a segment of 1-3 jobs to a new position
        for seg_len in [1, 2, 3]:
            for i in range(n_jobs - seg_len + 1):
                segment = best_order[i:i+seg_len]
                rest = best_order[:i] + best_order[i+seg_len:]
                for j in range(len(rest) + 1):
                    if j == i or j == i + seg_len:
                        continue
                    new_order = rest[:j] + segment + rest[j:]
                    ms = _makespan(new_order, processing_times)
                    if ms < best_ms - 1e-9:
                        best_order = new_order
                        best_ms = ms
                        improved = True
                        no_improve_count = 0
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
    
    return best_order


def _block_decomposition(processing_times):
    """
    Partition jobs into contiguous blocks based on processing time similarity.
    Uses a greedy clustering approach: sort by total processing time, then
    group similar jobs into blocks.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    totals = processing_times.sum(axis=1)
    sorted_indices = list(np.argsort(totals))
    
    # Determine block size based on problem size
    if n_jobs <= 10:
        block_size = n_jobs
        return [sorted_indices]
    
    # Use a heuristic to determine number of blocks
    n_blocks = max(2, min(8, n_jobs // 4))
    block_size = n_jobs // n_blocks
    remainder = n_jobs % n_blocks
    
    blocks = []
    idx = 0
    for b in range(n_blocks):
        size = block_size + (1 if b < remainder else 0)
        blocks.append(sorted_indices[idx:idx+size])
        idx += size
    
    return blocks


def order_jobs(processing_times):
    """
    Main function: construct a job permutation minimizing makespan.
    Uses NEH initialization, block-based decomposition, and local search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: NEH initialization
    initial_order = _neh_initialization(processing_times)
    best_order = list(initial_order)
    best_ms = _makespan(best_order, processing_times)
    
    # Step 2: Block-based decomposition and local optimization
    if n_jobs > 10 and time.time() < deadline - 5.0:
        blocks = _block_decomposition(processing_times)
        
        # Optimize internal order of each block
        optimized_blocks = []
        for block in blocks:
            if len(block) <= 1:
                optimized_blocks.append(block)
            else:
                # Use NEH within the block
                block_pt = processing_times[block]
                block_order = _neh_initialization(block_pt)
                optimized_blocks.append([block[i] for i in block_order])
        
        # Try different permutations of blocks using 2-opt on block order
        n_blocks = len(optimized_blocks)
        if n_blocks > 1 and n_blocks < 20:
            best_block_order = list(range(n_blocks))
            best_block_ms = float('inf')
            
            # Simple 2-opt on block order
            for iteration in range(100):
                if time.time() > deadline - 5.0:
                    break
                improved = False
                for i in range(n_blocks):
                    for j in range(i + 1, n_blocks):
                        new_block_order = best_block_order[:]
                        new_block_order[i], new_block_order[j] = new_block_order[j], new_block_order[i]
                        full_order = []
                        for b in new_block_order:
                            full_order.extend(optimized_blocks[b])
                        ms = _makespan(full_order, processing_times)
                        if ms < best_block_ms - 1e-9:
                            best_block_ms = ms
                            best_block_order = new_block_order
                            improved = True
                if not improved:
                    break
            
            # Build final order from best block arrangement
            final_order = []
            for b in best_block_order:
                final_order.extend(optimized_blocks[b])
            
            if _makespan(final_order, processing_times) < best_ms:
                best_order = final_order
                best_ms = _makespan(best_order, processing_times)
    
    # Step 3: Local search polish
    if time.time() < deadline - 2.0:
        remaining_time = deadline - time.time() - 2.0
        best_order = _improve_with_local_search(best_order, processing_times, time_limit=remaining_time)
    
    # Final safety check
    if time.time() > deadline - 1.0:
        pass  # Already at deadline, return what we have
    
    return best_order
# EVOLVE-BLOCK-END
