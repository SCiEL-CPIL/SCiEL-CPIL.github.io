"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(order, processing_times):
    """
    Calculate the makespan for a given job order in a flow shop.
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    
    prev_completion = np.zeros(n_machines)
    
    for j in order:
        curr_completion_0 = prev_completion[0] + processing_times[j, 0]
        
        if n_machines == 1:
            prev_completion[0] = curr_completion_0
            continue
            
        curr_completion = np.empty(n_machines)
        curr_completion[0] = curr_completion_0
        
        for m in range(1, n_machines):
            curr_completion[m] = max(curr_completion[m-1], prev_completion[m]) + processing_times[j, m]
            
        prev_completion = curr_completion
        
    return prev_completion[-1]


def _makespan_from_blocks(blocks, block_sequences, processing_times):
    """
    Compute makespan given a list of blocks and their internal sequences.
    blocks: list of lists of job indices (one per block)
    block_sequences: list of lists showing internal order within each block
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    
    prev_completion = np.zeros(n_machines)
    
    for block_idx, block in enumerate(blocks):
        seq = block_sequences[block_idx]
        for j in seq:
            curr_completion_0 = prev_completion[0] + processing_times[j, 0]
            
            if n_machines == 1:
                prev_completion[0] = curr_completion_0
                continue
                
            curr_completion = np.empty(n_machines)
            curr_completion[0] = curr_completion_0
            
            for m in range(1, n_machines):
                curr_completion[m] = max(curr_completion[m-1], prev_completion[m]) + processing_times[j, m]
                
            prev_completion = curr_completion
            
    return prev_completion[-1]


def _cluster_jobs(processing_times, n_clusters):
    """
    Cluster jobs into groups based on processing time correlation.
    Uses a simple greedy approach: sort by correlation with a reference job.
    Returns a list of lists of job indices.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= n_clusters:
        return [[j] for j in range(n_jobs)]
    
    # Compute pairwise cosine similarity
    # Normalize each job's processing time vector
    norms = np.linalg.norm(processing_times, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = processing_times / norms
    
    # Compute similarity matrix
    sim_matrix = normalized @ normalized.T
    
    # Greedy clustering: start with the job with highest total processing time
    job_sums = processing_times.sum(axis=1)
    start_job = np.argmax(job_sums)
    
    clusters = []
    assigned = np.zeros(n_jobs, dtype=bool)
    assigned[start_job] = True
    clusters.append([start_job])
    
    for _ in range(n_clusters - 1):
        # Find the next job that is most dissimilar to all current cluster centers
        # Use the job with lowest average similarity to existing clusters
        best_job = -1
        best_score = float('inf')
        
        for j in range(n_jobs):
            if assigned[j]:
                continue
            
            # Average similarity to all assigned jobs
            sim_to_assigned = sim_matrix[j, assigned].mean()
            
            # We want low similarity (dissimilar)
            if sim_to_assigned < best_score:
                best_score = sim_to_assigned
                best_job = j
        
        if best_job == -1:
            break
        
        assigned[best_job] = True
        clusters.append([best_job])
    
    # Assign remaining jobs to the most similar cluster
    for j in range(n_jobs):
        if not assigned[j]:
            # Find the cluster with highest average similarity
            best_cluster = 0
            best_sim = -float('inf')
            
            for c_idx, cluster in enumerate(clusters):
                avg_sim = sim_matrix[j, cluster].mean()
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_cluster = c_idx
            
            clusters[best_cluster].append(j)
    
    return clusters


def _optimize_block_internal(block_jobs, processing_times):
    """
    Optimize the internal sequence of jobs within a block using a simplified NEH.
    """
    n_jobs_in_block = len(block_jobs)
    
    if n_jobs_in_block <= 1:
        return block_jobs[:]
    
    # Get processing times for jobs in this block
    block_pt = processing_times[block_jobs]
    
    # Sort by sum of processing times (descending)
    job_sums = block_pt.sum(axis=1)
    sorted_indices = np.argsort(-job_sums)
    
    current_seq = [block_jobs[i] for i in sorted_indices]
    current_ms = _makespan(current_seq, processing_times)
    
    # Local search: try swapping adjacent pairs
    improved = True
    max_iters = 10
    iters = 0
    
    while improved and iters < max_iters and len(current_seq) > 1:
        improved = False
        iters += 1
        n = len(current_seq)
        
        for i in range(n - 1):
            # Try swapping positions i and i+1
            new_seq = current_seq[:]
            new_seq[i], new_seq[i+1] = new_seq[i+1], new_seq[i]
            new_ms = _makespan(new_seq, processing_times)
            
            if new_ms < current_ms:
                current_seq = new_seq
                current_ms = new_ms
                improved = True
    
    return current_seq


def _block_2opt_search(blocks, block_sequences, processing_times, deadline):
    """
    Perform 2-opt search on the block ordering.
    blocks: list of lists of job indices
    block_sequences: list of internal sequences for each block
    Returns updated blocks and block_sequences.
    """
    n_blocks = len(blocks)
    
    if n_blocks <= 1:
        return blocks, block_sequences
    
    current_ms = _makespan_from_blocks(blocks, block_sequences, processing_times)
    best_ms = current_ms
    best_blocks = [b[:] for b in blocks]
    best_seqs = [s[:] for s in block_sequences]
    
    no_improve_count = 0
    max_no_improve = 5
    
    while no_improve_count < max_no_improve and time.time() < deadline:
        improved = False
        
        for i in range(n_blocks - 1):
            if time.time() > deadline:
                break
            
            for j in range(i + 1, n_blocks):
                if time.time() > deadline:
                    break
                
                # Reverse blocks from i to j
                new_blocks = blocks[:i] + blocks[i:j+1][::-1] + blocks[j+1:]
                new_seqs = block_sequences[:i] + block_sequences[i:j+1][::-1] + block_sequences[j+1:]
                
                new_ms = _makespan_from_blocks(new_blocks, new_seqs, processing_times)
                
                if new_ms < current_ms:
                    blocks = new_blocks
                    block_sequences = new_seqs
                    current_ms = new_ms
                    improved = True
                    
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_blocks = [b[:] for b in blocks]
                        best_seqs = [s[:] for s in block_sequences]
                    
                    break
            
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
    
    return best_blocks, best_seqs


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using block-based decomposition.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Determine number of blocks based on problem size
    # For small problems, use fewer blocks; for larger, use more
    if n_jobs <= 10:
        n_blocks = max(2, n_jobs // 3)
    elif n_jobs <= 50:
        n_blocks = max(3, n_jobs // 5)
    else:
        n_blocks = max(5, min(20, n_jobs // 4))
    
    n_blocks = min(n_blocks, n_jobs)
    
    # Step 1: Cluster jobs into blocks
    clusters = _cluster_jobs(processing_times, n_blocks)
    
    # Step 2: Optimize internal sequence of each block
    block_sequences = []
    for block in clusters:
        opt_seq = _optimize_block_internal(block, processing_times)
        block_sequences.append(opt_seq)
    
    # Step 3: Order the blocks using 2-opt
    ordered_blocks, ordered_seqs = _block_2opt_search(clusters, block_sequences, processing_times, deadline)
    
    # Flatten to get final permutation
    final_order = []
    for seq in ordered_seqs:
        final_order.extend(seq)
    
    current_makespan = _makespan(final_order, processing_times)
    best_order = final_order[:]
    best_makespan = current_makespan
    
    # Step 4: Fine-tune with single-job moves within the remaining time
    if time.time() < deadline - 5.0:
        polish_order = best_order[:]
        polish_ms = best_makespan
        n = len(polish_order)
        
        for _ in range(5):  # A few polish iterations
            if time.time() > deadline:
                break
            any_improvement = False
            for i in range(n):
                if time.time() > deadline:
                    break
                job = polish_order[i]
                remaining = polish_order[:i] + polish_order[i+1:]
                best_pos = i
                best_pms = polish_ms
                
                for j in range(n):
                    if j == i:
                        continue
                    new_order = remaining[:j] + [job] + remaining[j:]
                    pms = _makespan(new_order, processing_times)
                    if pms < best_pms:
                        best_pms = pms
                        best_pos = j
                
                if best_pos != i:
                    remaining = polish_order[:i] + polish_order[i+1:]
                    polish_order = remaining[:best_pos] + [job] + remaining[best_pos:]
                    polish_ms = best_pms
                    any_improvement = True
                    break
            
            if not any_improvement:
                break
        
        if polish_ms < best_makespan:
            best_makespan = polish_ms
            best_order = polish_order[:]
    
    return best_order

# EVOLVE-BLOCK-END
