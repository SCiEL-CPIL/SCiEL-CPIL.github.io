<<<<<<< SEARCH
def _neh_block_internal(block_jobs, pt_list, n_machines):
    """
    Optimize the internal order of jobs within a block using NEH heuristic.
    """
    if len(block_jobs) <= 1:
        return list(block_jobs)
    
    # Sort by total processing time (descending)
    totals = [sum(pt_list[j]) for j in block_jobs]
    sorted_indices = sorted(range(len(block_jobs)), key=lambda i: totals[i], reverse=True)
    
    seq = [block_jobs[sorted_indices[0]]]
    
    for idx in sorted_indices[1:]:
        job = block_jobs[idx]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, pt_list)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    
    return seq
=======
def _neh_block_internal(block_jobs, pt_list, n_machines):
    """
    Optimize the internal order of jobs within a block using NEH heuristic.
    Uses faster makespan evaluation for insertion positions.
    """
    if len(block_jobs) <= 1:
        return list(block_jobs)
    
    # Sort by total processing time (descending)
    totals = [sum(pt_list[j]) for j in block_jobs]
    sorted_indices = sorted(range(len(block_jobs)), key=lambda i: totals[i], reverse=True)
    
    seq = [block_jobs[sorted_indices[0]]]
    
    for idx in sorted_indices[1:]:
        job = block_jobs[idx]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            # Build new sequence and compute makespan
            if pos == 0:
                new_seq = [job] + seq
            elif pos == len(seq):
                new_seq = seq + [job]
            else:
                new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, pt_list)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    
    return seq
>>>>>>> REPLACE

