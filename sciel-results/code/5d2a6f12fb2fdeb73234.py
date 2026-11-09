"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    # C[i][j] = completion time of job i on machine j
    # We can compute this in O(n_jobs * n_machines)
    completion = np.zeros((n_jobs, n_machines))
    for i, job_idx in enumerate(order):
        for j in range(n_machines):
            p = processing_times[job_idx, j]
            if j == 0:
                if i == 0:
                    completion[i, j] = p
                else:
                    completion[i, j] = completion[i-1, j] + p
            else:
                if i == 0:
                    completion[i, j] = completion[i, j-1] + p
                else:
                    completion[i, j] = max(completion[i-1, j], completion[i, j-1]) + p
    return completion[-1, -1]

def _neh_rule(processing_times):
    """Construct a permutation using the NEH heuristic."""
    n_jobs, n_machines = processing_times.shape
    jobs = list(range(n_jobs))
    
    # Sort jobs by total processing time in descending order
    # Tie-breaker: random or by last machine (standard NEH often uses total, then maybe specific rules)
    # Here we use total processing time descending.
    totals = processing_times.sum(axis=1)
    # To ensure determinism in sorting but allow randomness in later steps, 
    # we sort by (-total, job_index) or just -total. 
    # Standard NEH: Sort by total processing time descending.
    sorted_jobs = sorted(jobs, key=lambda j: -totals[j])
    
    # Insert each job into the best position in the current sequence
    seq = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        current_len = len(seq)
        
        for pos in range(current_len + 1):
            # Create new sequence
            new_seq = seq[:pos] + [job] + seq[pos:]
            # Evaluate makespan
            ms = _makespan(processing_times, new_seq)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        seq.insert(best_pos, job)
        
    return seq

def _local_search_insertion(processing_times, seq, deadline):
    """First-improvement insertion local search."""
    n_jobs = len(seq)
    current_seq = list(seq)
    current_ms = _makespan(processing_times, current_seq)
    
    improved = True
    while improved:
        improved = False
        start_time = time.time()
        
        for i in range(n_jobs):
            if time.time() > deadline:
                return current_seq, current_ms
            
            # Remove job i
            job = current_seq[i]
            temp_seq = current_seq[:i] + current_seq[i+1:]
            
            # Try inserting at all other positions
            for j in range(len(temp_seq) + 1):
                if j == i:
                    continue
                if j == i - 1:
                    # This would be the same position effectively after removal/insertion logic
                    # Actually, if we remove i, positions 0..n-2 are valid.
                    # Original i corresponds to insertion at i (if i < n-1) or i-1 (if i == n-1)?
                    # Let's just check all valid indices in temp_seq.
                    pass
                
                # Insert job at position j in temp_seq
                new_seq = temp_seq[:j] + [job] + temp_seq[j:]
                
                # Quick check: if the moved job is at the same relative spot, skip?
                # Simple check: if new_seq == current_seq: continue
                
                ms = _makespan(processing_times, new_seq)
                if ms < current_ms:
                    current_seq = new_seq
                    current_ms = ms
                    improved = True
                    # Restart the outer loop or continue? 
                    # First improvement: we can break inner and restart outer for simplicity/safety
                    break
            if improved:
                # Restart the scan from the beginning to ensure we don't miss things
                # Or continue? First improvement usually restarts or continues.
                # Restarting is safer for convergence but slower.
                # Let's restart the outer loop immediately.
                break
                
    return current_seq, current_ms

def _perturb(processing_times, seq, strength=3):
    """Or-opt style perturbation: remove a block of jobs and reinsert elsewhere."""
    n_jobs = len(seq)
    if n_jobs < 4:
        # Simple swap for very small problems
        i, j = random.sample(range(n_jobs), 2)
        new_seq = list(seq)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        return new_seq
        
    # Choose a random block length (1, 2, or 3)
    block_len = random.randint(1, min(strength, n_jobs - 1))
    # Choose start index
    start = random.randint(0, n_jobs - block_len)
    
    # Extract block
    block = seq[start:start + block_len]
    remaining = seq[:start] + seq[start + block_len:]
    
    # Choose insertion position in remaining
    insert_pos = random.randint(0, len(remaining))
    
    new_seq = remaining[:insert_pos] + block + remaining[insert_pos:]
    return new_seq

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    # Edge case: 0 or 1 job
    if n_jobs <= 1:
        return list(range(n_jobs))
        
    # Deadline: 150s limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial solution using NEH
    best_seq = _neh_rule(processing_times)
    best_ms = _makespan(processing_times, best_seq)
    
    # Local search on initial solution
    best_seq, best_ms = _local_search_insertion(processing_times, best_seq, deadline)
    
    # Iterated Local Search (ILS)
    # Random seed for reproducibility within the run, but different across calls?
    # The task doesn't specify deterministic requirement, but standard practice is to allow randomness.
    # However, to be safe against flaky scores, we might want to seed based on input?
    # No, standard is just random.
    
    while time.time() < deadline:
        # Perturb the current best solution
        current_seq = _perturb(processing_times, best_seq)
        
        # Local search from perturbed solution
        current_seq, current_ms = _local_search_insertion(processing_times, current_seq, deadline)
        
        # Accept if better (or equal, to allow random walk? No, strict improvement for ILS usually)
        if current_ms < best_ms:
            best_ms = current_ms
            best_seq = current_seq
        # Optionally, accept worse solutions with probability (SA-like) to escape local optima?
        # Pure ILS often uses acceptance criteria. Let's stick to strict improvement for now
        # or add a small probability of accepting worse solutions to diversify.
        # Given the time limit, strict improvement is safer.
        
    return list(best_seq)

# EVOLVE-BLOCK-END
