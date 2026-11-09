"""
Flow-shop permutation construction using Variable Neighborhood Search (VNS).
Strategy:
1. Initialize with NEH heuristic for a strong starting point.
2. Use VNS framework that systematically cycles through neighborhoods of increasing size:
   - N1: adjacent swaps
   - N2: 2-opt block reversals
   - N3: 3-opt segment relocations (Or-opt with block size 3)
3. When stuck in a local optimum for the current neighborhood, shake the solution
   by applying a random move from a larger neighborhood, then re-optimize in the
   smallest neighborhood.
4. This forces the search out of local optima that smaller move sets cannot escape.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_vec(order, P):
    """
    Compute makespan for a given job order using vectorized NumPy operations.
    P: (n_jobs, n_machines)
    order: list/array of job indices
    """
    if len(order) == 0:
        return 0.0
    
    n_jobs_seq = len(order)
    n_machines = P.shape[1]
    
    # Extract processing times for the sequence
    seq_P = P[order]
    
    # C_machines[m] holds the completion time of the last job processed on machine m
    c_machines = np.zeros(n_machines)
    last_completion = 0.0
    
    for j in range(n_jobs_seq):
        row = seq_P[j]
        c_curr = 0.0
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
        last_completion = c_curr
        
    return last_completion

def _neh_heuristic(P):
    """
    NEH heuristic for initial solution.
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan_vec(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def _apply_swap(seq, i, j):
    """Apply adjacent or arbitrary swap."""
    new_seq = seq.copy()
    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
    return new_seq

def _apply_2opt_reverse(seq, i, j):
    """Apply 2-opt: reverse the segment from i to j (inclusive)."""
    if i > j:
        i, j = j, i
    new_seq = seq[:i] + seq[i:j+1][::-1] + seq[j+1:]
    return new_seq

def _apply_or_opt(seq, start, length, ins_pos):
    """Apply Or-opt: move block of given length from start to ins_pos."""
    block = seq[start:start + length]
    remaining = seq[:start] + seq[start + length:]
    # Adjust ins_pos if it was after the removed block
    if ins_pos > start:
        ins_pos -= length
    new_seq = remaining[:ins_pos] + block + remaining[ins_pos:]
    return new_seq

def _local_search_N1(seq, P, time_limit):
    """Local search using adjacent swaps (N1 neighborhood)."""
    n = len(seq)
    current_seq = seq.copy()
    current_ms = _makespan_vec(current_seq, P)
    
    improved = True
    while improved and (time.time() < time_limit):
        improved = False
        for i in range(n - 1):
            # Check swap of i and i+1
            new_seq = _apply_swap(current_seq, i, i + 1)
            new_ms = _makespan_vec(new_seq, P)
            if new_ms < current_ms:
                current_seq = new_seq
                current_ms = new_ms
                improved = True
                # Restart scan from beginning after improvement
                break
    
    return current_seq, current_ms

def _local_search_N2(seq, P, time_limit):
    """Local search using 2-opt reversals (N2 neighborhood)."""
    n = len(seq)
    current_seq = seq.copy()
    current_ms = _makespan_vec(current_seq, P)
    
    improved = True
    while improved and (time.time() < time_limit):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, min(i + 6, n)):  # Limit segment length for speed
                new_seq = _apply_2opt_reverse(current_seq, i, j)
                new_ms = _makespan_vec(new_seq, P)
                if new_ms < current_ms:
                    current_seq = new_seq
                    current_ms = new_ms
                    improved = True
                    break
            if improved:
                break
    
    return current_seq, current_ms

def _shake_N2(seq, P):
    """Shake by applying a random 2-opt reversal."""
    n = len(seq)
    if n < 3:
        return seq
    i = random.randint(0, n - 3)
    j = random.randint(i + 2, n - 1)
    return _apply_2opt_reverse(seq, i, j)

def _shake_N3(seq, P):
    """Shake by applying a random Or-opt move with block size 2-3."""
    n = len(seq)
    if n < 4:
        return seq
    length = random.choice([2, 3])
    if n - length < 1:
        return seq
    start = random.randint(0, n - length)
    ins_pos = random.randint(0, n - length)
    # Avoid trivial move
    if ins_pos == start or ins_pos == start + 1:
        ins_pos = (ins_pos + 1) % (n - length)
    return _apply_or_opt(seq, start, length, ins_pos)

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using VNS with NEH initialization.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    time_limit = start_time + 145.0  # Safety margin
    
    # 1. Initialize with NEH
    current_seq = _neh_heuristic(P)
    current_ms = _makespan_vec(current_seq, P)
    
    best_seq = current_seq.copy()
    best_ms = current_ms
    
    # VNS Parameters
    max_k = 3  # Maximum neighborhood size
    min_k = 1
    max_shakes = 50  # Max shakes per neighborhood level before moving up
    
    k = min_k
    
    while (time.time() < time_limit) and k <= max_k:
        # Local search in current neighborhood
        if k == 1:
            current_seq, current_ms = _local_search_N1(current_seq, P, time_limit)
        elif k == 2:
            current_seq, current_ms = _local_search_N2(current_seq, P, time_limit)
        elif k == 3:
            # For N3, do a few iterations of Or-opt local search
            improved = True
            shake_count = 0
            while improved and shake_count < max_shakes and (time.time() < time_limit):
                improved = False
                shake_count += 1
                # Try random Or-opt moves
                for _ in range(20):
                    n = len(current_seq)
                    length = random.choice([1, 2, 3])
                    if n - length < 1:
                        continue
                    start = random.randint(0, n - length)
                    ins_pos = random.randint(0, n - length)
                    new_seq = _apply_or_opt(current_seq, start, length, ins_pos)
                    new_ms = _makespan_vec(new_seq, P)
                    if new_ms < current_ms:
                        current_seq = new_seq
                        current_ms = new_ms
                        improved = True
                        break
        
        if current_ms < best_ms:
            best_ms = current_ms
            best_seq = current_seq.copy()
        
        # Check if time is running out
        if time.time() >= time_limit - 5.0:
            break
        
        # If stuck (no improvement in local search), shake and increase k
        # Simple heuristic: if we just did local search and it didn't improve, move to next k
        # For simplicity, always increment k after each local search phase
        if k < max_k:
            k += 1
            # Shake the solution
            if k == 2:
                current_seq = _shake_N2(current_seq, P)
            elif k == 3:
                current_seq = _shake_N3(current_seq, P)
        
        # If we've cycled through all neighborhoods, reset to k=1 with a shake
        if k > max_k:
            # Final diversification: apply several random moves
            for _ in range(10):
                current_seq = _shake_N3(current_seq, P)
            k = min_k
    
    return best_seq
# EVOLVE-BLOCK-END
