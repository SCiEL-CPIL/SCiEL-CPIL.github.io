# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using a hybrid of:
    1. Fast NEH heuristic for initial solutions.
    2. Tabu Search with insertion moves (not just swaps) and adaptive tabu tenure.
    This addresses the stagnation of simple SA by using a more robust local search
    operator (insertion) and memory (tabu list) to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Fast fallback for very large instances
    if n_jobs * n_machines > 500000:
        return list(np.argsort(-processing_times.sum(axis=1)))
    
    def makespan(seq):
        """Compute makespan for a given sequence of job indices."""
        if len(seq) == 0:
            return 0.0
        completion = np.zeros(n_machines)
        for job in seq:
            prev = completion[0]
            curr = prev + processing_times[job, 0]
            completion[0] = curr
            for m in range(1, n_machines):
                prev = curr
                curr = max(prev, completion[m]) + processing_times[job, m]
                completion[m] = curr
        return completion[-1]
    
    def neh_heuristic():
        """NEH (Nawaz, Enscore, Ham) heuristic."""
        totals = processing_times.sum(axis=1)
        order = list(np.argsort(-totals))
        
        seq = [order[0]]
        for job in order[1:]:
            best_pos = 0
            best_ms = float('inf')
            for pos in range(len(seq) + 1):
                trial = seq[:pos] + [job] + seq[pos:]
                ms = makespan(trial)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            seq = seq[:best_pos] + [job] + seq[best_pos:]
        
        return seq
    
    def tabu_search(initial_seq, max_iters=20000):
        """
        Tabu Search with insertion moves.
        """
        current_seq = list(initial_seq)
        current_ms = makespan(current_seq)
        best_seq = list(current_seq)
        best_ms = current_ms
        
        # Tabu list: stores (job, from_pos, to_pos) or just job and positions
        # To keep it simple and fast, we store (job_id, from_index) as tabu for a fixed tenure.
        # Actually, a better tabu key for insertion is (job_id, from_pos).
        # If we move job j from i to k, we tabu moving j from i back to k (or similar).
        # Let's use a simple tabu list of (job, from_pos) pairs.
        tabu_tenure = max(10, n_jobs // 2)
        tabu_list = []  # List of (job, from_pos)
        tabu_counter = 0
        
        no_improve_count = 0
        iter_count = 0
        
        while iter_count < max_iters and time.time() < deadline:
            # Generate neighbors: all possible insertion moves
            # Move job from i to k (k != i, k != i+1 to avoid trivial swaps? No, allow all)
            best_neighbor_ms = float('inf')
            best_neighbor_seq = None
            best_neighbor_move = None
            
            # To speed up, we can limit the search space or use a random subset if n is large.
            # For n_jobs <= 100, n^2 = 10000 neighbors. Makespan is O(n*m).
            # 10000 * 100 * 10 = 10M ops. Might be slow in Python.
            # Let's optimize makespan calculation or limit neighbors.
            
            # Optimization: Precompute completion times? Hard with insertions.
            # Let's try to be smarter: only consider moves that are not tabu.
            # And maybe randomize the order of checking or limit to a subset if time is tight.
            
            # For efficiency, let's limit the number of moves checked per iteration if n is large.
            max_moves_to_check = min(n_jobs * n_jobs, 2000)
            moves = []
            for i in range(n_jobs):
                for k in range(n_jobs):
                    if i == k or k == i + 1: # Skip trivial
                        continue
                    # Check tabu
                    is_tabu = (current_seq[i], i) in tabu_list
                    if is_tabu:
                        continue
                    moves.append((i, k))
            
            if not moves:
                # If all moves are tabu, clear tabu list or pick best tabu move (aspiration)
                # For simplicity, clear tabu list and restart move generation
                tabu_list = []
                for i in range(n_jobs):
                    for k in range(n_jobs):
                        if i == k or k == i + 1:
                            continue
                        moves.append((i, k))
            
            # Shuffle moves to randomize tie-breaking and exploration
            random.shuffle(moves)
            
            # Check only a subset of moves if too many
            if len(moves) > max_moves_to_check:
                moves = moves[:max_moves_to_check]
            
            for i, k in moves:
                # Create new sequence
                job = current_seq[i]
                new_seq = current_seq[:i] + current_seq[i+1:]
                # Adjust k if i < k
                if i < k:
                    k -= 1
                new_seq = new_seq[:k] + [job] + new_seq[k:]
                
                ms = makespan(new_seq)
                
                # Aspiration criterion: accept if it improves best known
                if ms < best_ms:
                    is_tabu = False # Override tabu
                    
                if not is_tabu and ms < best_neighbor_ms:
                    best_neighbor_ms = ms
                    best_neighbor_seq = new_seq
                    best_neighbor_move = (i, k, job)
            
            if best_neighbor_seq is None:
                break # No moves found
            
            # Accept the best non-tabu move (or best if aspiration)
            current_seq = best_neighbor_seq
            current_ms = best_neighbor_ms
            
            # Update tabu list
            i, k, job = best_neighbor_move
            tabu_list.append((job, i))
            tabu_counter += 1
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
            
            # Update best
            if current_ms < best_ms:
                best_ms = current_ms
                best_seq = list(current_seq)
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            iter_count += 1
            
            # Adaptive tabu tenure: if no improvement for a while, increase tenure to force diversification?
            # Or decrease to intensify? Usually, if stuck, increase.
            if no_improve_count > 50:
                tabu_tenure = int(tabu_tenure * 1.2)
                no_improve_count = 0
            
        return best_seq, best_ms
    
    # Multi-start Tabu Search
    # Start with NEH
    best_seq = neh_heuristic()
    best_ms = makespan(best_seq)
    
    # Try a few more starts if time permits
    # 1. LPT (Longest Processing Time)
    lpt_seq = list(np.argsort(-processing_times.sum(axis=1)))
    ms_lpt = makespan(lpt_seq)
    if ms_lpt < best_ms:
        best_ms = ms_lpt
        best_seq = lpt_seq
        
    # 2. SPT (Shortest Processing Time)
    spt_seq = list(np.argsort(processing_times.sum(axis=1)))
    ms_spt = makespan(spt_seq)
    if ms_spt < best_ms:
        best_ms = ms_spt
        best_seq = spt_seq

    # Run Tabu Search from the best initial solution
    final_seq, final_ms = tabu_search(best_seq)
    
    if final_ms < best_ms:
        best_ms = final_ms
        best_seq = final_seq
        
    # If time remains, try another start from a perturbed version of the best
    if time.time() < deadline - 10:
        # Perturb best_seq
        perturbed = list(best_seq)
        # Randomly swap a few pairs
        for _ in range(3):
            i, j = random.sample(range(n_jobs), 2)
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
        
        final_seq2, final_ms2 = tabu_search(perturbed, max_iters=10000)
        if final_ms2 < best_ms:
            best_ms = final_ms2
            best_seq = final_seq2

    return list(best_seq)
# EVOLVE-BLOCK-END
