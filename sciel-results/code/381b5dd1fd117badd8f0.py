"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """
    Compute the makespan for a given permutation of job indices.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0
    
    prev_job_completion = np.zeros(n_machines)
    
    for job in perm:
        current_job_completion_on_prev_machine = 0.0
        for m in range(n_machines):
            p = processing_times[job, m]
            if m == 0:
                start = prev_job_completion[0]
                current_job_completion_on_prev_machine = start + p
            else:
                start = max(current_job_completion_on_prev_machine, prev_job_completion[m])
                current_job_completion_on_prev_machine = start + p
            
            prev_job_completion[m] = current_job_completion_on_prev_machine
    
    return prev_job_completion[-1]


def _initialize_static_sort(processing_times):
    """
    Initialize permutation by sorting jobs by a composite key derived from 
    total processing time and maximum processing time (bottleneck).
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    total_pt = processing_times.sum(axis=1)
    max_pt = processing_times.max(axis=1)
    
    # Composite key: Total processing time is primary, max processing time as tie-breaker/adjustment
    # The "longest path" for a job is its total processing time.
    # We add a small weight to the max to prioritize jobs with a single huge bottleneck.
    keys = total_pt + 0.1 * max_pt
    
    # Sort in descending order of keys
    sorted_indices = np.argsort(-keys)
    
    return list(sorted_indices)


def _tabu_search_diversified(initial_perm, processing_times, time_limit=10.0):
    """
    Tabu search with enhanced diversification.
    Uses a tabu list to prohibit recent moves.
    If stuck, applies a stronger random perturbation.
    """
    n_jobs = len(initial_perm)
    if n_jobs <= 1:
        return initial_perm
    
    current_perm = list(initial_perm)
    current_makespan = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_makespan = current_makespan
    
    # Tabu list: stores (move_type, i, j) -> expiration_iteration
    tabu_list = {}
    # Dynamic tabu tenure: larger for larger problems
    tabu_tenure = max(5, n_jobs // 2)
    
    start_time = time.time()
    iteration = 0
    no_improve_count = 0
    max_no_improve = 30  # Lower threshold to trigger diversification sooner
    
    while time.time() - start_time < time_limit:
        iteration += 1
        best_neighbor_perm = None
        best_neighbor_makespan = current_makespan
        best_neighbor_move = None
        
        # Generate neighborhood
        # For small n, explore all adjacent swaps and some moves.
        # For large n, sample.
        moves_to_try = []
        
        if n_jobs <= 40:
            # All adjacent swaps
            for i in range(n_jobs - 1):
                moves_to_try.append(('swap', i, i + 1))
            # Sample moves
            for _ in range(min(100, n_jobs * 2)):
                i = np.random.randint(0, n_jobs)
                j = np.random.randint(0, n_jobs)
                if i != j:
                    moves_to_try.append(('move', i, j))
        else:
            # Sample only
            num_samples = 300
            for _ in range(num_samples):
                if np.random.rand() < 0.5:
                    i = np.random.randint(0, n_jobs - 1)
                    moves_to_try.append(('swap', i, i + 1))
                else:
                    i = np.random.randint(0, n_jobs)
                    j = np.random.randint(0, n_jobs)
                    if i != j:
                        moves_to_try.append(('move', i, j))
        
        for move in moves_to_try:
            m_type, i, j = move
            
            # Normalize move key for tabu check
            if m_type == 'swap':
                move_key = ('swap', min(i, j), max(i, j))
            else:
                # For moves, the tabu key is (i, j) where i is source, j is dest
                # Note: 'move' in this implementation removes from i and inserts at j
                # We need to be consistent with how we generate and check.
                move_key = ('move', i, j)
            
            # Check tabu
            is_tabu = move_key in tabu_list and tabu_list[move_key] > iteration
            
            # Generate new permutation
            new_perm = current_perm.copy()
            if m_type == 'swap':
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            else:
                # Move job at i to position j
                job_to_move = new_perm.pop(i)
                if i < j:
                    j -= 1
                new_perm.insert(j, job_to_move)
            
            new_makespan = _makespan(new_perm, processing_times)
            
            # Aspiration: accept if it improves global best
            is_aspiration = new_makespan < best_makespan
            
            if is_tabu and not is_aspiration:
                continue
            
            if new_makespan < best_neighbor_makespan:
                best_neighbor_makespan = new_makespan
                best_neighbor_perm = new_perm
                best_neighbor_move = move_key
        
        if best_neighbor_perm is None:
            # No improving non-tabu move found
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Diversification: Stronger perturbation
                # Apply 3-5 random moves
                num_kicks = min(5, n_jobs // 2)
                for _ in range(num_kicks):
                    i = np.random.randint(0, n_jobs)
                    j = np.random.randint(0, n_jobs)
                    if i != j:
                        job_to_move = current_perm.pop(i)
                        if i < j:
                            j -= 1
                        current_perm.insert(j, job_to_move)
                
                current_makespan = _makespan(current_perm, processing_times)
                no_improve_count = 0
                # Clear tabu list to allow new directions
                tabu_list.clear()
            else:
                # Mild perturbation or accept a random non-tabu move if available?
                # For now, just try a random move
                i = np.random.randint(0, n_jobs)
                j = np.random.randint(0, n_jobs)
                if i != j:
                    job_to_move = current_perm.pop(i)
                    if i < j:
                        j -= 1
                    current_perm.insert(j, job_to_move)
                    current_makespan = _makespan(current_perm, processing_times)
        else:
            no_improve_count = 0
            # Update tabu list
            tabu_list[best_neighbor_move] = iteration + tabu_tenure
            
            # Clean up tabu list
            for key in list(tabu_list.keys()):
                if tabu_list[key] <= iteration:
                    del tabu_list[key]
            
            # Accept move
            current_perm = best_neighbor_perm
            current_makespan = best_neighbor_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_perm = list(current_perm)
    
    return best_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on local synthetic flow-shop instances.
    Uses static sorting with composite key and diversified tabu search.
    """
    if len(processing_times) == 0:
        return []
    
    # Initial solution: Static Sort
    initial_perm = _initialize_static_sort(processing_times)
    
    # Refine with tabu search
    n_jobs = len(processing_times)
    if n_jobs <= 20:
        time_limit = 15.0
    elif n_jobs <= 50:
        time_limit = 10.0
    else:
        time_limit = 5.0
    
    final_perm = _tabu_search_diversified(initial_perm, processing_times, time_limit=time_limit)
    
    return list(final_perm)
# EVOLVE-BLOCK-END
