<<<<<<< SEARCH
def _tabu_search(processing_times, sequence, deadline):
    """
    Tabu Search with dynamic tabu list and aspiration criterion.
    Uses insertion moves with tabu tenure management.
    """
    n_jobs = len(sequence)
    if n_jobs < 2:
        return list(sequence), _calc_makespan_fast(processing_times, sequence)
    
    current_seq = list(sequence)
    current_makespan = _calc_makespan_fast(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    # Dynamic tabu list: stores (i, j) moves and their expiration times
    tabu_list = {}  # (i, j) -> step when it becomes non-tabu
    tabu_tenure_base = max(10, int(n_jobs * 0.5))
    tabu_tenure_max = max(20, int(n_jobs))
    
    step = 0
    no_improve_count = 0
    max_no_improve = 5000
    
    # Track recent best to prevent cycling
    recent_best_makespan = current_makespan
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        step += 1
        
        # Dynamic tabu tenure: increase when stuck, decrease when improving
        if no_improve_count > 1000:
            current_tenure = tabu_tenure_max
        elif no_improve_count > 500:
            current_tenure = int((tabu_tenure_base + tabu_tenure_max) / 2)
        else:
            current_tenure = tabu_tenure_base
        
        # Clean expired tabu entries
        expired_keys = [k for k, exp in tabu_list.items() if exp <= step]
        for k in expired_keys:
            del tabu_list[k]
        
        # Find best non-tabu move (or aspiration move)
        best_move = None
        best_move_makespan = float('inf')
        best_move_is_tabu = False
        
        # Sample a subset of moves for efficiency on large problems
        if n_jobs <= 50:
            i_range = range(n_jobs)
            j_range = range(n_jobs)
        else:
            # Sample positions for larger problems
            i_samples = random.sample(range(n_jobs), min(n_jobs, 20))
            j_samples = random.sample(range(n_jobs), min(n_jobs, 20))
            i_range = i_samples
            j_range = j_samples
        
        for i in i_range:
            if time.time() >= deadline:
                break
            for j in j_range:
                if j == i:
                    continue
                
                move_key = (i, j)
                is_tabu = move_key in tabu_list
                
                new_seq = _insertion_move(current_seq, i, j)
                new_makespan = _calc_makespan_fast(processing_times, new_seq)
                
                # Aspiration criterion: accept tabu move if it improves global best
                if is_tabu and new_makespan >= best_makespan:
                    continue
                
                if new_makespan < best_move_makespan:
                    best_move_makespan = new_makespan
                    best_move = (i, j)
                    best_move_is_tabu = is_tabu
        
        if best_move is None:
            # No valid move found, break
            break
        
        i, j = best_move
        
        # Apply the move
        current_seq = _insertion_move(current_seq, i, j)
        current_makespan = best_move_makespan
        
        # Add move to tabu list
        tabu_list[(i, j)] = step + current_tenure
        # Also add reverse move
        reverse_i, reverse_j = j, i
        if reverse_i != reverse_j:
            tabu_list[(reverse_i, reverse_j)] = step + current_tenure
        
        # Update best
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_seq = list(current_seq)
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Periodic diversification: if stuck too long, perturb the solution
        if no_improve_count > 2000 and n_jobs >= 4:
            # Apply a random 2-opt reversal to escape local optimum
            i, j = sorted(random.sample(range(n_jobs), 2))
            current_seq = current_seq[:i] + list(reversed(current_seq[i:j+1])) + current_seq[j+1:]
            current_makespan = _calc_makespan_fast(processing_times, current_seq)
            no_improve_count = 0
            # Clear tabu list to allow fresh exploration
            tabu_list.clear()
    
    return best_seq, best_makespan
=======
def _tabu_search(processing_times, sequence, deadline):
    """
    Optimized Tabu Search.
    """
    n_jobs = len(sequence)
    if n_jobs < 2:
        return list(sequence), _calc_makespan(processing_times, sequence)
    
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    best_seq = list(current_seq)
    best_makespan = current_makespan
    
    # Tabu list: (i, j) -> expiration step
    tabu_list = {}
    # Tenure based on problem size
    tabu_tenure = max(5, int(n_jobs * 0.3))
    
    step = 0
    no_improve_count = 0
    max_no_improve = 1000 # Reduced to allow more diversification attempts
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        step += 1
        
        # Clean expired tabu entries (simple check)
        if step % 10 == 0: # Clean every 10 steps to reduce overhead
            expired_keys = [k for k, exp in tabu_list.items() if exp <= step]
            for k in expired_keys:
                del tabu_list[k]
        
        best_move = None
        best_move_makespan = float('inf')
        
        # For efficiency, we might restrict the search space if n_jobs is large
        # But for now, we try all i, but maybe sample j?
        # Let's try all i, and for each i, all j.
        # To speed up, we break early if we find a move that is significantly better? No, we need best.
        
        # Optimization: Only check a subset of j if n_jobs > 100
        if n_jobs > 100:
            j_samples = random.sample(range(n_jobs), 50)
        else:
            j_samples = range(n_jobs)
            
        for i in range(n_jobs):
            if time.time() >= deadline:
                return best_seq, best_makespan
                
            job_to_move = current_seq[i]
            
            for j in j_samples:
                if j == i:
                    continue
                
                move_key = (i, j)
                is_tabu = move_key in tabu_list
                
                new_seq = _insertion_move(current_seq, i, j)
                new_makespan = _calc_makespan(processing_times, new_seq)
                
                # Aspiration
                if is_tabu and new_makespan >= best_makespan:
                    continue
                
                if new_makespan < best_move_makespan:
                    best_move_makespan = new_makespan
                    best_move = (i, j)
        
        if best_move is None:
            break
        
        i, j = best_move
        
        current_seq = _insertion_move(current_seq, i, j)
        current_makespan = best_move_makespan
        
        # Add to tabu
        tabu_list[(i, j)] = step + tabu_tenure
        # Reverse move
        # If we moved i to j, the reverse is moving the job at j (which is now the one we moved? No.)
        # Insertion move definition: Remove i, insert at j.
        # The inverse move is complex to define simply as (j, i).
        # We'll just add (i, j).
        
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_seq = list(current_seq)
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count > 500:
            # Diversification
            if n_jobs >= 4:
                i, j = sorted(random.sample(range(n_jobs), 2))
                current_seq = current_seq[:i] + list(reversed(current_seq[i:j+1])) + current_seq[j+1:]
                current_makespan = _calc_makespan(processing_times, current_seq)
                no_improve_count = 0
                tabu_list.clear()
            else:
                break
    
    return best_seq, best_makespan
>>>>>>> REPLACE

