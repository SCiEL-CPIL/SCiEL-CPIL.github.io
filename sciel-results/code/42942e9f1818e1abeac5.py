"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
N = 256

def _has_4ap_fast(selected_bool, x):
    """Check if adding x to selected_bool creates a 4-AP. selected_bool is a boolean array."""
    max_d1 = (N - x) // 3
    for d in range(1, max_d1 + 1):
        if selected_bool[x+d] and selected_bool[x+2*d] and selected_bool[x+3*d]:
            return True
    max_d2 = min(x - 1, (N - x) // 2)
    for d in range(1, max_d2 + 1):
        a = x - d
        if selected_bool[a] and selected_bool[x+d] and selected_bool[x+2*d]:
            return True
    max_d3 = min((x - 1) // 2, N - x)
    for d in range(1, max_d3 + 1):
        a = x - 2*d
        if selected_bool[a] and selected_bool[x-d] and selected_bool[x+d]:
            return True
    max_d4 = (x - 1) // 3
    for d in range(1, max_d4 + 1):
        a = x - 3*d
        if selected_bool[a] and selected_bool[x-2*d] and selected_bool[x-d]:
            return True
    return False

def _count_3aps_fast(selected_bool):
    """Count 3-APs using boolean array."""
    count = 0
    for a in range(1, N+1):
        if not selected_bool[a]:
            continue
        for d in range(1, (N - a) // 2 + 1):
            if selected_bool[a+d] and selected_bool[a+2*d]:
                count += 1
    return count

def _count_3aps_delta(selected_bool, x):
    """Count 3-APs that involve x (as first, middle, or last term)."""
    delta = 0
    max_d1 = (N - x) // 2
    for d in range(1, max_d1 + 1):
        if selected_bool[x+d] and selected_bool[x+2*d]:
            delta += 1
    max_d2 = min(x - 1, N - x)
    for d in range(1, max_d2 + 1):
        if selected_bool[x-d] and selected_bool[x+d]:
            delta += 1
    max_d3 = (x - 1) // 2
    for d in range(1, max_d3 + 1):
        if selected_bool[x-2*d] and selected_bool[x-d]:
            delta += 1
    return delta

def _grasp_construct(rng):
    """GRASP randomized greedy construction."""
    selected = np.zeros(N+1, dtype=bool)
    # Randomize the order of candidates slightly
    order = list(range(1, N+1))
    rng.shuffle(order)
    for x in order:
        if not _has_4ap_fast(selected, x):
            selected[x] = True
    return selected

def _tabu_search(selected, current_score, deadline, rng):
    """Tabu search with add/remove/swap moves."""
    tabu_list = {}  # x -> iteration when it becomes non-tabu
    tabu_tenure = 8
    best_selected = selected.copy()
    best_score = current_score
    iter_count = 0
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve < max_no_improve:
        iter_count += 1
        # Build candidate moves
        moves = []
        
        # Add moves
        for x in range(1, N+1):
            if not selected[x]:
                if x in tabu_list and tabu_list[x] > iter_count:
                    continue
                if not _has_4ap_fast(selected, x):
                    delta = _count_3aps_delta(selected, x)
                    moves.append(('add', x, delta))
        
        # Remove moves
        for x in range(1, N+1):
            if selected[x]:
                if x in tabu_list and tabu_list[x] > iter_count:
                    continue
                delta = -_count_3aps_delta(selected, x)
                moves.append(('remove', x, delta))
        
        # Swap moves
        sel_list = [x for x in range(1, N+1) if selected[x]]
        unsel_list = [x for x in range(1, N+1) if not selected[x]]
        if sel_list and unsel_list:
            # Limit swap candidates for speed
            n_swap = min(len(sel_list), len(unsel_list), 20)
            sel_sample = rng.sample(sel_list, n_swap) if len(sel_list) > n_swap else sel_list
            unsel_sample = rng.sample(unsel_list, n_swap) if len(unsel_list) > n_swap else unsel_list
            for x_out in sel_sample:
                for x_in in unsel_sample:
                    if x_out in tabu_list and tabu_list[x_out] > iter_count:
                        continue
                    if x_in in tabu_list and tabu_list[x_in] > iter_count:
                        continue
                    # Check validity
                    selected[x_out] = False
                    if not _has_4ap_fast(selected, x_in):
                        delta = -_count_3aps_delta(selected, x_out) + _count_3aps_delta(selected, x_in)
                        moves.append(('swap', (x_out, x_in), delta))
                    selected[x_out] = True
        
        if not moves:
            break
        
        # Select best move (or best non-tabu)
        moves.sort(key=lambda m: m[2], reverse=True)
        best_move = moves[0]
        mtype, mdata, mdelta = best_move
        
        # Apply move
        if mtype == 'add':
            x = mdata
            selected[x] = True
            current_score += mdelta
            tabu_list[x] = iter_count + tabu_tenure
        elif mtype == 'remove':
            x = mdata
            selected[x] = False
            current_score += mdelta
            tabu_list[x] = iter_count + tabu_tenure
        elif mtype == 'swap':
            x_out, x_in = mdata
            selected[x_out] = False
            selected[x_in] = True
            current_score += mdelta
            tabu_list[x_out] = iter_count + tabu_tenure
            tabu_list[x_in] = iter_count + tabu_tenure
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_selected = selected.copy()
            no_improve = 0
        else:
            no_improve += 1
    
    return best_selected, best_score

def _local_search_add(selected):
    """Greedy add: add any element that doesn't create a 4-AP."""
    changed = True
    while changed:
        changed = False
        for x in range(1, N+1):
            if not selected[x]:
                if not _has_4ap_fast(selected, x):
                    selected[x] = True
                    changed = True
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline with 5s margin
    
    rng = random.Random(42)
    
    best_selected = None
    best_score = -1
    
    # Multiple GRASP restarts with tabu search
    n_restarts = 20
    time_per_restart = (deadline - start_time) / n_restarts
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # GRASP construction
        selected = _grasp_construct(rng)
        current_score = _count_3aps_fast(selected)
        
        # Local search: add all possible elements
        selected = _local_search_add(selected)
        current_score = _count_3aps_fast(selected)
        
        # Tabu search
        tabu_deadline = min(deadline - 5, time.time() + time_per_restart * 0.7)
        selected, current_score = _tabu_search(selected, current_score, tabu_deadline, rng)
        
        # Final local search
        selected = _local_search_add(selected)
        current_score = _count_3aps_fast(selected)
        
        if current_score > best_score:
            best_score = current_score
            best_selected = selected.copy()
    
    # Convert to sorted list
    result = [x for x in range(1, N+1) if best_selected[x]]
    return sorted(result)
# EVOLVE-BLOCK-END
