"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 60s limit
    
    n = 256
    max_sum = 2 * n  # 512
    
    # Precompute squares up to max_sum
    squares = set()
    for i in range(1, int(math.isqrt(max_sum)) + 2):
        squares.add(i * i)
    
    # First, filter out numbers where 2*a is a square (diagonal constraint)
    valid_numbers = [a for a in range(1, n + 1) if 2 * a not in squares]
    
    num_valid = len(valid_numbers)
    val_to_idx = {v: i for i, v in enumerate(valid_numbers)}
    
    # Build conflict sets: which valid numbers conflict with which
    # a and b conflict if a + b is a perfect square (a != b)
    conflict = [set() for _ in range(num_valid)]
    for i in range(num_valid):
        a = valid_numbers[i]
        for s in squares:
            b = s - a
            if b != a and b in val_to_idx:
                conflict[i].add(val_to_idx[b])
    
    # Also add reverse conflicts (make symmetric)
    for i in range(num_valid):
        for j in conflict[i]:
            conflict[j].add(i)
    
    # Precompute conflict bitmasks for fast operations
    adj_bits = []
    for i in range(num_valid):
        bits = 0
        for j in conflict[i]:
            bits |= (1 << j)
        adj_bits.append(bits)
    
    def is_valid_set(selected_set):
        """Verify a set is valid (no pair sums to a square, including diagonals)."""
        for a in selected_set:
            if 2 * a in squares:
                return False
            for b in selected_set:
                if b >= a:
                    if a + b in squares:
                        return False
        return True
    
    def greedy_min_degree():
        """Greedy: repeatedly pick the vertex with minimum degree in remaining graph."""
        remaining = set(range(num_valid))
        selected = []
        while remaining:
            min_deg = float('inf')
            best_idx = -1
            for idx in remaining:
                deg = len(conflict[idx] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    best_idx = idx
            selected.append(best_idx)
            remaining.discard(best_idx)
            for neighbor in conflict[best_idx]:
                remaining.discard(neighbor)
        return selected
    
    def greedy_max_degree():
        """Greedy: repeatedly pick the vertex with maximum degree in remaining graph."""
        remaining = set(range(num_valid))
        selected = []
        while remaining:
            max_deg = -1
            best_idx = -1
            for idx in remaining:
                deg = len(conflict[idx] & remaining)
                if deg > max_deg:
                    max_deg = deg
                    best_idx = idx
            selected.append(best_idx)
            remaining.discard(best_idx)
            for neighbor in conflict[best_idx]:
                remaining.discard(neighbor)
        return selected
    
    def greedy_random():
        """Greedy with random tie-breaking."""
        remaining = set(range(num_valid))
        selected = []
        while remaining:
            min_deg = float('inf')
            candidates = []
            for idx in remaining:
                deg = len(conflict[idx] & remaining)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [idx]
                elif deg == min_deg:
                    candidates.append(idx)
            best_idx = random.choice(candidates)
            selected.append(best_idx)
            remaining.discard(best_idx)
            for neighbor in conflict[best_idx]:
                remaining.discard(neighbor)
        return selected
    
    def local_search(initial_selected, time_limit):
        """
        Local search with add/remove/swap moves.
        State: a set of selected vertices (indices into valid_numbers).
        Moves:
          - Add: add a vertex that doesn't conflict with any selected vertex
          - Remove: remove a vertex
          - Swap: remove one vertex, add another (net zero or +1 change)
        Objective: maximize |selected| while maintaining validity.
        """
        selected = set(initial_selected)
        best_selected = set(selected)
        best_size = len(selected)
        
        # Track which vertices are NOT in selected but could potentially be added
        # A vertex can be added if it has no conflicts with any vertex in selected
        def can_add(v, sel):
            """Check if v can be added to sel without creating conflicts."""
            return (adj_bits[v] & sum((1 << u) for u in sel)) == 0
        
        # Faster: use bitmask of selected
        sel_mask = 0
        for v in selected:
            sel_mask |= (1 << v)
        
        def can_add_fast(v, mask):
            return (adj_bits[v] & mask) == 0
        
        def get_addable(mask):
            """Get all vertices not in mask that can be added."""
            addable = []
            for v in range(num_valid):
                if not (mask & (1 << v)):
                    if (adj_bits[v] & mask) == 0:
                        addable.append(v)
            return addable
        
        # Simulated annealing style
        temperature = 10.0
        cooling = 0.999
        min_temp = 0.01
        
        while temperature > min_temp:
            if time.time() > time_limit:
                break
            
            # Try a random move
            move_type = random.random()
            
            if move_type < 0.4:
                # Add move: try to add a random addable vertex
                addable = get_addable(sel_mask)
                if addable:
                    v = random.choice(addable)
                    new_selected = selected | {v}
                    new_mask = sel_mask | (1 << v)
                    # This is always valid by construction
                    selected = new_selected
                    sel_mask = new_mask
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
            
            elif move_type < 0.6:
                # Remove move: remove a random vertex
                if selected:
                    v = random.choice(list(selected))
                    new_selected = selected - {v}
                    new_mask = sel_mask & ~(1 << v)
                    # Always valid (removing can't create conflicts)
                    # Accept with probability based on whether it helps future additions
                    # For now, accept if it allows more additions
                    addable_before = len(get_addable(sel_mask))
                    addable_after = len(get_addable(new_mask))
                    if addable_after >= addable_before or random.random() < 0.3:
                        selected = new_selected
                        sel_mask = new_mask
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
            
            elif move_type < 0.8:
                # Swap move: remove one, try to add one or two
                if selected:
                    v_remove = random.choice(list(selected))
                    temp_mask = sel_mask & ~(1 << v_remove)
                    addable = get_addable(temp_mask)
                    if addable:
                        # Try adding 1 or 2 vertices
                        if len(addable) >= 2 and random.random() < 0.3:
                            v1, v2 = random.sample(addable, 2)
                            # Check v1 and v2 don't conflict with each other
                            if (adj_bits[v1] & (1 << v2)) == 0:
                                new_mask = temp_mask | (1 << v1) | (1 << v2)
                                new_selected = (selected - {v_remove}) | {v1, v2}
                                selected = new_selected
                                sel_mask = new_mask
                                if len(selected) > best_size:
                                    best_size = len(selected)
                                    best_selected = set(selected)
                        else:
                            v_add = random.choice(addable)
                            new_mask = temp_mask | (1 << v_add)
                            new_selected = (selected - {v_remove}) | {v_add}
                            selected = new_selected
                            sel_mask = new_mask
                            if len(selected) > best_size:
                                best_size = len(selected)
                                best_selected = set(selected)
            
            else:
                # Perturbation: remove a random subset and re-greedy fill
                if len(selected) > 2:
                    k = min(3, len(selected))
                    to_remove = random.sample(list(selected), k)
                    new_selected = selected - set(to_remove)
                    new_mask = sel_mask
                    for v in to_remove:
                        new_mask &= ~(1 << v)
                    
                    # Greedy fill
                    while True:
                        addable = get_addable(new_mask)
                        if not addable:
                            break
                        # Pick min-degree addable
                        best_v = -1
                        best_deg = float('inf')
                        for v in addable:
                            deg = bin(adj_bits[v] & new_mask).count('1')
                            if deg < best_deg:
                                best_deg = deg
                                best_v = v
                        if best_v == -1:
                            break
                        new_selected.add(best_v)
                        new_mask |= (1 << best_v)
                    
                    selected = new_selected
                    sel_mask = new_mask
                    if len(selected) > best_size:
                        best_size = len(selected)
                        best_selected = set(selected)
            
            temperature *= cooling
        
        return best_selected
    
    # Generate initial solutions using different heuristics
    initial_solutions = []
    
    # Deterministic min-degree greedy
    initial_solutions.append(greedy_min_degree())
    
    # Deterministic max-degree greedy  
    initial_solutions.append(greedy_max_degree())
    
    # Multiple random greedy restarts
    for _ in range(10):
        initial_solutions.append(greedy_random())
    
    # Also try a simple sequential greedy
    def greedy_sequential():
        selected = []
        for i in range(num_valid):
            # Check if i can be added
            conflict_found = False
            for j in selected:
                if i in conflict[j]:
                    conflict_found = True
                    break
            if not conflict_found:
                selected.append(i)
        return selected
    
    initial_solutions.append(greedy_sequential())
    
    # Find the best initial solution
    best_initial = max(initial_solutions, key=len)
    best_size = len(best_initial)
    
    # Run local search from the best initial solution and from several others
    # with time budget split among them
    num_ls_runs = 5
    time_per_run = (deadline - (time.time() - start_time)) / num_ls_runs
    time_per_run = max(time_per_run, 1.0)
    
    # Prioritize: run LS from the best initial first, then from diverse others
    ls_order = [best_initial]
    # Add diverse initial solutions
    seen_sizes = {len(best_initial)}
    for sol in initial_solutions:
        if len(sol) not in seen_sizes:
            ls_order.append(sol)
            seen_sizes.add(len(sol))
        if len(ls_order) >= num_ls_runs:
            break
    
    # Pad with random restarts if needed
    while len(ls_order) < num_ls_runs:
        ls_order.append(greedy_random())
    
    for i, init_sol in enumerate(ls_order):
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 1.0:
            break
        
        # Allocate time: more time for earlier runs
        if i == 0:
            t_limit = start_time + deadline * 0.4
        else:
            t_limit = start_time + deadline * (0.4 + 0.6 * i / (num_ls_runs - 1)) if num_ls_runs > 1 else deadline
        
        result = local_search(set(init_sol), t_limit)
        
        if len(result) > best_size:
            best_size = len(result)
            best_initial = list(result)
    
    # Final verification and conversion back to values
    if best_initial:
        best_values = [valid_numbers[i] for i in best_initial]
        # Verify validity
        best_set = set(best_values)
        if not is_valid_set(best_set):
            # Fallback: rebuild from scratch with simple greedy
            fallback = greedy_min_degree()
            best_values = [valid_numbers[i] for i in fallback]
        return sorted(best_values)
    else:
        # Ultimate fallback
        return [valid_numbers[0]]
# EVOLVE-BLOCK-END
