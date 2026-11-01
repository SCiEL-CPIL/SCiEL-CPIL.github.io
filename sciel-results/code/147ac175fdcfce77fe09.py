import time
import random
import math

# EVOLVE-BLOCK-START

def _is_valid(rows):
    """Check if the set of rows is valid (all diffs distinct globally)."""
    used = set()
    for row in rows:
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                d = row[j] - row[i]
                if d in diffs:
                    return False
                diffs.add(d)
        if diffs & used:
            return False
        used |= diffs
    return True

def _greedy_init(n_rows, k):
    """
    Greedy initialization: each row takes the smallest next mark with no difference collision.
    """
    rows = []
    used = set()
    for r in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if len(new_diffs) != len(row):
                continue
            if new_diffs & used:
                continue
            row.append(x)
            used |= new_diffs
        rows.append(row)
    return rows

def _row_diffs(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _generate_sidon_sets(k, max_entry):
    """
    Generate all (k+1)-term Sidon sets (distinct internal differences)
    starting at 0 with max entry <= max_entry.
    Returns list of tuples, each tuple is a row.
    """
    sets = []
    
    def backtrack(pos, current, current_diffs):
        if pos == k + 1:
            sets.append(tuple(current))
            return
        
        if pos == 0:
            backtrack(1, [0], set())
            return
        
        prev = current[-1]
        min_next = prev + 1
        max_next = max_entry
        
        for x in range(min_next, max_next + 1):
            new_diffs = {x - a for a in current}
            if len(new_diffs) != len(current):
                continue
            if new_diffs & current_diffs:
                continue
            current.append(x)
            backtrack(pos + 1, current, current_diffs | new_diffs)
            current.pop()
    
    backtrack(0, [], set())
    return sets

def _build_conflict_graph(sidon_sets, k):
    """
    Build conflict graph: two sets conflict if they share any difference.
    Returns adjacency as list of sets.
    """
    n = len(sidon_sets)
    # Precompute diff sets for each sidon set
    diff_sets = [_row_diffs(list(s)) for s in sidon_sets]
    
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if diff_sets[i] & diff_sets[j]:
                adj[i].add(j)
                adj[j].add(i)
    
    return adj, diff_sets

def _max_independent_set_greedy(adj, n, time_limit=5.0):
    """
    Greedy + local search for maximum independent set.
    Returns the independent set as a list of indices.
    """
    start = time.time()
    deadline = start + time_limit
    
    # Greedy: pick vertices in order of increasing degree
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: degrees[i])
    
    selected = set()
    for v in order:
        if all(u not in selected for u in adj[v]):
            selected.add(v)
    
    # Local search: try to add vertices, remove conflicting ones
    best_set = set(selected)
    best_size = len(best_set)
    
    for restart in range(20):
        if time.time() > deadline:
            break
        
        current_set = set(best_set)
        
        # Randomly perturb
        for _ in range(5):
            if current_set:
                v = random.choice(list(current_set))
                current_set.discard(v)
            elif n > 0:
                v = random.randrange(n)
                current_set.add(v)
        
        # Try to add vertices
        improved = True
        while improved:
            improved = False
            candidates = [v for v in range(n) if v not in current_set]
            random.shuffle(candidates)
            for v in candidates:
                if all(u not in current_set for u in adj[v]):
                    current_set.add(v)
                    improved = True
        
        # Try to swap: remove one, add multiple
        if current_set:
            for v in list(current_set):
                if time.time() > deadline:
                    break
                trial = set(current_set)
                trial.discard(v)
                # Try to add as many as possible
                candidates = [u for u in range(n) if u not in trial]
                random.shuffle(candidates)
                for u in candidates:
                    if all(w not in trial for w in adj[u]):
                        trial.add(u)
                if len(trial) > len(current_set):
                    current_set = trial
                    improved = True
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
    
    return list(best_set)

def _local_search_mis(adj, n, initial_set, time_limit=10.0):
    """
    More sophisticated local search for MIS.
    Uses simulated annealing on the independent set.
    """
    start = time.time()
    deadline = start + time_limit
    
    current_set = set(initial_set)
    best_set = set(current_set)
    best_size = len(current_set)
    
    T = 1.0
    cool = 0.999
    
    while time.time() < deadline and T > 0.01:
        if not current_set and n > 0:
            v = random.randrange(n)
            current_set.add(v)
            T = max(T, 0.5)
            continue
        
        # Choose operation: add, remove, or swap
        op = random.random()
        
        if op < 0.4:
            # Add a vertex
            candidates = [v for v in range(n) if v not in current_set]
            if not candidates:
                T = max(0.01, T * cool)
                continue
            v = random.choice(candidates)
            conflicts = adj[v] & current_set
            if not conflicts:
                current_set.add(v)
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = set(current_set)
            else:
                # Accept with probability based on temperature
                delta = 1 - len(conflicts)  # net change in size
                if delta >= 0 or random.random() < math.exp(delta / T):
                    current_set.add(v)
                    current_set -= conflicts
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
        
        elif op < 0.7:
            # Remove a vertex
            if current_set:
                v = random.choice(list(current_set))
                current_set.discard(v)
        
        else:
            # Swap: remove one, try to add another
            if current_set:
                v = random.choice(list(current_set))
                current_set.discard(v)
                candidates = [u for u in range(n) if u not in current_set and u != v]
                if candidates:
                    u = random.choice(candidates)
                    if all(w not in current_set for w in adj[u]):
                        current_set.add(u)
        
        T = max(0.01, T * cool)
    
    return list(best_set)

def _validate_move_fast(cand_row, old_diffs, all_used, pi, old_val, new_val):
    """
    Fast validation for a single element change at index pi.
    Returns (new_diffs, is_valid).
    """
    removed = set()
    added = set()
    
    # Diffs with elements before pi
    for i in range(pi):
        removed.add(old_val - cand_row[i])
        added.add(new_val - cand_row[i])
        
    # Diffs with elements after pi
    for j in range(pi + 1, len(cand_row)):
        removed.add(cand_row[j] - old_val)
        added.add(cand_row[j] - new_val)
        
    # Check for duplicates in added (internal conflict)
    if len(added) < len(removed):
        return None, False
        
    # Check internal conflict: added vs remaining old diffs
    remaining_old = old_diffs - removed
    if added & remaining_old:
        return None, False
        
    # Check cross-row conflict: added vs diffs in other rows
    other_rows_diffs = all_used - old_diffs
    if added & other_rows_diffs:
        return None, False
        
    new_diffs = remaining_old | added
    return new_diffs, True

def _simulated_annealing(rows, n_rows, k, time_budget, T_init=100.0, T_min=0.1, cool=0.99995):
    """
    Simulated annealing to minimize the global max entry.
    Returns improved rows and the best value found.
    """
    current_rows = [r[:] for r in rows]
    current_val = _global_max(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_val = current_val
    
    row_maxes = [max(r) for r in current_rows]
    all_used = _all_used(current_rows)
    row_diffs = [_row_diffs(r) for r in current_rows]
    
    start_time = time.time()
    deadline = start_time + time_budget
    T = T_init
    no_improve = 0
    max_no_improve = 15000
    restarts = 0
    max_restarts = 10
    
    while time.time() < deadline and no_improve < max_no_improve:
        ri = random.randrange(n_rows)
        pi = random.randrange(1, k + 1)
        
        old_val = current_rows[ri][pi]
        min_allowed = current_rows[ri][pi - 1] + 1
        
        # Build candidate list
        candidates = []
        
        # Try decreasing values (most important for minimization)
        max_decrease = min(50, old_val - min_allowed)
        for d in range(1, max_decrease + 1):
            candidates.append(old_val - d)
        
        # Try some increasing values for exploration
        if pi < k:
            max_allowed = current_rows[ri][pi + 1] - 1
        else:
            max_allowed = current_val + 20
        max_increase = min(15, max_allowed - old_val)
        for d in range(1, max_increase + 1):
            candidates.append(old_val + d)
        
        # Random jump
        if random.random() < 0.2:
            low = min_allowed
            high = max_allowed if pi < k else current_val + 20
            if low <= high:
                candidates.append(random.randint(low, high))
        
        if not candidates:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
        
        random.shuffle(candidates)
        
        moved = False
        for new_val in candidates:
            if new_val <= current_rows[ri][pi - 1]:
                continue
            if pi < k and new_val >= current_rows[ri][pi + 1]:
                continue
            
            cand_row = current_rows[ri][:]
            cand_row[pi] = new_val
            
            new_diffs, valid = _validate_move_fast(cand_row, row_diffs[ri], all_used, pi, old_val, new_val)
            if not valid:
                continue
            
            cand_row_max = max(cand_row)
            old_row_max = row_maxes[ri]
            row_maxes[ri] = cand_row_max
            new_global_max = max(row_maxes)
            row_maxes[ri] = old_row_max
            
            delta = new_global_max - current_val
            
            accept = False
            if delta < 0:
                accept = True
            elif delta == 0:
                if random.random() < 0.1:
                    accept = True
            else:
                if T > T_min:
                    prob = math.exp(-delta / max(T, 0.1))
                    if random.random() < prob:
                        accept = True
            
            if accept:
                all_used -= row_diffs[ri]
                all_used |= new_diffs
                current_rows[ri] = cand_row
                row_diffs[ri] = new_diffs
                row_maxes[ri] = cand_row_max
                current_val = max(row_maxes)
                
                if current_val < best_val:
                    best_val = current_val
                    best_rows = [r[:] for r in current_rows]
                    no_improve = 0
                else:
                    no_improve += 1
                
                moved = True
                break
        
        if not moved:
            no_improve += 1
        
        T = max(T_min, T * cool)
        
        if no_improve >= max_no_improve:
            restarts += 1
            if restarts >= max_restarts:
                break
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            all_used = _all_used(current_rows)
            row_diffs = [_row_diffs(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = T_init * 0.5
            no_improve = 0
    
    return best_rows, best_val

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Greedy initialization
    greedy_rows = _greedy_init(n_rows, k)
    best_rows = [r[:] for r in greedy_rows]
    best_val = _global_max(greedy_rows)
    
    # 2. Generate Sidon sets and build conflict graph
    # Start with a reasonable max_entry estimate
    initial_max = best_val
    sidon_sets = _generate_sidon_sets(k, initial_max)
    
    if len(sidon_sets) >= n_rows:
        adj, diff_sets = _build_conflict_graph(sidon_sets, k)
        n_sets = len(sidon_sets)
        
        # 3. Find maximum independent set
        mis_budget = min(30.0, deadline - time.time() - 20.0)
        if mis_budget > 5.0:
            greedy_mis = _max_independent_set_greedy(adj, n_sets, time_limit=mis_budget * 0.3)
            
            if len(greedy_mis) >= n_rows:
                # Use first n_rows sets from the MIS
                selected_indices = greedy_mis[:n_rows]
                mis_rows = [list(sidon_sets[i]) for i in selected_indices]
                mis_val = _global_max(mis_rows)
                
                if mis_val < best_val:
                    best_val = mis_val
                    best_rows = [r[:] for r in mis_rows]
                
                # Refine with local search on MIS
                ls_budget = min(20.0, deadline - time.time() - 15.0)
                if ls_budget > 5.0:
                    refined_mis = _local_search_mis(adj, n_sets, greedy_mis, time_limit=ls_budget)
                    if len(refined_mis) >= n_rows:
                        # Try to find a better subset of size n_rows
                        # Sort by max entry and take first n_rows
                        refined_sorted = sorted(refined_mis, key=lambda i: sidon_sets[i][-1])
                        selected = refined_sorted[:n_rows]
                        refined_rows = [list(sidon_sets[i]) for i in selected]
                        refined_val = _global_max(refined_rows)
                        
                        if refined_val < best_val:
                            best_val = refined_val
                            best_rows = [r[:] for r in refined_rows]
    
    # 4. Simulated annealing - main optimization
    sa_budget = deadline - time.time() - 5.0
    if sa_budget > 10.0:
        # Run SA in multiple phases
        phase1_budget = min(sa_budget * 0.6, 60.0)
        phase2_budget = sa_budget - phase1_budget - 2.0
        
        if phase1_budget > 5.0:
            best_rows, best_val = _simulated_annealing(
                best_rows, n_rows, k, phase1_budget,
                T_init=200.0, T_min=1.0, cool=0.9999
            )
        
        if time.time() < deadline - 5.0:
            phase2_budget = min(deadline - time.time() - 5.0, phase2_budget)
            if phase2_budget > 5.0:
                best_rows, best_val = _simulated_annealing(
                    best_rows, n_rows, k, phase2_budget,
                    T_init=50.0, T_min=0.1, cool=0.99995
                )
    
    # 5. Final polish with very low temperature
    if time.time() < deadline - 2.0:
        polish_budget = min(deadline - time.time() - 2.0, 10.0)
        if polish_budget > 2.0:
            best_rows, best_val = _simulated_annealing(
                best_rows, n_rows, k, polish_budget,
                T_init=5.0, T_min=0.05, cool=0.99999
            )
    
    # Final validity check
    if not _is_valid(best_rows):
        if _is_valid(greedy_rows):
            return greedy_rows
        # Fallback
        fallback_row = [0, 1, 3, 7, 15, 31]
        return [fallback_row[:] for _ in range(n_rows)]
    
    return best_rows

# EVOLVE-BLOCK-END
