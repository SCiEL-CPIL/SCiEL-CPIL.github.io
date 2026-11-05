import time
import random
import math

def construct(n_rows, k):
    """
    Constructs a set of n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct across all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave buffer for evaluator
    
    # The problem is to find a set of n_rows "difference sets" (each of size k(k+1)/2)
    # that are disjoint.
    # We use a hybrid approach:
    # 1. Initial construction using a greedy heuristic that minimizes "fragmentation" of the difference space.
    # 2. Local search (simulated annealing) to improve the scope.
    
    best_rows = None
    best_scope = float('inf')
    
    # --- Phase 1: Initial Construction ---
    # We try multiple seeds to find diverse initial solutions.
    # The greedy heuristic prefers differences that cluster with existing ones
    # to reduce the number of "gaps" in the difference set, which allows
    # the set to grow more compactly.
    
    max_initial_seeds = 100
    for seed in range(max_initial_seeds):
        if time.time() > deadline - 30:  # Reserve time for local search
            break
        
        rows = _greedy_construct(n_rows, k, seed)
        if rows is None:
            continue
        
        scope = _scope(rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in rows]
            
            # If we hit the theoretical lower bound, we can stop early
            # (though for (7,5) the bound is 105, and we might not reach it easily)
            if scope <= 105:
                break

    if best_rows is None:
        # Fallback: simple greedy if heuristic failed
        return _simple_greedy(n_rows, k)

    # --- Phase 2: Local Search / Simulated Annealing ---
    # We attempt to reduce the scope by modifying rows.
    # Moves:
    # 1. Rebuild a single row from scratch, respecting used diffs of other rows.
    # 2. Swap elements within a row (less effective, but cheap).
    
    # We focus on "Row Rebuilding" as it allows global optimization of that row.
    
    current_rows = [list(r) for r in best_rows]
    current_scope = best_scope
    
    # Simulated Annealing parameters
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    rng = random.Random(12345)
    
    while time.time() < deadline - 5:
        if T < T_min:
            break
            
        # Select a random row to rebuild
        row_idx = rng.randint(0, n_rows - 1)
        
        # Save current row
        old_row = current_rows[row_idx]
        old_diffs = _row_diffs(old_row)
        
        # Remove old diffs from global used set
        # For efficiency, we maintain a global set of used diffs.
        # But rebuilding requires checking against all other rows.
        # Let's maintain a global `used_diffs` set for the current state.
        
        # Actually, maintaining a global set is easier if we rebuild the whole thing or update carefully.
        # Let's maintain `all_diffs` set for the current `current_rows`.
        
        # We need to remove old_diffs from the global set.
        # But we don't have a global set stored outside this loop easily without re-computing.
        # Let's compute the set of diffs for all rows except row_idx.
        
        other_diffs = set()
        for i, r in enumerate(current_rows):
            if i != row_idx:
                other_diffs.update(_row_diffs(r))
        
        # Now, rebuild row_idx
        new_row = _rebuild_row(k, other_diffs, rng)
        
        if new_row is None:
            T *= cooling
            continue
            
        new_scope = max([r[-1] for r in current_rows[:row_idx] + [new_row] + current_rows[row_idx+1:]])
        
        # Acceptance criterion
        if new_scope < current_scope:
            current_rows[row_idx] = new_row
            current_scope = new_scope
            T = max(T * cooling, T_min) # Restart or keep cooling? Standard is keep cooling.
        else:
            delta = new_scope - current_scope
            # Accept worse solution with probability exp(-delta / T)
            if rng.random() < math.exp(-delta / T):
                current_rows[row_idx] = new_row
                current_scope = new_scope
            T *= cooling

        if current_scope < best_scope:
            best_scope = current_scope
            best_rows = [list(r) for r in current_rows]

    return best_rows

def _scope(rows):
    return max(r[-1] for r in rows)

def _row_diffs(row):
    diffs = set()
    for i in range(len(row)):
        for j in range(i+1, len(row)):
            diffs.add(row[j] - row[i])
    return diffs

def _greedy_construct(n_rows, k, seed):
    """
    Constructs rows one by one.
    For each row, we choose elements greedily.
    Heuristic: Minimize the number of "isolated" differences (differences not close to other used diffs).
    This encourages packing differences into dense clusters.
    """
    rng = random.Random(seed)
    used_diffs = set()
    rows = []
    
    for _ in range(n_rows):
        row = [0]
        for _ in range(k):
            last_val = row[-1]
            best_x = None
            best_score = float('inf')
            candidates = []
            
            # Search for x > last_val
            # Bound x reasonably
            upper_bound = 500 
            
            for x in range(last_val + 1, upper_bound + 1):
                new_diffs = {x - a for a in row}
                if new_diffs & used_diffs:
                    continue
                
                # Score: How well do these new diffs fit into the existing used_diffs?
                # We count how many of the new diffs are "isolated" (no neighbor in used_diffs within dist 3)
                # And we prefer smaller x.
                
                score = 0
                for d in new_diffs:
                    # Check if d is close to any used diff
                    # Optimization: used_diffs is a set, so O(1) lookup
                    # But checking neighbors requires iteration or sorted structure.
                    # For simplicity and speed, we just check a small window.
                    is_isolated = True
                    for delta in range(1, 4):
                        if (d - delta) in used_diffs or (d + delta) in used_diffs:
                            is_isolated = False
                            break
                    if is_isolated:
                        score += 10.0 # Heavy penalty for isolation
                
                # Add small penalty for x
                score += 0.1 * x
                
                candidates.append((score, x))
                if score < best_score:
                    best_score = score
                    best_x = x
                    
            if best_x is None:
                return None
                
            # Pick from top candidates to add diversity
            candidates.sort(key=lambda t: t[0])
            top_n = min(5, len(candidates))
            chosen = candidates[rng.randint(0, top_n - 1)][1]
            
            row.append(chosen)
            used_diffs.update({chosen - a for a in row[:-1]})
            
        rows.append(row)
        
    return rows

def _rebuild_row(k, forbidden_diffs, rng):
    """
    Rebuilds a single row of k+1 elements starting at 0,
    avoiding any difference in `forbidden_diffs`.
    Uses a randomized greedy approach.
    """
    row = [0]
    for _ in range(k):
        last_val = row[-1]
        candidates = []
        
        # Search window
        # We need to find x such that all x-a (a in row) are not in forbidden_diffs
        # and x > last_val
        
        # To keep it fast, we don't search too far.
        # If we fail, we return None.
        upper_bound = 200 
        
        for x in range(last_val + 1, upper_bound + 1):
            valid = True
            for a in row:
                if x - a in forbidden_diffs:
                    valid = False
                    break
            if valid:
                # Heuristic: prefer x that minimizes new "isolated" diffs?
                # Or just random?
                # Let's add a small bias towards smaller x to keep scope down,
                # but allow randomness.
                candidates.append(x)
                
                if len(candidates) > 20:
                    break
        
        if not candidates:
            return None
            
        # Pick randomly from candidates, weighted towards smaller x?
        # Simple random is fine for SA diversity.
        # But to help convergence, let's pick the smallest 50% and choose from there.
        # Or just random.
        x = rng.choice(candidates)
        row.append(x)
        
    return row

def _simple_greedy(n_rows, k):
    """
    Fallback simple greedy.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 1
        while len(row) < k + 1:
            new_diffs = {x - a for a in row}
            if not (new_diffs & used):
                row.append(x)
                used.update(new_diffs)
            x += 1
        rows.append(row)
    return rows
