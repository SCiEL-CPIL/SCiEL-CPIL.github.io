import time
import random
import math
import sys
from collections import defaultdict

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0,
    such that all positive differences within rows are distinct across all rows.
    Minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill

    def get_max_val(rows):
        if not rows:
            return 0
        return max(max(row) for row in rows)

    def is_valid(rows, k):
        """Check if all differences are distinct across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def row_diffs_set(row):
        """Compute the set of all positive differences in a row."""
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(row[j] - row[i])
        return diffs

    def compute_all_used(rows):
        """Compute the union of all differences across all rows."""
        all_used = set()
        for row in rows:
            all_used |= row_diffs_set(row)
        return all_used

    def greedy_row(used, k, randomize=False, bias=0, max_limit=None):
        """
        Construct a single row of k+1 elements starting at 0,
        given the used differences. Returns the row or None if impossible.
        """
        row = [0]
        x = 0
        for _ in range(k):
            found = False
            # Try increasing x
            while True:
                x += 1
                # Check if x is too large (safety)
                if max_limit is not None and x >= max_limit:
                    return None
                if x > 500:
                    return None
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used = used | new_diffs
                    found = True
                    break
                if randomize and bias > 0 and random.random() < bias:
                    # Skip some values to randomize
                    skip = random.randint(1, 3)
                    x += skip
        return row

    def randomized_greedy_row(forbidden_diffs, max_limit=None, max_candidates=10, safety_bound=500):
        """
        Construct a random valid row of length k+1 starting at 0, strictly increasing,
        with all internal differences distinct and disjoint from forbidden_diffs,
        with max <= max_limit if specified.
        """
        row = [0]
        used_in_row = set()
        last_val = 0
        
        for _ in range(k):
            candidates = []
            x = last_val + 1
            while len(candidates) < max_candidates:
                if max_limit is not None and x >= max_limit:
                    break
                if x > safety_bound:
                    break
                new_diffs = {x - a for a in row}
                # Check: new_diffs must not intersect forbidden_diffs
                # And new_diffs must not intersect used_in_row (already distinct within row)
                # And len(new_diffs) == len(row) ensures distinctness of new diffs among themselves
                if not (new_diffs & forbidden_diffs) and not (new_diffs & used_in_row) and len(new_diffs) == len(row):
                    candidates.append(x)
                x += 1
                # Safety break to avoid infinite loop if no candidates found early
                if x - last_val > 200:
                    break

            if not candidates:
                return None
            
            # Pick randomly from candidates, biased towards smaller values
            # Use inverse probability or just uniform for simplicity
            # Let's use a soft-max style selection: weight = 1 / (1 + index)
            weights = [1.0 / (i + 1) for i in range(len(candidates))]
            total_weight = sum(weights)
            r = random.random() * total_weight
            cumulative = 0
            chosen_idx = 0
            for i, w in enumerate(weights):
                cumulative += w
                if r <= cumulative:
                    chosen_idx = i
                    break
            
            chosen_x = candidates[chosen_idx]
            row.append(chosen_x)
            new_diffs = {chosen_x - a for a in row[:-1]}
            used_in_row |= new_diffs
            last_val = chosen_x
            
        return row

    def construct_solution(seed_val):
        """Construct a full solution using greedy with optional randomization."""
        random.seed(seed_val)
        used = set()
        rows = []
        for _ in range(n_rows):
            # Try with some randomization
            row = greedy_row(used, k, randomize=True, bias=0.3)
            if row is None:
                # Fallback to pure greedy
                row = greedy_row(used, k, randomize=False, bias=0)
                if row is None:
                    return None
            rows.append(row)
        return rows

    # Phase 1: Generate initial solutions
    best_solution = None
    best_max = float('inf')
    
    # First, try the pure greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = greedy_row(used, k, randomize=False, bias=0)
        if row is None:
            break
        rows.append(row)
    if len(rows) == n_rows:
        best_solution = [list(r) for r in rows]
        best_max = get_max_val(best_solution)
    
    # Phase 2: Try randomized constructions to find diverse starting points
    for seed_val in range(200):
        if time.time() > deadline:
            break
        rows = construct_solution(seed_val)
        if rows is not None and len(rows) == n_rows:
            curr_max = get_max_val(rows)
            if curr_max < best_max:
                best_max = curr_max
                best_solution = [list(r) for r in rows]
                if best_max <= 105:
                    break
    
    if best_solution is None:
        # Fallback: should not happen
        return [[0] + list(range(1, k+1)) for _ in range(n_rows)]

    # Phase 3: Simulated Annealing with row-level re-optimization
    # This is the core improvement: instead of just local improve, we use SA
    # to escape local minima by occasionally accepting worse solutions.
    
    current_rows = [list(r) for r in best_solution]
    all_diffs = [row_diffs_set(r) for r in current_rows]
    current_max = get_max_val(current_rows)
    
    T = 10.0
    cooling_rate = 0.995
    no_improve_count = 0
    improvement_threshold = 200  # steps without improvement before cooling adjustment
    
    while time.time() < deadline:
        if T < 0.01:
            # Restart with higher temperature if stuck for a long time
            if no_improve_count > improvement_threshold:
                T = 5.0
                no_improve_count = 0
            else:
                break
        
        i = random.randint(0, n_rows - 1)
        
        # Calculate forbidden diffs for row i
        forbidden = set()
        for j in range(n_rows):
            if j != i:
                forbidden |= all_diffs[j]
        
        # Try to find a better row
        # Strategy: First try to find a row with max < current_max
        # If current_max is the bottleneck, try to reduce it
        # Use max_limit = current_max - 1 to force improvement, or current_max for same-level
        max_limit = current_max
        new_row = randomized_greedy_row(forbidden, max_limit=max_limit, max_candidates=15)
        
        if new_row is None:
            # If we couldn't find one under current max, try to find any valid row
            # This might increase max, but could lead to better global structure later
            new_row = randomized_greedy_row(forbidden, max_limit=None, max_candidates=15)
            
        if new_row is None:
            continue
            
        new_max_row = new_row[-1]
        
        # Swap and compute new global max
        old_row = current_rows[i]
        current_rows[i] = new_row
        all_diffs[i] = row_diffs_set(new_row)
        
        new_global_max = get_max_val(current_rows)
        
        delta = new_global_max - current_max
        
        accept = False
        if delta < 0:
            accept = True
            no_improve_count = 0
        else:
            if T > 0:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    accept = True
        
        if accept:
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_solution = [list(r) for r in current_rows]
                if best_max <= 105:
                    break
        else:
            # Reject: revert
            current_rows[i] = old_row
            all_diffs[i] = row_diffs_set(old_row)
        
        T *= cooling_rate
        no_improve_count += 1
        
        # Occasionally, if stuck, try a more aggressive perturbation
        if no_improve_count % 500 == 0 and T > 1.0:
            # Perturb a random row significantly
            i = random.randint(0, n_rows - 1)
            forbidden = set()
            for j in range(n_rows):
                if j != i:
                    forbidden |= all_diffs[j]
            
            # Try a row with higher max to escape
            new_row = randomized_greedy_row(forbidden, max_limit=None, max_candidates=5)
            if new_row is not None:
                old_row = current_rows[i]
                current_rows[i] = new_row
                all_diffs[i] = row_diffs_set(new_row)
                current_max = get_max_val(current_rows)
                # Accept this perturbation if it's not too bad
                if current_max < best_max + 20:
                    no_improve_count = 0
                else:
                    # Revert if too bad
                    current_rows[i] = old_row
                    all_diffs[i] = row_diffs_set(old_row)
                    current_max = get_max_val(current_rows)

    # Phase 4: Final polish - try to reduce max further with greedy reconstruction
    if best_solution is not None and time.time() < deadline - 10:
        improved = [list(r) for r in best_solution]
        improved_max = get_max_val(improved)
        
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            
            # Find the row with the largest max value
            max_idx = max(range(len(improved)), key=lambda i: max(improved[i]))
            max_val = max(improved[max_idx])
            
            if max_val <= 105:
                break
            
            # Remove this row's differences from used
            used = set()
            for i, row in enumerate(improved):
                if i != max_idx:
                    used |= row_diffs_set(row)
            
            # Try to reconstruct this row with a lower max
            best_row = None
            for attempt in range(20):
                row = randomized_greedy_row(used, max_limit=max_val, max_candidates=10)
                if row is not None:
                    new_max = max(row)
                    if best_row is None or new_max < max(best_row):
                        best_row = row
                        if new_max < max_val:
                            break
            
            if best_row is not None and max(best_row) < max_val:
                improved[max_idx] = best_row
                new_improved_max = get_max_val(improved)
                if new_improved_max < improved_max:
                    improved_max = new_improved_max
                    if improved_max < best_max:
                        best_max = improved_max
                        best_solution = [list(r) for r in improved]
                        if best_max <= 105:
                            break
            else:
                # No improvement found for this row, try another
                pass
    
    return best_solution
