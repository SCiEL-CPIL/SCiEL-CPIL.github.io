import time
import random
import sys
from functools import lru_cache

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs
    
    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 2. Simulated annealing to reduce scope
    best_solution, best_scope = _simulated_annealing(
        best_solution, n_rows, k, deadline
    )
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 3. Local improvement to reduce scope
    improved = _improve_scope(best_solution, n_rows, k, deadline)
    new_scope = max(max(row) for row in improved)
    if new_scope < best_scope:
        best_solution = improved
        best_scope = new_scope
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 4. Try backtracking for smaller scopes if time remains
    if time.time() < deadline - 15.0:
        attempt = 0
        while time.time() < deadline - 10.0:
            target_scope = best_scope - 1
            if target_scope < lower_bound:
                break
            
            found = False
            for _ in range(15):
                if time.time() > deadline - 10.0:
                    break
                
                solution = _backtracking_search(n_rows, k, target_scope, deadline, attempt)
                if solution:
                    new_scope = max(max(row) for row in solution)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = solution
                        found = True
                        if best_scope <= lower_bound:
                            return best_solution
                        break
                attempt += 1
            
            if not found:
                break
    
    # 5. Final polish
    best_solution = _polish_scope(best_solution, n_rows, k, deadline)
    
    return best_solution


def _simulated_annealing(initial_solution, n_rows, k, deadline):
    """
    Simulated annealing on row elements.
    State: list of rows (each row is list of k+1 increasing ints starting at 0)
    Energy: scope + penalty * (number of duplicate differences)
    Moves: shift one element in a row up or down while maintaining strict increase
    """
    k_plus_1 = k + 1
    rng = random.Random(42)
    
    # Copy initial solution
    current = [list(row) for row in initial_solution]
    best = [list(row) for row in initial_solution]
    
    def compute_energy(rows):
        """Compute energy = scope + penalty * num_conflicts"""
        scope = max(max(row) for row in rows)
        # Count duplicate differences
        diff_count = {}
        for row in rows:
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    d = row[j] - row[i]
                    diff_count[d] = diff_count.get(d, 0) + 1
        num_conflicts = sum(c - 1 for c in diff_count.values() if c > 1)
        penalty = 10000
        return scope + penalty * num_conflicts, num_conflicts
    
    def is_valid(rows):
        """Check if all differences are distinct and rows are valid"""
        used = set()
        for row in rows:
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    current_energy, _ = compute_energy(current)
    best_energy, _ = compute_energy(best)
    
    # Annealing schedule
    T_start = 50.0
    T_end = 0.01
    cooling_rate = 0.995
    
    T = T_start
    no_improve_count = 0
    
    while time.time() < deadline - 5.0 and T > T_end:
        # Generate a move: pick a row and position, shift by delta
        row_idx = rng.randrange(n_rows)
        # Don't shift position 0 (always 0)
        pos = rng.randrange(1, k_plus_1)
        delta = rng.choice([-2, -1, 1, 2, 3])
        
        old_val = current[row_idx][pos]
        new_val = old_val + delta
        
        # Check bounds: must maintain strict increase
        if pos > 0 and new_val <= current[row_idx][pos - 1]:
            T *= cooling_rate
            continue
        if pos < k_plus_1 - 1 and new_val >= current[row_idx][pos + 1]:
            T *= cooling_rate
            continue
        if new_val <= 0:
            T *= cooling_rate
            continue
        
        # Apply move
        current[row_idx][pos] = new_val
        
        new_energy, _ = compute_energy(current)
        delta_E = new_energy - current_energy
        
        # Accept with probability exp(-delta_E / T) if worsening
        if delta_E <= 0 or (T > 0 and rng.random() < pow(2.718281828, -delta_E / T)):
            current_energy = new_energy
            no_improve_count = 0
        else:
            # Reject: revert
            current[row_idx][pos] = old_val
            no_improve_count += 1
        
        # Update best if valid and better
        if is_valid(current):
            scope = max(max(row) for row in current)
            if scope < best_energy:
                best = [list(row) for row in current]
                best_energy = scope
        
        # Cool down
        T *= cooling_rate
        
        # If stuck for too long, restart from best with perturbation
        if no_improve_count > 200:
            current = [list(row) for row in best]
            # Perturb: make a few random valid changes
            for _ in range(5):
                r = rng.randrange(n_rows)
                p = rng.randrange(1, k_plus_1)
                d = rng.choice([-1, 1])
                nv = current[r][p] + d
                if p > 0 and nv > current[r][p-1] and (p == k_plus_1-1 or nv < current[r][p+1]) and nv > 0:
                    current[r][p] = nv
            current_energy, _ = compute_energy(current)
            no_improve_count = 0
            T = max(T, 10.0)  # Reheat
    
    # Final local search from best
    best = _improve_scope(best, n_rows, k, deadline)
    best_scope = max(max(row) for row in best)
    
    return best, best_scope

def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        rows.append(row)
    return rows

def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search to build n_rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, and max element <= max_scope.
    Uses bitsets for fast difference checking.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    rows = []
    k_plus_1 = k + 1
    
    # Use a list of booleans for used differences (faster than set for small ranges)
    max_diff = max_scope
    used_diffs = [False] * (max_diff + 1)
    
    def build_row(pos, current_mark, row_so_far):
        if time.time() > deadline:
            return None
            
        if pos == k_plus_1:
            return list(row_so_far)
        
        # Find valid candidates
        valid_candidates = []
        # Prune upper bound: we need at least (k_plus_1 - pos - 1) more elements,
        # each at least 1 greater than previous, so max possible x is:
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None
            
        for x in range(current_mark + 1, max_x + 1):
            # Check if any new difference is already used
            conflict = False
            for a in row_so_far:
                d = x - a
                if used_diffs[d]:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
        
        if not valid_candidates:
            return None
        
        # Randomize the order of candidates to escape local minima
        # But bias towards smaller values to keep scope small
        n_smallest = min(12, len(valid_candidates))
        smallest = valid_candidates[:n_smallest]
        
        # With some probability, also consider larger candidates
        if len(valid_candidates) > n_smallest and rng.random() < 0.3:
            extra = rng.sample(valid_candidates[n_smallest:], min(6, len(valid_candidates) - n_smallest))
            candidates_to_try = smallest + extra
        else:
            candidates_to_try = smallest
        
        rng.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            # Mark new differences
            new_diffs = []
            for a in row_so_far:
                d = x - a
                new_diffs.append(d)
                used_diffs[d] = True
            
            res = build_row(pos + 1, x, row_so_far + [x])
            if res is not None:
                return res
            
            # Unmark differences
            for d in new_diffs:
                used_diffs[d] = False
        
        return None
    
    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        # Mark all differences in this row as used
        for j in range(k_plus_1):
            for i2 in range(j + 1, k_plus_1):
                used_diffs[row[i2] - row[j]] = True
    
    return rows

def _improve_scope(solution, n_rows, k, deadline):
    """
    Try to reduce the scope of a valid solution by local search.
    We try to shift elements down to reduce the maximum value.
    Includes 2-element swap perturbations to escape local minima.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1
    rng = random.Random(123)
    
    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])
    
    def try_shift_down(row_idx, pos, new_val):
        """Try to shift element at row_idx[pos] to new_val (which is lower)."""
        nonlocal used_diffs
        row = best[row_idx]
        original = row[pos]
        
        if new_val <= row[pos - 1] or new_val >= original:
            return False
        
        # Check bounds with next element
        if pos < k_plus_1 - 1 and new_val >= row[pos + 1]:
            return False
        
        # Compute new diffs
        new_diffs = set()
        for i in range(pos):
            new_diffs.add(new_val - row[i])
        for j in range(pos + 1, k_plus_1):
            new_diffs.add(row[j] - new_val)
        
        # Remove old diffs
        old_diffs = set()
        for i in range(pos):
            old_diffs.add(original - row[i])
        for j in range(pos + 1, k_plus_1):
            old_diffs.add(row[j] - original)
        
        # Check conflicts
        if new_diffs & (used_diffs - old_diffs):
            return False
        
        # Apply
        used_diffs.difference_update(old_diffs)
        used_diffs.update(new_diffs)
        row[pos] = new_val
        return True
    
    improved = True
    stagnation = 0
    while improved and time.time() < deadline - 3.0:
        improved = False
        stagnation = 0
        
        # Pass 1: Try to shift elements down (from end to start of each row)
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos in range(k_plus_1 - 1, 0, -1):
                original = row[pos]
                # Try shifting down by 1, 2, ...
                for new_val in range(original - 1, row[pos - 1], -1):
                    if try_shift_down(row_idx, pos, new_val):
                        improved = True
                        break
        
        # Pass 2: 2-element swap perturbation to escape local minima
        if not improved and stagnation < 10:
            # Pick two positions in different rows (or same row) and try swapping values
            r1 = rng.randrange(n_rows)
            r2 = rng.randrange(n_rows)
            p1 = rng.randrange(1, k_plus_1)
            p2 = rng.randrange(1, k_plus_1)
            
            if r1 == r2 and p1 == p2:
                stagnation += 1
                continue
            
            v1 = best[r1][p1]
            v2 = best[r2][p2]
            
            # Try swapping: set best[r1][p1] = v2, best[r2][p2] = v1
            # But must maintain increasing order
            valid = True
            if r1 == r2:
                if p1 < p2:
                    if v2 <= best[r1][p1 - 1] or v2 >= best[r1][p1 + 1] if p1 + 1 < k_plus_1 else False:
                        valid = False
                    if v1 <= best[r1][p2 - 1] or v1 >= best[r1][p2 + 1] if p2 + 1 < k_plus_1 else False:
                        valid = False
                else:
                    if v2 <= best[r1][p1 - 1] or v2 >= best[r1][p1 + 1] if p1 + 1 < k_plus_1 else False:
                        valid = False
                    if v1 <= best[r1][p2 - 1] or v1 >= best[r1][p2 + 1] if p2 + 1 < k_plus_1 else False:
                        valid = False
            else:
                if v2 <= best[r1][p1 - 1] or (p1 < k_plus_1 - 1 and v2 >= best[r1][p1 + 1]):
                    valid = False
                if v1 <= best[r2][p2 - 1] or (p2 < k_plus_1 - 1 and v1 >= best[r2][p2 + 1]):
                    valid = False
            
            if valid:
                # Check difference conflicts
                # Remove old diffs for both positions
                old_diffs = set()
                for i in range(p1):
                    old_diffs.add(v1 - best[r1][i])
                for j in range(p1 + 1, k_plus_1):
                    old_diffs.add(best[r1][j] - v1)
                for i in range(p2):
                    old_diffs.add(v2 - best[r2][i])
                for j in range(p2 + 1, k_plus_1):
                    old_diffs.add(best[r2][j] - v2)
                
                # Compute new diffs
                new_diffs = set()
                for i in range(p1):
                    new_diffs.add(v2 - best[r1][i])
                for j in range(p1 + 1, k_plus_1):
                    new_diffs.add(best[r1][j] - v2)
                for i in range(p2):
                    new_diffs.add(v1 - best[r2][i])
                for j in range(p2 + 1, k_plus_1):
                    new_diffs.add(best[r2][j] - v1)
                
                if not (new_diffs & (used_diffs - old_diffs)):
                    # Apply swap
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    best[r1][p1] = v2
                    best[r2][p2] = v1
                    improved = True
        
        if not improved:
            stagnation += 1
    
    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]

def _polish_scope(solution, n_rows, k, deadline):
    """
    Final polish: try to reduce the scope by 1 at a time using targeted backtracking.
    """
    if not solution:
        return solution
    
    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    
    # Try to reduce scope by 1
    while best_scope > 0 and time.time() < deadline - 2.0:
        target_scope = best_scope - 1
        if target_scope < 0:
            break
        
        # Try a few random restarts
        found = False
        for attempt in range(20):
            if time.time() > deadline - 2.0:
                break
            
            sol = _backtracking_search(n_rows, k, target_scope, deadline, attempt * 7 + 13)
            if sol:
                new_scope = max(max(row) for row in sol)
                if new_scope <= target_scope:
                    best = sol
                    best_scope = new_scope
                    found = True
                    break
        
        if not found:
            break
    
    return best
