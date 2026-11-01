"""Difference triangle set: minimize largest entry for n_rows rows of k+1 increasing ints from 0.

Structural approach:
1. Feasibility-guided greedy: build rows by tracking a global difference set,
   using a priority queue to always place the "tightest" next value.
2. Targeted descent: iteratively reduce the global max by trying to shift
   elements downward, checking only affected differences.
3. Differential evolution: maintain a population of candidate solutions,
   crossover/mutate to explore the space, select by scope.
4. Deadline guard: explicit time tracking with safety margin.
"""
import time
import random
import sys
from heapq import heappush, heappop

_START_TIME = None
_DEADLINE = 145.0  # safety margin before 150s kill


def _elapsed():
    return time.time() - _START_TIME


def _time_left():
    return _DEADLINE - _elapsed()


def _row_diffs(row):
    """Return sorted list of all positive differences in a row."""
    n = len(row)
    diffs = []
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            diffs.append(row[j] - ri)
    return diffs


def _row_diff_set(row):
    """Return a set of all positive differences in a row."""
    n = len(row)
    s = set()
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s


def _validate(rows):
    """Check all differences are distinct across all rows. Return True if valid."""
    seen = set()
    for row in rows:
        d = _row_diff_set(row)
        n = len(row)
        if len(d) != n * (n - 1) // 2:
            return False
        if d & seen:
            return False
        seen |= d
    return True


def _scope(rows):
    """Return the maximum entry across all rows."""
    return max(max(r) for r in rows)


def _greedy_build_packed(n_rows, k, rng, initial_scope=None):
    """
    Build rows greedily with a global difference set.
    Uses a strategy of always picking the smallest feasible next value,
    but with randomization to escape local minima.
    """
    used = set()
    rows = []
    
    for r in range(n_rows):
        row = [0]
        while len(row) < k + 1:
            last = row[-1]
            # Try values starting from last+1, with some randomization
            start = last + 1
            if initial_scope is not None:
                # Bias towards smaller values
                max_try = min(initial_scope, last + 50)
            else:
                max_try = last + 100
            
            found = False
            for attempt in range(20):
                if attempt == 0:
                    # First try: smallest feasible
                    x = start
                    while x <= max_try:
                        ok = True
                        for a in row:
                            d = x - a
                            if d in used:
                                ok = False
                                break
                        if ok:
                            row.append(x)
                            for a in row[:-1]:
                                used.add(x - a)
                            found = True
                            break
                        x += 1
                    if found:
                        break
                else:
                    # Subsequent tries: random offset
                    x = start + rng.randint(0, 20)
                    if x > max_try:
                        continue
                    ok = True
                    for a in row:
                        d = x - a
                        if d in used:
                            ok = False
                            break
                    if ok:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                        found = True
                        break
            
            if not found:
                # Fallback: search wider
                x = start
                while True:
                    ok = True
                    for a in row:
                        d = x - a
                        if d in used:
                            ok = False
                            break
                    if ok:
                        row.append(x)
                        for a in row[:-1]:
                            used.add(x - a)
                        break
                    x += 1
        
        rows.append(row)
    
    return rows


def _targeted_descent(rows, n_rows, k, rng, time_budget):
    """
    Targeted descent: repeatedly try to reduce the global maximum by
    shifting elements downward. Focus on the row with the current max.
    """
    rows = [list(r) for r in rows]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    start = time.time()
    
    # Precompute diff sets for each row
    row_diff_sets = [_row_diff_set(rows[r]) for r in range(n_rows)]
    global_diffs = set()
    for rs in row_diff_sets:
        global_diffs |= rs
    
    while time.time() - start < time_budget and _time_left() > 3.0:
        # Find the row with the maximum element
        max_row = 0
        max_val = -1
        for r in range(n_rows):
            if rows[r][-1] > max_val:
                max_val = rows[r][-1]
                max_row = r
        
        if max_val <= best_scope:
            # Try to reduce other rows too
            max_row = rng.randrange(n_rows)
        
        row = rows[max_row]
        improved = False
        
        # Try reducing the last element
        for new_val in range(row[-1] - 1, row[-2], -1):
            new_row = list(row)
            new_row[-1] = new_val
            
            # Check if this creates any conflicts
            new_diffs = _row_diff_set(new_row)
            if len(new_diffs) != k * (k + 1) // 2:
                continue
            
            # Check against all other rows' diffs
            conflict = False
            for other_r in range(n_rows):
                if other_r == max_row:
                    continue
                if new_diffs & row_diff_sets[other_r]:
                    conflict = True
                    break
            
            if not conflict:
                # Accept the move
                old_diffs = row_diff_sets[max_row]
                # Remove old diffs from global
                global_diffs -= old_diffs
                # Add new diffs
                global_diffs |= new_diffs
                row_diff_sets[max_row] = new_diffs
                rows[max_row] = new_row
                
                new_scope = _scope(rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [list(r) for r in rows]
                improved = True
                break
        
        if not improved:
            # Try reducing a random non-last element in a random row
            r = rng.randrange(n_rows)
            if len(rows[r]) <= 2:
                continue
            pos = rng.randrange(1, len(rows[r]) - 1)
            row = rows[r]
            old = row[pos]
            
            for new_val in range(old - 1, row[pos - 1], -1):
                if pos < k and new_val >= row[pos + 1]:
                    continue
                
                new_row = list(row)
                new_row[pos] = new_val
                new_diffs = _row_diff_set(new_row)
                if len(new_diffs) != k * (k + 1) // 2:
                    continue
                
                conflict = False
                for other_r in range(n_rows):
                    if other_r == r:
                        continue
                    if new_diffs & row_diff_sets[other_r]:
                        conflict = True
                        break
                
                if not conflict:
                    old_diffs = row_diff_sets[r]
                    global_diffs -= old_diffs
                    global_diffs |= new_diffs
                    row_diff_sets[r] = new_diffs
                    rows[r] = new_row
                    
                    new_scope = _scope(rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = [list(r) for r in rows]
                    break
    
    return best_rows, best_scope


def _differential_evolution(rows, n_rows, k, rng, time_budget, population_size=8):
    """
    Differential evolution over a population of candidate solutions.
    Each candidate is a set of rows. We mutate by perturbing one element
    of one row and accepting if it reduces scope or maintains validity.
    """
    population = [[list(r) for r in rows] for _ in range(population_size)]
    best_rows = [list(r) for r in rows]
    best_scope = _scope(rows)
    start = time.time()
    
    # Precompute for each candidate
    pop_diff_sets = []
    for cand in population:
        pop_diff_sets.append([_row_diff_set(cand[r]) for r in range(n_rows)])
    
    pop_scopes = [_scope(cand) for cand in population]
    
    # Find the best in population
    best_idx = pop_scopes.index(min(pop_scopes))
    best_rows = [list(r) for r in population[best_idx]]
    best_scope = pop_scopes[best_idx]
    
    generation = 0
    while time.time() - start < time_budget and _time_left() > 3.0:
        generation += 1
        
        # Select parents
        i = rng.randrange(population_size)
        j, l = rng.sample([x for x in range(population_size) if x != i], 2)
        
        # Mutate: pick a random row and position
        r = rng.randrange(n_rows)
        pos = rng.randrange(1, k + 1)
        
        # Differential: use the difference between two parents
        delta = population[j][r][pos] - population[l][r][pos]
        if delta == 0:
            delta = rng.randint(-5, 5)
        
        new_val = population[i][r][pos] + delta
        
        # Ensure validity
        if new_val <= population[i][r][pos - 1]:
            continue
        if pos < k and new_val >= population[i][r][pos + 1]:
            continue
        if new_val <= 0:
            continue
        
        # Create mutant
        mutant = [list(row) for row in population[i]]
        mutant[r][pos] = new_val
        
        # Check validity of mutant
        new_diffs = _row_diff_set(mutant[r])
        if len(new_diffs) != k * (k + 1) // 2:
            continue
        
        # Check for conflicts with other rows in the mutant
        conflict = False
        for other_r in range(n_rows):
            if other_r == r:
                continue
            if new_diffs & _row_diff_set(mutant[other_r]):
                conflict = True
                break
        
        if conflict:
            continue
        
        # Evaluate mutant
        mutant_scope = _scope(mutant)
        
        # Replace if better or with some probability
        if mutant_scope < pop_scopes[i] or (mutant_scope == pop_scopes[i] and rng.random() < 0.1):
            population[i] = mutant
            pop_diff_sets[i][r] = new_diffs
            pop_scopes[i] = mutant_scope
            
            if mutant_scope < best_scope:
                best_scope = mutant_scope
                best_rows = [list(row) for row in mutant]
        
        # Occasionally restart a random member with a fresh greedy build
        if generation % 50 == 0 and _time_left() > 10.0:
            restart_idx = rng.randrange(population_size)
            fresh = _greedy_build_packed(n_rows, k, rng, initial_scope=best_scope)
            fresh_scope = _scope(fresh)
            if fresh_scope < pop_scopes[restart_idx]:
                population[restart_idx] = fresh
                pop_diff_sets[restart_idx] = [_row_diff_set(fresh[r]) for r in range(n_rows)]
                pop_scopes[restart_idx] = fresh_scope
                if fresh_scope < best_scope:
                    best_scope = fresh_scope
                    best_rows = [list(row) for row in fresh]
    
    return best_rows, best_scope


def _backtrack_solve(n_rows, k, max_val, time_budget):
    """
    Backtracking solver to find n_rows rows of length k+1 with max entry <= max_val.
    Uses recursive construction with forward checking.
    """
    start_time = time.time()
    used_diffs = set()
    rows = []
    diffs_per_row = k * (k + 1) // 2
    
    if max_val < diffs_per_row:
        return None

    def solve_row(row_idx, current_row, current_used, time_limit):
        if time.time() - start_time > time_limit:
            return None
        
        if len(current_row) == k + 1:
            return list(current_row)
        
        pos = len(current_row)
        if pos == 0:
            candidates = [0]
        else:
            last_val = current_row[-1]
            candidates = range(last_val + 1, max_val + 1)
            
        for val in candidates:
            new_diffs = set()
            conflict = False
            for prev in current_row:
                d = val - prev
                if d in current_used:
                    conflict = True
                    break
                new_diffs.add(d)
            
            if conflict:
                continue
            
            new_used = current_used | new_diffs
            result = solve_row(row_idx, current_row + [val], new_used, time_limit)
            if result is not None:
                return result
        
        return None

    for r in range(n_rows):
        if time.time() - start_time > time_budget:
            return None
        
        row = solve_row(r, [0], used_diffs, time_budget)
        if row is None:
            return None
        rows.append(row)
        used_diffs |= _row_diff_set(row)
        
    return rows


def construct(n_rows, k):
    global _START_TIME
    _START_TIME = time.time()

    rng = random.Random(42)

    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Multiple greedy builds with different randomizations
    for t in range(15):
        if _time_left() < 10.0:
            break
        rows = _greedy_build_packed(n_rows, k, rng)
        cur_scope = _scope(rows)
        if cur_scope < best_scope:
            best_scope = cur_scope
            best_rows = [list(r) for r in rows]

    if best_rows is None:
        best_rows = _greedy_build_packed(n_rows, k, rng)
        best_scope = _scope(best_rows)

    # Phase 2: Targeted descent to reduce scope
    if _time_left() > 15.0:
        budget = min(30.0, _time_left() - 10.0)
        td_rows, td_scope = _targeted_descent(best_rows, n_rows, k, rng, budget)
        if td_scope < best_scope:
            best_scope = td_scope
            best_rows = td_rows

    # Phase 3: Differential evolution for further optimization
    if _time_left() > 20.0:
        budget = min(40.0, _time_left() - 10.0)
        de_rows, de_scope = _differential_evolution(best_rows, n_rows, k, rng, budget)
        if de_scope < best_scope:
            best_scope = de_scope
            best_rows = de_rows

    # Phase 4: Try backtracking with binary search if we have time
    total_diffs = n_rows * k * (k + 1) // 2
    lo = total_diffs
    hi = best_scope
    
    if _time_left() > 20.0 and hi - lo < 100:
        # Try to find a tighter bound via backtracking
        for S in range(lo, min(hi + 1, lo + 50)):
            if _time_left() < 15.0:
                break
            bt_time = min(5.0, _time_left() - 10.0)
            rows = _backtrack_solve(n_rows, k, S, bt_time)
            if rows is not None and _validate(rows):
                bt_scope = _scope(rows)
                if bt_scope < best_scope:
                    best_scope = bt_scope
                    best_rows = rows
                break

    # Phase 5: Final polish with targeted descent
    if _time_left() > 10.0:
        budget = min(15.0, _time_left() - 5.0)
        td_rows, td_scope = _targeted_descent(best_rows, n_rows, k, rng, budget)
        if td_scope < best_scope:
            best_scope = td_scope
            best_rows = td_rows

    # Final verification
    if not _validate(best_rows):
        # Fallback to a fresh greedy build
        best_rows = _greedy_build_packed(n_rows, k, rng)
        best_scope = _scope(best_rows)

    return best_rows
