import time
import sys
import random
import math
import numpy as np

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 115
    rng = random.Random(42)
    
    # Greedy construction for initial solution
    best_solution = _greedy_construct(n_rows, k, rng)
    if not best_solution:
        return []
    
    best_score = max(max(r) for r in best_solution)
    
    # Phase 1: Multi-start with row reconstruction
    n_starts = 5
    for start_i in range(n_starts):
        if time.time() > deadline - 40:
            break
            
        if start_i == 0:
            current = [r[:] for r in best_solution]
        else:
            rng_local = random.Random(42 + start_i * 1000)
            current = _greedy_construct(n_rows, k, rng_local, randomize=True)
            if not current:
                continue
        
        # Row-level reconstruction with simulated annealing
        improved = _row_reconstruction_sa(current, n_rows, k, deadline, rng)
        
        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
    
    # Phase 2: Element-level local search for polish
    if time.time() < deadline - 20:
        final = _local_search_minimize_fast([r[:] for r in best_solution], n_rows, k, deadline)
        if final:
            final_score = max(max(r) for r in final)
            if final_score < best_score:
                best_solution = final
                best_score = final_score
    
    # Phase 3: Second polish pass with more iterations
    if time.time() < deadline - 5:
        final2 = _local_search_minimize_fast([r[:] for r in best_solution], n_rows, k, deadline)
        if final2:
            final2_score = max(max(r) for r in final2)
            if final2_score < best_score:
                best_solution = final2
                best_score = final2_score
    
    return best_solution

def _row_diffs_set(row):
    """Compute the set of all positive differences in a row."""
    diffs = set()
    n = len(row)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.add(row[j] - row[i])
    return diffs

def _compute_all_used(rows):
    """Compute the union of all differences across all rows."""
    all_used = set()
    for row in rows:
        all_used |= _row_diffs_set(row)
    return all_used

def _global_max(rows):
    """Compute the maximum entry across all rows."""
    return max(max(row) for row in rows)

def _random_valid_row(forbidden_diffs, max_limit, k, max_candidates=8, rng=None):
    """
    Construct a random valid row of length k+1 starting at 0, strictly increasing,
    with all internal differences distinct and disjoint from forbidden_diffs,
    with max entry <= max_limit.
    """
    if rng is None:
        rng = random
    row = [0]
    used_in_row = set()
    last_val = 0
    
    for step in range(k):
        candidates = []
        x = last_val + 1
        # Wider search window for earlier steps
        search_width = 300 if step < k // 2 else 200
        limit = min(max_limit, last_val + search_width) if max_limit is not None else last_val + search_width
        
        while x <= limit:
            new_diffs = {x - a for a in row}
            if not (new_diffs & forbidden_diffs) and not (new_diffs & used_in_row):
                candidates.append(x)
            
            x += 1
            if len(candidates) >= max_candidates:
                break
        
        if not candidates:
            return None
        
        # Weight toward smaller candidates to keep the row compact
        if len(candidates) > 1 and rng.random() < 0.7:
            # Pick from the first half of candidates
            half = max(1, len(candidates) // 2)
            chosen_x = rng.choice(candidates[:half])
        else:
            chosen_x = rng.choice(candidates)
        
        row.append(chosen_x)
        new_diffs = {chosen_x - a for a in row[:-1]}
        used_in_row |= new_diffs
        last_val = chosen_x
        
    return row

def _row_reconstruction_sa(rows, n_rows, k, deadline, rng):
    """
    Row-level simulated annealing: repeatedly replace one row with a random valid
    row, accepting worse solutions with probability based on temperature.
    """
    current_rows = [r[:] for r in rows]
    current_val = _global_max(current_rows)
    best_rows = [r[:] for r in current_rows]
    best_val = current_val
    
    row_maxes = [max(r) for r in current_rows]
    row_diffs = [_row_diffs_set(r) for r in current_rows]
    
    T = 80.0
    T_min = 0.5
    cool = 0.999
    no_improve = 0
    max_no_improve = 3000
    
    while time.time() < deadline - 15 and no_improve < max_no_improve:
        # Preferentially target the row that achieves the global max
        if rng.random() < 0.7:
            max_rows = [i for i in range(n_rows) if row_maxes[i] == current_val]
            if max_rows:
                ri = rng.choice(max_rows)
            else:
                max_rows = [i for i in range(n_rows) if row_maxes[i] >= current_val - 3]
                ri = rng.choice(max_rows) if max_rows else rng.randrange(n_rows)
        elif rng.random() < 0.5:
            # Target second-highest rows
            sorted_rows = sorted(range(n_rows), key=lambda i: row_maxes[i], reverse=True)
            ri = rng.choice(sorted_rows[:max(1, n_rows // 3)])
        else:
            ri = rng.randrange(n_rows)
        
        forbidden = set()
        for j in range(n_rows):
            if j != ri:
                forbidden |= row_diffs[j]
        
        # Target: try to beat current global max, or at least not worsen it
        target_max = current_val - 1
        if target_max < k:
            target_max = current_rows[ri][-1] - 1
        
        new_row = _random_valid_row(forbidden, target_max, k, max_candidates=15, rng=rng)
        
        if new_row is None:
            target_max2 = current_val
            new_row = _random_valid_row(forbidden, target_max2, k, max_candidates=12, rng=rng)
        
        if new_row is None:
            target_max3 = current_val + 20
            new_row = _random_valid_row(forbidden, target_max3, k, max_candidates=10, rng=rng)
        
        if new_row is None:
            no_improve += 1
            T = max(T_min, T * cool)
            continue
        
        new_row_max = new_row[-1]
        other_maxes = [row_maxes[j] for j in range(n_rows) if j != ri]
        new_global_max = max(max(other_maxes) if other_maxes else 0, new_row_max)
        
        delta = new_global_max - current_val
        
        accept = False
        if delta < 0:
            accept = True
        elif delta == 0:
            if rng.random() < 0.5:
                accept = True
        else:
            if T > T_min:
                prob = math.exp(-delta / max(T, 0.1))
                if rng.random() < prob:
                    accept = True
        
        if accept:
            new_diffs = _row_diffs_set(new_row)
            current_rows[ri] = new_row
            row_diffs[ri] = new_diffs
            row_maxes[ri] = new_row_max
            current_val = new_global_max
            
            if current_val < best_val:
                best_val = current_val
                best_rows = [r[:] for r in current_rows]
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
            
        T = max(T_min, T * cool)
        
        # Restart if stuck for too long
        if no_improve > max_no_improve // 3:
            current_rows = [r[:] for r in best_rows]
            current_val = best_val
            row_diffs = [_row_diffs_set(r) for r in current_rows]
            row_maxes = [max(r) for r in current_rows]
            T = 80.0
            no_improve = 0
    
    return best_rows

def _local_search_minimize_fast(rows, n_rows, k, deadline):
    """
    Fast local search with incremental diff tracking.
    """
    best_rows = [r[:] for r in rows]
    n = k + 1
    
    # Build initial diff set
    all_diffs = set()
    for r in best_rows:
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.add(r[j] - r[i])
    
    progress = True
    while progress and time.time() < deadline - 5:
        progress = False
        # Prioritize positions in rows that contribute to the global max
        row_maxes = [max(r) for r in best_rows]
        global_max = max(row_maxes)
        indices = []
        for r_idx in range(n_rows):
            for pos in range(1, k+1):
                indices.append((r_idx, pos))
        
        # Sort to try high-max rows first, but still shuffle within groups
        indices.sort(key=lambda x: (-row_maxes[x[0]], random.random()))
        
        for r_idx, pos in indices:
            if time.time() > deadline - 5:
                break
                
            row = best_rows[r_idx]
            old_val = row[pos]
            
            # Skip if this position doesn't affect the max and we're not in a high-max row
            if row_maxes[r_idx] < global_max and pos != k:
                # Still try but with lower priority (already sorted)
                pass
            
            # Compute diffs involving (r_idx, pos)
            diffs_involving = set()
            for i in range(n):
                if i == pos:
                    continue
                if i < pos:
                    diffs_involving.add(old_val - row[i])
                else:
                    diffs_involving.add(row[i] - old_val)
            
            forbidden_diffs = all_diffs - diffs_involving
            
            # Try to find smaller valid value
            lower_bound = row[pos-1] + 1
            upper_bound = old_val - 1
            if pos + 1 < n:
                upper_bound = min(upper_bound, row[pos+1] - 1)
            
            if upper_bound >= lower_bound:
                found = False
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val
                        
                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        # Update: remove old diffs, add new diffs
                        all_diffs.difference_update(diffs_involving)
                        all_diffs.update(new_diffs)
                        best_rows[r_idx][pos] = new_val
                        progress = True
                        found = True
                        break
                
                if found:
                    continue
    
    return best_rows

def _greedy_construct(n_rows, k, rng=None, randomize=False):
    """
    Greedy construction of initial solution.
    """
    if rng is None:
        rng = random
    
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Try values starting from current_max + 1
            for x in range(current_max + 1, current_max + 500):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    # Add randomization if requested
                    if randomize and rng.random() < 0.1 and x < 200:
                        skip = rng.randint(1, 3)
                        x += skip
                        new_diffs = []
                        valid = True
                        for a in row:
                            d = x - a
                            if d in used_diffs:
                                valid = False
                                break
                            new_diffs.append(d)
                        if not valid:
                            x -= skip
                            new_diffs = []
                            for a in row:
                                d = x - a
                                new_diffs.append(d)
                    
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                # Try a wider search with some randomization
                for _retry in range(50):
                    x = current_max + 1 + rng.randint(0, 100)
                    new_diffs = []
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        row.append(x)
                        used_diffs.update(new_diffs)
                        current_max = x
                        found = True
                        break
                if not found:
                    return None
        rows.append(row)
    return rows
