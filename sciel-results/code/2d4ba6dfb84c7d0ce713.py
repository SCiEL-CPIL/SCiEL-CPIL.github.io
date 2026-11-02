import time
import random
import math
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 115  # Safety margin before 120s limit

    # Generate initial solution using greedy construction
    initial_solution = _greedy_construct(n_rows, k)
    if not initial_solution:
        return []
    
    # Verify initial solution is valid
    if not _is_valid_solution(initial_solution, n_rows, k):
        return []

    best_solution = [r[:] for r in initial_solution]
    best_score = max(max(r) for r in best_solution)

    # Pre-generate a pool of candidate rows for branch-and-bound
    # We'll generate rows with max entry up to best_score + some buffer
    # This pool will be used for the B&B search
    candidate_pool = _generate_candidate_pool(n_rows, k, best_score + 10, deadline)

    # Use branch-and-bound with Lagrangian-style pruning
    bb_solution = _branch_and_bound_search(n_rows, k, best_score, candidate_pool, deadline)
    if bb_solution is not None:
        bb_score = max(max(r) for r in bb_solution)
        if bb_score < best_score:
            best_score = bb_score
            best_solution = bb_solution

    # Iterated Local Search with valid perturbations
    no_improve_count = 0
    iter_count = 0
    while time.time() < deadline - 20:
        iter_count += 1
        # Perturb the best solution using valid moves
        candidate = [r[:] for r in best_solution]

        # Perform several random valid moves (perturbation)
        n_moves = random.randint(3, 8)
        for _ in range(n_moves):
            move_type = random.random()
            if move_type < 0.6:
                # Single position change
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
                new_val = _find_random_valid_value(candidate, r_idx, pos, n_rows, k)
                if new_val is not None and new_val != candidate[r_idx][pos]:
                    candidate[r_idx][pos] = new_val
            elif move_type < 0.8:
                # Swap two positions within a row
                r_idx = random.randint(0, n_rows - 1)
                if k >= 2:
                    p1, p2 = random.sample(range(1, k + 1), 2)
                    if _try_swap(candidate, r_idx, p1, p2, n_rows, k):
                        pass  # swap applied
            else:
                # Rebuild one row greedily
                if n_rows > 1:
                    r_idx = random.randint(0, n_rows - 1)
                    other_rows = [r for i, r in enumerate(candidate) if i != r_idx]
                    used_diffs_other = _compute_all_diffs(other_rows)
                    new_row = _greedy_row_given_diffs(used_diffs_other, k)
                    if new_row is not None:
                        candidate[r_idx] = new_row

        # Local search to minimize scope
        improved = _local_search_minimize(candidate, n_rows, k, deadline)

        if improved:
            # Validate improved solution
            if _is_valid_solution(improved, n_rows, k):
                curr_score = max(max(r) for r in improved)
                if curr_score < best_score:
                    best_score = curr_score
                    best_solution = improved
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1

        # If we've had many iterations without improvement, try a stronger perturbation
        if no_improve_count >= 4:
            no_improve_count = 0
            # Try rebuilding 1-2 rows
            n_rebuild = random.randint(1, min(2, n_rows))
            for _ in range(n_rebuild):
                if time.time() > deadline - 20:
                    break
                r_idx = random.randint(0, n_rows - 1)
                other_rows = [r for i, r in enumerate(candidate) if i != r_idx]
                used_diffs_other = _compute_all_diffs(other_rows)
                new_row = _greedy_row_given_diffs(used_diffs_other, k)
                if new_row is not None:
                    candidate[r_idx] = new_row
            improved2 = _local_search_minimize(candidate, n_rows, k, deadline)
            if improved2:
                if _is_valid_solution(improved2, n_rows, k):
                    curr_score2 = max(max(r) for r in improved2)
                    if curr_score2 < best_score:
                        best_score = curr_score2
                        best_solution = improved2

    # Final polish pass with remaining time
    if time.time() < deadline - 3:
        polished = _local_search_minimize(best_solution, n_rows, k, deadline)
        if polished:
            # Validate the polished solution before accepting
            if _is_valid_solution(polished, n_rows, k):
                curr_score = max(max(r) for r in polished)
                if curr_score < best_score:
                    best_score = curr_score
                    best_solution = polished

    # Final validation check
    if _is_valid_solution(best_solution, n_rows, k):
        return best_solution
    
    # Fallback to initial solution if best_solution became invalid
    if _is_valid_solution(initial_solution, n_rows, k):
        return initial_solution
    
    return []

def _is_valid_solution(rows, n_rows, k):
    """Check if the solution is valid: correct dimensions, strictly increasing, distinct diffs."""
    if not rows:
        return False
    if len(rows) != n_rows:
        return False
    
    all_diffs = set()
    for r in rows:
        if len(r) != k + 1:
            return False
        if r[0] != 0:
            return False
        
        # Check strictly increasing
        for i in range(len(r) - 1):
            if r[i] >= r[i + 1]:
                return False
        
        # Check differences
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                d = r[j] - r[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
                
    return True


def _generate_candidate_pool(n_rows, k, max_entry, deadline):
    """Generate a pool of valid Sidon rows with max entry <= max_entry."""
    pool = []
    rng = random.Random(12345)
    num_to_generate = 200  # Generate up to 200 candidate rows
    
    for _ in range(num_to_generate):
        if time.time() > deadline - 30:
            break
        row = _random_valid_row(None, max_entry, k, rng, max_candidates=3)
        if row is not None:
            pool.append(row)
    
    return pool


def _random_valid_row(forbidden_diffs, max_limit, k, rng, max_candidates=5):
    """
    Construct a random valid row of length k+1 starting at 0, strictly increasing,
    with all internal differences distinct and disjoint from forbidden_diffs,
    with max entry <= max_limit.
    """
    row = [0]
    used_in_row = set()
    
    for _ in range(k):
        candidates = []
        start = row[-1] + 1
        end = max_limit if max_limit is not None else start + 300
        
        for x in range(start, end):
            new_diffs = {x - a for a in row}
            if forbidden_diffs and new_diffs & forbidden_diffs:
                continue
            if new_diffs & used_in_row:
                continue
            candidates.append(x)
            if len(candidates) >= max_candidates:
                break
        
        if not candidates:
            return None
        
        chosen_x = rng.choice(candidates)
        row.append(chosen_x)
        used_in_row |= {chosen_x - a for a in row[:-1]}
        
    return row


def _branch_and_bound_search(n_rows, k, current_best, candidate_pool, deadline):
    """
    Branch-and-bound search for minimum maximum entry.
    Uses Lagrangian-style pruning based on conflict density.
    """
    if not candidate_pool:
        return None
    
    # Sort candidates by max entry (ascending) for better pruning
    candidate_pool.sort(key=lambda r: max(r))
    
    # Compute diff sets for all candidates
    candidate_diffs = [_row_diffs(r) for r in candidate_pool]
    
    best_found = current_best
    best_rows = None
    
    # Recursive B&B with time limit
    selected = []
    used_diffs = set()
    
    def bbb(idx, depth):
        nonlocal best_found, best_rows
        
        if time.time() > deadline - 15:
            return False
        
        if depth == n_rows:
            # Complete solution found
            sol_max = max(max(r) for r in selected)
            if sol_max < best_found:
                best_found = sol_max
                best_rows = [r[:] for r in selected]
            return True
        
        # Lagrangian-style lower bound:
        # Estimate minimum additional conflicts if we pick from remaining candidates
        # Simple bound: at least 1 more row needed, each has max >= some value
        remaining_candidates = candidate_pool[idx:]
        if not remaining_candidates:
            return False
        
        # Pruning: if even the smallest remaining candidate's max exceeds best_found, prune
        min_remaining_max = min(max(r) for r in remaining_candidates)
        if min_remaining_max >= best_found and depth > 0:
            # We might still improve if we can find a combination with lower max
            # But this is a weak bound; let's be more careful
            pass
        
        # Conflict-based pruning: count how many candidates are compatible with used_diffs
        compatible_count = 0
        for j in range(idx, len(candidate_pool)):
            if not (candidate_diffs[j] & used_diffs):
                compatible_count += 1
        
        # If not enough compatible candidates remain, prune
        if compatible_count < (n_rows - depth):
            return False
        
        # Try candidates in order of increasing max entry
        tried = set()
        for j in range(idx, len(candidate_pool)):
            if time.time() > deadline - 15:
                return False
            
            if j in tried:
                continue
            tried.add(j)
            
            row = candidate_pool[j]
            rd = candidate_diffs[j]
            
            # Skip if conflicts with already used diffs
            if rd & used_diffs:
                continue
            
            # Skip if this row's max already exceeds best_found (no improvement possible)
            if max(row) >= best_found:
                # But we might still need this row for completeness...
                # Actually, if max(row) >= best_found, it can't improve the objective
                # unless it allows other rows to be smaller, but since we're minimizing max,
                # picking a row with max >= best_found can't help
                continue
            
            # Select this row
            selected.append(row)
            new_used = used_diffs | rd
            
            # Recurse
            if bbb(j + 1, depth + 1):
                selected.pop()
                return True
            
            selected.pop()
            
            # If we've tried enough candidates at this level, stop
            if len(tried) >= 20:
                break
        
        return False
    
    bbb(0, 0)
    
    if best_rows is not None:
        return best_rows
    return None


def _greedy_row_given_diffs(used_diffs, k):
    """Build a single row greedily given already-used differences."""
    row = [0]
    current_max = 0
    while len(row) < k + 1:
        found = False
        for x in range(current_max + 1, current_max + 2000):
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
    return row

def _try_swap(rows, r_idx, p1, p2, n_rows, k):
    """Try swapping values at positions p1 and p2 in row r_idx. Returns True if successful."""
    row = rows[r_idx]
    v1, v2 = row[p1], row[p2]
    if v1 == v2:
        return False
    # After swap: row[p1]=v2, row[p2]=v1
    # Check that new ordering is still strictly increasing
    new_row = row[:]
    new_row[p1] = v2
    new_row[p2] = v1
    for i in range(len(new_row) - 1):
        if new_row[i] >= new_row[i + 1]:
            return False

    # Compute all diffs, remove old diffs involving p1 or p2, add new ones
    all_diffs = _compute_all_diffs(rows)

    # Old diffs involving p1 or p2
    old_involving = set()
    for i in range(len(row)):
        if i == p1 or i == p2:
            continue
        old_involving.add(abs(row[i] - v1))
        old_involving.add(abs(row[i] - v2))
    old_involving.add(abs(v1 - v2))

    forbidden = all_diffs - old_involving

    # New diffs involving p1 (now v2) or p2 (now v1)
    new_involving = set()
    for i in range(len(row)):
        if i == p1 or i == p2:
            continue
        new_involving.add(abs(new_row[i] - v2))
        new_involving.add(abs(new_row[i] - v1))
    new_involving.add(abs(v1 - v2))

    # Check no new diff collides with forbidden
    if new_involving & forbidden:
        return False

    # Apply swap
    row[p1] = v2
    row[p2] = v1
    return True

def _row_diffs(row):
    """Return the set of all positive differences within a single row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s

def _compute_all_diffs(rows):
    """Return the union of all positive differences across all rows."""
    all_diffs = set()
    for r in rows:
        n = len(r)
        for i in range(n):
            ri = r[i]
            for j in range(i + 1, n):
                all_diffs.add(r[j] - ri)
    return all_diffs

def _find_random_valid_value(rows, r_idx, pos, n_rows, k):
    """Find a random valid value for rows[r_idx][pos] different from current."""
    row = rows[r_idx]
    old_val = row[pos]

    # Compute all diffs once
    all_diffs = _compute_all_diffs(rows)

    diffs_involving_old = set()
    for i in range(len(row)):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)

    forbidden_diffs = all_diffs - diffs_involving_old

    # Search for a new value
    lower_bound = row[pos - 1] + 1
    current_max = max(max(r) for r in rows)
    upper_bound = min(current_max + 30, 250)
    if pos + 1 < len(row):
        upper_bound = min(upper_bound, row[pos + 1] - 1)

    if upper_bound < lower_bound:
        return None

    # Collect valid candidates, bias toward lower values
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue

        valid = True
        new_diffs = set()
        for i in range(len(row)):
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
            valid_candidates.append(new_val)

    if not valid_candidates:
        return None

    # Weighted random choice favoring lower values
    if len(valid_candidates) <= 5:
        return random.choice(valid_candidates)

    max_candidate = max(valid_candidates)
    weights = []
    for v in valid_candidates:
        weight = (max_candidate - v + 1) ** 2
        weights.append(weight)

    total_weight = sum(weights)
    r = random.uniform(0, total_weight)
    cumulative = 0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return valid_candidates[i]
    return valid_candidates[-1]

def _local_search_minimize(rows, n_rows, k, deadline):
    """Greedy local search to minimize max entry with incremental diff tracking."""
    best_rows = [r[:] for r in rows]
    used_diffs = _compute_all_diffs(best_rows)

    # Precompute per-row diff sets for faster updates
    row_diffs = [_row_diffs(best_rows[i]) for i in range(n_rows)]

    while time.time() < deadline - 5:
        progress = False

        # Randomize row order for diversity
        row_order = list(range(n_rows))
        random.shuffle(row_order)

        for r_idx in row_order:
            if time.time() > deadline - 5:
                break

            row = best_rows[r_idx]
            rd = row_diffs[r_idx]

            # Process positions right-to-left
            for pos in range(k, 0, -1):
                old_val = row[pos]

                # Compute diffs involving old_val at this position
                diffs_involving_old = set()
                for i in range(len(row)):
                    if i == pos:
                        continue
                    if i < pos:
                        diffs_involving_old.add(old_val - row[i])
                    else:
                        diffs_involving_old.add(row[i] - old_val)

                forbidden_diffs = used_diffs - diffs_involving_old

                lower_bound = row[pos - 1] + 1
                upper_bound = old_val - 1
                if pos + 1 < len(row):
                    upper_bound = min(upper_bound, row[pos + 1] - 1)

                if upper_bound < lower_bound:
                    continue

                # Search for smallest valid value
                found = None
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(len(row)):
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
                        found = new_val
                        break

                if found is not None and found < old_val:
                    # Update used_diffs and row_diffs incrementally
                    for d in diffs_involving_old:
                        used_diffs.discard(d)
                        rd.discard(d)
                    for d in new_diffs:
                        used_diffs.add(d)
                        rd.add(d)
                    row[pos] = found
                    progress = True

        if not progress:
            break

    return best_rows

def _greedy_construct(n_rows, k):
    """Fast greedy baseline construction with randomization for diversity."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            search_start = current_max + 1
            search_end = current_max + 1500
            for x in range(search_start, search_end):
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
